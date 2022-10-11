import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
import matplotlib.cm as cm
import torch
# from sklearn.neighbors import NearestNeighbors
import logging
import quaternion as q
import os, imageio
from argparse import Namespace


def append_to_dict(dict, key, val):
    if key not in dict.keys():
        dict[key] = []
    dict[key].append(val)
    return dict


def padding(lists):
    max_len = np.max([len(ll) for ll in lists])
    if len(lists[0].shape) == 2:
        out = torch.zeros([len(lists), max_len, lists[0].shape[-1]])
    elif len(lists[0].shape) == 1:
        out = torch.zeros([len(lists), max_len])
    for idx, ll in enumerate(lists):
        out[idx][:len(ll)] = ll
    return out


def to_unfolded_cubemap2(src: np.array):
    # src shape: (6, img_h, img_w, ch)
    # 0:'FRONT', 1:'LEFT', 2:'RIGHT', 3:'BACK'
    b, img_h, img_w, ch = src.shape
    assert b == 4
    black = np.zeros_like(src[0])
    top = np.concatenate([black, black, black, black], axis=1)
    middle = np.concatenate(src[[0, 1, 2, 3]], axis=1)
    bottom = np.concatenate([black, black, black, black], axis=1)
    return np.concatenate([top, middle, bottom])


def to_folded_cubemap(src: np.array):
    # src shape: (6, img_h, img_w, ch)
    # 0:'FRONT', 1:'LEFT', 2:'RIGHT', 3:'BACK'
    img_h, img_w, ch = src.shape
    top, middle, bottom = np.split(src, 3, axis=0)
    _, _, out_up, _ = np.split(top, 4, axis=1)
    out_back, out_left, out_front, out_right = np.split(middle, 4, axis=1)
    _, _, out_down, _ = np.split(bottom, 4, axis=1)
    return np.stack([out_back, out_down, out_front, out_left, out_right, out_up])


def bbox_to_cubemap(batch, img_size):
    out = []
    img_h, img_w = img_size
    for loc, arr in batch.items():
        if loc == "back":
            offset_x = 0
            offset_y = img_h - 1
        elif loc == "left":
            offset_x = img_w - 1
            offset_y = img_h - 1
        elif loc == "front":
            offset_x = img_w * 2 - 1
            offset_y = img_h - 1
        elif loc == "right":
            offset_x = img_w * 3 - 1
            offset_y = img_h - 1
        arr_transformed = arr.copy()
        arr_transformed[:, 0] += offset_x
        arr_transformed[:, 1] += offset_y
        arr_transformed[:, 2] += offset_x
        arr_transformed[:, 3] += offset_y
        out.append(arr_transformed)
    return np.concatenate(out)


def to_unfolded_cubemap(src: np.array):
    # src shape: (6, img_h, img_w, ch)
    # 0:'BACK', 1:'DOWN', 2:'FRONT', 3:'LEFT', 4:'RIGHT', 5:'UP'
    b, img_h, img_w, ch = src.shape
    assert b == 6
    # black = np.zeros_like(src[0])
    # top = np.concatenate([black, black, src[5], black], axis=1)
    # middle = np.concatenate(src[[0, 3, 2, 4]], axis=1)
    # bottom = np.concatenate([black, black, src[1], black], axis=1)
    return np.stack(src[[0, 3, 2, 4]]) #back left front right


def get_dist(x, y): #x: input y: target
    return np.sqrt((x[:, :, 0] - y[:, 0, None]) ** 2 + (x[:, :, 2] - y[:, 2, None]) ** 2)


def cam_to_world(rotation, translation):
    T_world_camera = np.eye(4)#.float()#.cuda().float()
    T_world_camera[0:3, 0:3] = rotation #torch.from_numpy(rotation).float().cuda()
    T_world_camera[0:3, 3] = translation
    return T_world_camera


def cam_to_world_tensor(rotation, translation):
    T_world_camera = torch.eye(4).cpu().float()
    T_world_camera[0:3, 0:3] = torch.from_numpy(rotation).float().cpu()
    T_world_camera[0:3, 3] = translation
    return T_world_camera


def cam_to_world_tensor_batch(rotation, translation):
    B = rotation.shape[0]
    T_world_camera = torch.eye(4)[None].repeat(B, 1, 1).float()
    T_world_camera[:, 0:3, 0:3] = torch.from_numpy(rotation).float()
    T_world_camera[:, 0:3, 3] = translation
    return T_world_camera

#
#Calculate the geometric distance between estimated points and original points
#
def geometricDistance(correspondence, h):
    p1 = np.transpose(np.matrix([correspondence[0].item(0), correspondence[0].item(1), 1]))
    estimatep2 = np.dot(h, p1)
    estimatep2 = (1/estimatep2.item(2))*estimatep2

    p2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3), 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)


def classToString(_classes, clss):
    return _classes[clss]

def as_euler_angles(q):
    n = np.linalg.norm(q)
    alpha_beta_gamma = 2*np.arccos(np.sqrt((q[0]**2 + q[3]**2)/n))
    return alpha_beta_gamma


def uniform_quat(angle1, angle2):
    angle1_euler = (as_euler_angles(angle1) * 180 / np.pi)
    angle2_euler = (as_euler_angles(angle2) * 180 / np.pi)
    return angle1_euler, angle2_euler


def calcul_angle(angle1, angle2):
    angle1_euler, angle2_euler = uniform_quat(angle1.cpu().detach().numpy(), angle2.cpu().detach().numpy())
    return (angle1_euler - angle2_euler + 360) % 360


def rotate_xyz(xyz, angle):
    xyz = xyz.reshape(4, -1)
    rot_mtx = np.array([[np.cos(angle * np.pi / 180), 0, np.sin(angle * np.pi / 180), 0], [0, 1, 0, 0], [-np.sin(angle * np.pi / 180), 0, np.cos(angle * np.pi / 180), 0], [0, 0, 0, 1]])
    out = np.matmul(rot_mtx, xyz.reshape(4, -1))
    out = out.reshape(xyz.shape)
    return out


def dict_equal(dict1, dict2):
    assert (set(dict1.keys()) == set(dict2.keys())), "Sets of keys between 2 dictionaries are different."
    for k in dict1.keys():
        assert (type(dict1[k]) == type(dict2[k])), "Type of key '{:s}' if different.".format(k)
        if type(dict1[k]) == np.ndarray:
            assert (dict1[k].dtype == dict2[k].dtype), "Numpy Type of key '{:s}' if different.".format(k)
            assert (np.allclose(dict1[k], dict2[k])), "Value for key '{:s}' do not match.".format(k)
        else:
            assert (dict1[k] == dict2[k]), "Value for key '{:s}' do not match.".format(k)
    return True


def visualize_3d(xyz, rgb=None):
    X1 = xyz[:, :, 0].reshape(-1)
    Y1 = xyz[:, :, 2].reshape(-1)
    Z1 = xyz[:, :, 1].reshape(-1)

    # Draw IM1 to IM2
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    try:
        ax.scatter(X1, Y1, Z1, c=np.reshape(rgb, [-1, 3]) / 255, s=0.5)
    except:
        ax.scatter(X1, Y1, Z1, s=0.5)
    plt.xlabel('x-axis')
    plt.ylabel('z-axis')

    max_range = np.array([X1.max() - X1.min(), Y1.max() - Y1.min(), Z1.max() - Z1.min()]).max()
    Xb1 = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X1.max() + X1.min())
    Yb1 = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y1.max() + Y1.min())
    Zb1 = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z1.max() + Z1.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb1, Yb1, Zb1):
        ax.plot([xb], [yb], [zb], 'w')

    plt.grid()
    plt.show()

def get_camera_matrix(width, height, hfov, vfov=90):
    """Returns a camera matrix from image size and fov."""
    xc = (width - 1.0) / 2.0
    zc = (height - 1.0) / 2.0
    f = (width / 2.0) / np.tan(np.deg2rad(hfov / 2.0))
    fx = (width / 2.0) / np.tan(np.deg2rad(hfov / 2.0))
    fz = (height / 2.0) / np.tan(np.deg2rad(vfov / 2.0))
    camera_matrix = {"xc": xc, "zc": zc, "f": f, "fx": fx, "fz":fz}
    camera_matrix = Namespace(**camera_matrix)
    return camera_matrix

def get_point_cloud_from_z_panoramic_with_worldpose(Y, camera_matrix, cam_angles, scale, img_width, current_position, current_rotation, num_camera = 12):
    """Projects the depth image Y into a 3D point cloud.
    Inputs:
        Y is ...xHxW
        camera_matrix
    Outputs:
        X is positive going right
        Y is positive into the image
        Z is positive up in the image
        XYZ is ...xHxWx3
    """
    cam_lens1 = np.linspace(0, img_width, num_camera + 1)[:-1].astype(np.int32)
    cam_lens2 = np.linspace(0, img_width, num_camera + 1)[1:].astype(np.int32)
    point_cloud = []
    for cam_len1, cam_len2, cam_angle in zip(cam_lens1, cam_lens2, cam_angles):
        Y_ = Y[:, cam_len1:cam_len2] # H x W_c
        x, z = np.meshgrid(np.arange(Y_.shape[-1]), np.arange(Y_.shape[-2] - 1, -1, -1)) #0~W_c and H~0
        X = (x[::scale, ::scale] - camera_matrix.xc) / camera_matrix.f * Y_[::scale, ::scale]
        Z = (z[::scale, ::scale] - camera_matrix.zc) / camera_matrix.f * Y_[::scale, ::scale]
        XYZ = np.concatenate(
            (X[..., np.newaxis], Y_[::scale, ::scale][..., np.newaxis], Z[..., np.newaxis]),
            axis=X.ndim,
        )
        XYZ_shape = XYZ.shape
        XYZ = XYZ.reshape(-1, 3)
        XYZ_homo = np.stack([XYZ[...,0], XYZ[...,2], -XYZ[...,1], np.ones([len(XYZ)])])
        sensor_rot = q.as_euler_angles(current_rotation)[1] + (cam_angle - np.pi)
        Tcw = cam_to_world(q.as_rotation_matrix(q.from_euler_angles([0,sensor_rot,0])), current_position + np.array([0, 0.88, 0]))
        XYZ = np.matmul(Tcw, XYZ_homo)[:3].transpose(1,0).reshape(XYZ_shape)
        point_cloud.append(XYZ)
    point_cloud = np.concatenate(point_cloud, 1)
    return point_cloud

def get_point_cloud_from_z_panoramic(Y, img_width, num_camera, cam_angles, camera_matrix, scale=1):
    """Projects the depth image Y into a 3D point cloud.
    Inputs:
        Y is ...xHxW
        camera_matrix
    Outputs:
        X is positive going right
        Y is positive into the image
        Z is positive up in the image
        XYZ is ...xHxWx3
    """
    cam_lens1 = np.linspace(0, img_width, num_camera+1)[:-1].astype(np.int32)
    cam_lens2 = np.linspace(0, img_width, num_camera+1)[1:].astype(np.int32)
    point_cloud = []
    for cam_len1, cam_len2, cam_angle in zip(cam_lens1, cam_lens2, cam_angles):
        x, z = np.meshgrid(np.arange(Y[:,cam_len1:cam_len2].shape[-1]), np.arange(Y[:,cam_len1:cam_len2].shape[-2] - 1, -1, -1))
        X = (x[::scale, ::scale] - camera_matrix.xc) * Y[:,cam_len1:cam_len2][::scale, ::scale] / camera_matrix.fx
        Z = (z[::scale, ::scale] - camera_matrix.zc) * Y[:,cam_len1:cam_len2][::scale, ::scale] / camera_matrix.fz
        XYZ = np.concatenate(
            (X[..., np.newaxis], -Y[:,cam_len1:cam_len2][::scale, ::scale][..., np.newaxis], Z[..., np.newaxis]),
            axis=X.ndim,
        )
        if cam_angle > np.pi:
            cam_angle -= 2 * np.pi
        R = np.array([[np.cos(cam_angle), -np.sin(cam_angle), 0],
                      [np.sin(cam_angle), np.cos(cam_angle),0],
                      [0,0,1]])
        XYZ = np.matmul(XYZ, R)
        point_cloud.append(XYZ)
    point_cloud = np.concatenate(point_cloud, 1)[:,:,[0,2,1]]
    return point_cloud


def get_point_cloud_from_z(Y, camera_matrix, scale=1):
    """Projects the depth image Y into a 3D point cloud.
    Inputs:
        Y is ...xHxW
        camera_matrix
    Outputs:
        X is positive going right
        Y is positive into the image
        Z is positive up in the image
        XYZ is ...xHxWx3
    """
    x, z = np.meshgrid(np.arange(Y.shape[-1]), np.arange(Y.shape[-2] - 1, -1, -1))
    for i in range(Y.ndim - 2):
        x = np.expand_dims(x, axis=0)
        z = np.expand_dims(z, axis=0)
    X = (x[::scale, ::scale] - camera_matrix.xc) * Y[::scale, ::scale] / camera_matrix.f
    Z = (z[::scale, ::scale] - camera_matrix.zc) * Y[::scale, ::scale] / camera_matrix.f
    XYZ = np.concatenate(
        (X[..., np.newaxis], Y[::scale, ::scale][..., np.newaxis], Z[..., np.newaxis]),
        axis=X.ndim,
    )
    XYZ = XYZ[...,[0, 2, 1]]
    XYZ[...,-1] = -XYZ[...,-1]
    return XYZ

