import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
import matplotlib.cm as cm
import torch
from sklearn.neighbors import NearestNeighbors
import logging
import quaternion as q
import os, imageio


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

def make_spatial_temporal_graph(T, H, W):
    M = np.zeros([T * H * W, T * H * W])
    for t in range(T):
        for j in range(H):
            for i in range(W):
                cur_pos = t * H * W + j * W + i
                #current frame
                #self
                M[cur_pos, t * H * W + j * W + i] = 1
                #left
                if i != 0:
                    M[cur_pos, t * H * W + j * W + i - 1] = 1
                else:
                    M[cur_pos, t * H * W + (j + 1) * W + i - 1] = 1
                #right
                if i != W - 1:
                    M[cur_pos, t * H * W + j * W + i + 1] = 1
                else:
                    M[cur_pos, t * H * W + (j - 1) * W + i + 1] = 1
                #top
                if j != 0:
                    M[cur_pos, t * H * W + (j - 1) * W + i] = 1
                #bottom
                if j != H - 1:
                    M[cur_pos, t * H * W + (j + 1) * W + i] = 1

                #previous frame
                if t > 0:
                    #self
                    M[cur_pos, (t-1) * H * W + j * W + i] = 1
                    #left
                    if i != 0:
                        M[cur_pos, (t-1) * H * W + j * W + i - 1] = 1
                    else:
                        M[cur_pos, (t-1) * H * W + (j + 1) * W + i - 1] = 1
                    #right
                    if i != W - 1:
                        M[cur_pos, (t-1) * H * W + j * W + i + 1] = 1
                    else:
                        M[cur_pos, (t-1) * H * W + (j - 1) * W + i + 1] = 1
                    #top
                    if j != 0:
                        M[cur_pos, (t-1) * H * W + (j - 1) * W + i] = 1
                    #bottom
                    if j != H - 1:
                        M[cur_pos, (t-1) * H * W + (j + 1) * W + i] = 1

                #next frame
                if t < T - 1:
                    #self
                    M[cur_pos, (t+1) * H * W + j * W + i] = 1
                    #left
                    if i != 0:
                        M[cur_pos, (t+1) * H * W + j * W + i - 1] = 1
                    else:
                        M[cur_pos, (t+1) * H * W + (j + 1) * W + i - 1] = 1
                    #right
                    if i != W - 1:
                        M[cur_pos, (t+1) * H * W + j * W + i + 1] = 1
                    else:
                        M[cur_pos, (t+1) * H * W + (j - 1) * W + i + 1] = 1
                    #top
                    if j != 0:
                        M[cur_pos, (t+1) * H * W + (j - 1) * W + i] = 1
                    #bottom
                    if j != H - 1:
                        M[cur_pos, (t+1) * H * W + (j + 1) * W + i] = 1

    return M

def make_spatial_graph(H, W):
    M = np.zeros([H * W, H * W])
    for i in range(W):
        for j in range(H):
            M[j * W + i, j * W + i] = 1  # self
            # left
            if i != 0:
                M[j * W + i, j * W + i - 1] = 1
            else:
                M[j * W + i, (j + 1) * W + i - 1] = 1
            # right
            if i != W - 1:
                M[j * W + i, j * W + i + 1] = 1
            else:
                M[j * W + i, (j - 1) * W + i + 1] = 1
            # top
            if j != 0:
                M[j * W + i, (j - 1) * W + i] = 1
            # bottom
            if j != H - 1:
                M[j * W + i, (j + 1) * W + i] = 1

            # top left
            if i != 0 and j != 0:
                M[j * W + i, (j - 1) * W + i - 1] = 1
            elif i == 0 and j != 0:
                M[j * W + i, j * W + i - 1] = 1
            # top right
            if i != W - 1 and j != 0:
                M[j * W + i, (j - 1) * W + i + 1] = 1
            elif i == W - 1 and j != 0:
                M[j * W + i, (j - 2) * W + i + 1] = 1
            # bottom left
            if i != 0 and j != H - 1:
                M[j * W + i, (j + 1) * W + i - 1] = 1
            elif i == 0 and j != H - 1:
                M[j * W + i, (j + 2) * W + i - 1] = 1
            # bottom right
            if i != W - 1 and j != H - 1:
                M[j * W + i, (j + 1) * W + i + 1] = 1
            elif i == W - 1 and j != H - 1:
                M[j * W + i, j * W + i + 1] = 1
    return M

def save_poioutgif(images, actions, pois, save_name, img_size):
    out_img = images#[0, t]
    fig = plt.figure()
    ax = plt.subplot(aspect='equal')
    ax.grid(False)
    out_img = np.asarray(out_img, dtype="uint8")
    plt.imshow(out_img)
    pois_t = pois#[0, t]
    # front:0, left:1, right:2, back:3
    # pois_t[:, 1:] *= 112
    for jj in range(len(pois_t)):
        if pois_t[jj, 0] == 3:
            continue
        elif pois_t[jj, 0] == 0:
            pois_t[jj, 1] += img_size
            pois_t[jj, 3] += img_size
        elif pois_t[jj, 0] == 2:
            pois_t[jj, 1] += img_size*2
            pois_t[jj, 3] += img_size*2

        plt.gca().add_patch(
            plt.Rectangle((int(pois_t[jj, 1]), int(pois_t[jj, 2])),
                          int((pois_t[jj, 3] - pois_t[jj, 1])),
                          int((pois_t[jj, 4] - pois_t[jj, 2])), fill=False,
                          edgecolor='r', linewidth=2))

        plt.gca().text(pois_t[jj, 1], pois_t[jj, 2],
                       '{:s}'.format(classToString(classes, int(pois_t[jj, -1]))),
                       bbox=dict(facecolor='b', edgecolor='b', alpha=0.3),
                       fontsize=8, color='black', fontweight='bold')

    fig.savefig(save_name + '.png')
    plt.close(fig)

    # for i in range(3):
    #     out_gif.append(out_img)

    # return out_gif


def classToString(_classes, clss):
    return _classes[clss]


def make_spatial_rois(H, W, d):
    return np.stack((np.linspace(0, W, int(W*d)+1)[:-1][None].repeat(int(H*d), 0).flatten(),
                     np.linspace(0, H, int(H*d)+1)[:-1][:, None].repeat(int(W*d), 1).flatten(),
                     np.linspace(0, W, int(W*d)+1)[1:][None].repeat(int(H*d), 0).flatten(),
                     np.linspace(0, H, int(H*d)+1)[1:][:, None].repeat(int(W*d), 1).flatten()), -1)


def as_euler_angles(q):
    n = np.linalg.norm(q)
    alpha_beta_gamma = 2*np.arccos(np.sqrt((q[0]**2 + q[3]**2)/n))
    return alpha_beta_gamma


def uniform_quat(angle1, angle2):
    angle1_euler = (as_euler_angles(angle1) * 180 / np.pi)
    angle2_euler = (as_euler_angles(angle2) * 180 / np.pi)
    return angle1_euler, angle2_euler


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


def get_remain_time(avg_speed, num_remained):
    """
    avg_speed: sec/frame
    """
    remain_time = avg_speed * num_remained
    remain_time_str = "eta: %.0fh %.0fm"%(remain_time // 3600, (remain_time / 60) % 60)
    return remain_time_str