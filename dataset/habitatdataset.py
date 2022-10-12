import torch.utils.data as data
import numpy as np
import joblib
import torch
import quaternion as q
import torchvision.transforms as transforms
from PIL import Image
from torchvision.ops import nms as torch_nms
import cv2
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from utils.debug_utils import log_time
from os import path as osp


from habitat_sim.sensors.noise_models import redwood_depth_noise_model
from habitat_sim.sensors.noise_models.redwood_depth_noise_model import (
    RedwoodNoiseModelCPUImpl,
)
TIME_DEBUG = False

def get_haveseen(all_object_positions, all_object_categories, object_pose, object_category):
    have_seen = np.zeros(len(object_pose))
    if len(all_object_positions) > 0:
        dists = np.linalg.norm(np.array(all_object_positions)[None] - np.array(object_pose)[:, None], axis=-1)
        for i in range(len(object_pose)):
            is_same = (dists[i].min(-1) < 0.2) & (all_object_categories == object_category[i])
            if is_same.any():
                have_seen[i] = 1
    have_seen = np.array(have_seen).astype(np.float32)
    return have_seen

class ILDataset(data.Dataset):
    def __init__(self, cfg, data_list, transform):
        self.data_list = data_list
        self.img_size = cfg.IMG_SHAPE
        self.action_dim = 4
        self.max_input_length = cfg.max_input_length
        self.config = cfg
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        eval_augmentation = [
            transforms.ToTensor(),
            normalize
            ]
        self.transform_eval = transforms.Compose(eval_augmentation)
        self.transform = transform
        self.redwood_depth_noise = RedwoodNoiseModelCPUImpl(
            np.load(
                osp.join(
                    osp.dirname(redwood_depth_noise_model.__file__),
                    "data",
                    "redwood-depth-dist-model.npy",
                )
            ),
            noise_multiplier=0.5,
        )

    def __getitem__(self, index):
        if TIME_DEBUG: s = log_time()
        res = self.pull_image(index)
        if TIME_DEBUG : s, get_step_t = log_time(s, 'get data', return_time=True)
        return res

    def __len__(self):
        return len(self.data_list)

    def get_dist(self, input_position):
        return np.linalg.norm(input_position[-1] - input_position[0], ord=2)

    def pull_image(self, index):
        if TIME_DEBUG: s = log_time()
        try:
            input_data = joblib.load(self.data_list[index])
        except:
            print(self.data_list[index])
            print("Data loading error")
        if TIME_DEBUG : s, get_step_t = log_time(s, 'load data', return_time=True)
        scene = self.data_list[index].split('/')[-1].split('_')[0]
        target_indices = np.array(input_data['target_idx'])
        aux_info = {'have_been': None, 'distance': None, 'have_seen': None}

        orig_data_len = len(input_data['position'])
        start_idx = np.random.randint(orig_data_len - 10) if orig_data_len > 10 else 0
        end_idx = - 1

        input_rgb = np.array(input_data['rgb'][start_idx:end_idx], dtype=np.float32)
        input_length = np.minimum(len(input_rgb), self.max_input_length)
        input_rgb_out = np.zeros([self.max_input_length, *input_rgb.shape[1:]])
        input_rgb_out[:input_length] = input_rgb[:input_length]

        input_rgb_transformed = []
        for i in range(len(input_rgb)):
            im = Image.fromarray(np.uint8(input_rgb[i]))
            input_rgb_transformed.append(self.transform(im))
        input_rgb_transformed = torch.stack([input_rgb_transformed[i] for i in range(len(input_rgb))], 0)
        input_rgb_transformed_out = np.zeros([self.max_input_length, *input_rgb_transformed.shape[1:]])
        input_rgb_transformed_out[:input_length] = input_rgb_transformed[:input_length]

        input_rgb_eval_transformed = []
        for i in range(len(input_rgb)):
            im = Image.fromarray(np.uint8(input_rgb[i]))
            input_rgb_eval_transformed.append(self.transform_eval(im))
        input_rgb_eval_transformed = torch.stack([input_rgb_eval_transformed[i] for i in range(len(input_rgb))], 0)
        input_rgb_eval_transformed_out = np.zeros([self.max_input_length, *input_rgb_eval_transformed.shape[1:]])
        input_rgb_eval_transformed_out[:input_length] = input_rgb_eval_transformed[:input_length]

        input_dep = np.array(input_data['depth'][start_idx:end_idx], dtype=np.float32)
        input_dep_out = np.zeros([self.max_input_length, *input_dep.shape[1:]])
        input_dep_out[:input_length] = input_dep[:input_length]

        input_act = np.array(input_data['action'][start_idx:start_idx+input_length], dtype=np.int8)
        input_act_out = np.ones([self.max_input_length]) * (-100)
        input_act_out[:input_length] = input_act -1 if self.action_dim == 3 else input_act

        max_num_object = self.config.memory.num_objects
        have_seen = np.zeros([self.max_input_length, max_num_object])
        input_object = input_data['object'][start_idx:start_idx + input_length]
        if 'object_id' in input_data:
            input_object_id = input_data['object_id'][start_idx:start_idx + input_length]
        input_object_category = input_data['object_category'][start_idx:start_idx + input_length]
        if 'object_pose' in input_data:
            input_object_pose = input_data['object_pose'][start_idx:start_idx + input_length]
        if 'object_score' in input_data:
            input_object_score = input_data['object_score'][start_idx:start_idx + input_length]
        else:
            input_object_score = np.ones([len(input_object_pose), 300])

        input_object_out = np.zeros((self.max_input_length, max_num_object, 5))
        input_object_score_out = np.zeros((self.max_input_length, max_num_object))
        input_object_category_out = np.ones((self.max_input_length, max_num_object)) * (-1)
        input_object_mask_out = np.zeros((self.max_input_length, max_num_object))
        input_object_pose_out = -100. * np.ones((self.max_input_length, max_num_object, 3))
        input_object_id_out = np.ones((self.max_input_length, max_num_object)) * (-1)
        input_object_relpose_out = -100. * np.ones((self.max_input_length, max_num_object, 2))
        for i in range(len(input_object)):
            if len(input_object[i]) > 0:
                input_object_t = np.array(input_object[i]).reshape(-1, 4)
                input_object_score_t = np.array(input_object_score[i]).reshape(-1)
                input_object_score_t = input_object_score_t[:len(input_object_t)]
                keep = torch_nms(torch.from_numpy(input_object_t).float(), torch.from_numpy(input_object_score_t).float(), 0.5)
                input_object_t = np.array(input_object_t[keep]).reshape(-1, 4)
                input_object_score_t = np.array(input_object_score_t[keep]).reshape(-1)
                input_object_category_t = np.array(input_object_category[i]).reshape(-1)[keep].reshape(-1)
                if 'object_id' in input_data:
                    input_object_id_t = np.array(input_object_id[i]).reshape(-1)[keep].reshape(-1)
                if 'object_pose' in input_data:
                    input_object_pose_t = np.array(input_object_pose[i]).reshape(-1, 3)
                    input_object_pose_t = np.array(input_object_pose_t[keep]).reshape(-1, 3)
                num_object_t = len(input_object_t)

                if self.config.scene_data == "mp3d":
                    size = (input_object_t[:, 2]-input_object_t[:, 0])*(input_object_t[:, 3]-input_object_t[:, 1])
                    idx = np.argsort(-size)
                    input_object_out[i, :min(max_num_object, num_object_t), 1:] = input_object_t[idx][:min(max_num_object, num_object_t), :4]
                    input_object_score_out[i, :min(max_num_object, num_object_t)] = input_object_score_t[idx][:min(max_num_object, num_object_t)]
                    input_object_category_out[i, :min(max_num_object, num_object_t)] = input_object_category_t[idx][:min(max_num_object, num_object_t)]

                    if 'object_id' in input_data:
                        input_object_id_out[i, :min(max_num_object, num_object_t)] = input_object_id_t[:min(max_num_object, num_object_t)]
                    if 'object_pose' in input_data:
                        input_object_pose_out[i, :min(max_num_object, num_object_t)] = input_object_pose_t[idx][:min(max_num_object, num_object_t)]
                    input_object_mask_out[i, :min(max_num_object, num_object_t)] = 1
                else:
                    input_object_out[i, :min(max_num_object, num_object_t), 1:] = input_object_t[:min(max_num_object, num_object_t), :4]
                    input_object_score_out[i, :min(max_num_object, num_object_t)] = input_object_score_t[:min(max_num_object, num_object_t)]
                    input_object_category_out[i, :min(max_num_object, num_object_t)] = input_object_category_t[:min(max_num_object, num_object_t)]
                    if 'object_id' in input_data:
                        input_object_id_out[i, :min(max_num_object, num_object_t)] = input_object_id_t[:min(max_num_object, num_object_t)]
                    if 'object_pose' in input_data:
                        input_object_pose_out[i, :min(max_num_object, num_object_t)] = input_object_pose_t[:min(max_num_object, num_object_t)]
                    input_object_mask_out[i, :min(max_num_object, num_object_t)] = 1

                if i > 10:
                    have_seen_i = get_haveseen(np.concatenate([input_object_pose_out[i_] for i_ in range(i)]), np.concatenate([input_object_category_out[i_] for i_ in range(i)]), input_object_pose_out[i],
                                             input_object_category_out[i])
                    have_seen[i] = have_seen_i
                    # dists = np.linalg.norm(input_object_pose_out[i][:,None] - np.concatenate([input_object_pose_out[i_] for i_ in range(i)])[None], axis=-1)
                    # have_seen[i, dists.min(-1) < 0.2] = 1

        targets = np.zeros([self.max_input_length])
        targets[:input_length] = target_indices[start_idx:start_idx+input_length]
        target_img = np.stack(input_data['target_img'])[:,:,:,:4]
        B, H, W, C = target_img.shape

        target_img_orig_out = np.zeros([5, *target_img[0].shape[:2], 3])
        target_img_orig_out[:len(target_img)] = np.stack(target_img,0)[...,:3] * 255.

        target_pose = np.stack(input_data['target_pose']) #targets[:, t].long()
        target_pose = target_pose[target_indices[start_idx:start_idx+input_length].astype(np.int32)]
        target_pose_out = np.zeros([self.max_input_length, 3])
        target_pose_out[:input_length] = target_pose

        relpose = np.zeros([self.max_input_length,3])
        target_img_out = np.zeros([5, *target_img[0].shape])
        target_img_out[:len(target_img)] = target_img

        target_img_transformed = []
        for i in range(len(target_img)):
            im = Image.fromarray(np.uint8(target_img[i][:,:,:3]* 255.))
            target_img_transformed.append(torch.cat([self.transform(im), torch.from_numpy(target_img[i][:,:,-1][None])], 0))
        target_img_transformed = torch.stack(target_img_transformed, 0)
        target_img_transformed_out = np.zeros([5, *target_img_transformed[0].shape])
        target_img_transformed_out[:len(target_img_transformed)] = target_img_transformed

        target_loc_object_out = np.zeros((5, max_num_object, 5))
        target_loc_object_category_out = np.zeros((5, max_num_object))
        target_loc_object_mask_out = np.zeros((5, max_num_object))
        target_loc_object_pose_out = np.zeros((5, max_num_object, 3))
        target_loc_object_id_out = np.zeros((5, max_num_object))
        target_loc_object_score_out = np.zeros((5, max_num_object))

        if 'target_loc_object' in input_data:
            target_loc_object = input_data['target_loc_object']
            target_loc_object_score = input_data['target_loc_object_score']
            target_loc_object_category = input_data['target_loc_object_category']
            if 'target_loc_object_id' in input_data:
                target_loc_object_id = input_data['target_loc_object_id']
            if 'target_loc_object_pose' in input_data:
                target_loc_object_pose = input_data['target_loc_object_pose']
            for i in range(len(target_loc_object)):
                if len(target_loc_object[i]) > 0:
                    target_loc_object_t = np.array(target_loc_object[i]).reshape(-1, 4)
                    target_loc_object_t[:, 0] = np.maximum(0, target_loc_object_t[:, 0]-5)
                    target_loc_object_t[:, 1] = np.maximum(0, target_loc_object_t[:, 1]-5)
                    target_loc_object_t[:, 2] = np.minimum(W, target_loc_object_t[:, 2]+5)
                    target_loc_object_t[:, 3] = np.minimum(H, target_loc_object_t[:, 3]+5)
                    target_loc_object_category_t = np.array(target_loc_object_category[i]).reshape(-1)
                    target_loc_object_score_t = np.array(target_loc_object_score[i]).reshape(-1)
                    if 'target_loc_object_id' in input_data:
                        target_loc_object_id_t = np.array(target_loc_object_id[i]).reshape(-1)
                    if 'target_loc_object_pose' in input_data:
                        target_loc_object_pose_t = np.array(target_loc_object_pose[i]).reshape(-1, 3)
                    num_object_t = len(target_loc_object_t)
                    if self.config.scene_data == "mp3d":
                        size = (target_loc_object_t[:, 2]-target_loc_object_t[:, 0])*(target_loc_object_t[:, 3]-target_loc_object_t[:, 1])
                        idx = np.argsort(-size)
                        target_loc_object_out[i, :min(max_num_object, num_object_t), 1:] = target_loc_object_t[idx][:min(max_num_object, num_object_t), :4]
                        target_loc_object_score_out[i, :min(max_num_object, num_object_t), 1:] = target_loc_object_score_t[idx][:min(max_num_object, num_object_t), :4]
                        target_loc_object_category_out[i, :min(max_num_object, num_object_t)] = target_loc_object_category_t[idx][:min(max_num_object, num_object_t)]
                        target_loc_object_mask_out[i, :min(max_num_object, num_object_t)] = 1
                        if 'target_loc_object_pose' in input_data:
                            target_loc_object_pose_out[i, :min(max_num_object, num_object_t)] = target_loc_object_pose_t[:min(max_num_object, num_object_t)]
                    else:
                        target_loc_object_out[i, :min(max_num_object, num_object_t), 1:] = target_loc_object_t[:min(max_num_object, num_object_t), :4]
                        target_loc_object_score_out[i, :min(max_num_object, num_object_t), 1:] = target_loc_object_score_t[idx][:min(max_num_object, num_object_t), :4]
                        target_loc_object_category_out[i, :min(max_num_object, num_object_t)] = target_loc_object_category_t[:min(max_num_object, num_object_t)]
                        target_loc_object_mask_out[i, :min(max_num_object, num_object_t)] = 1
                        if 'target_loc_object_id' in input_data:
                            target_loc_object_id_out[i, :min(max_num_object, num_object_t)] = target_loc_object_id_t[:min(max_num_object, num_object_t)]
                        if 'target_loc_object_pose' in input_data:
                            target_loc_object_pose_out[i, :min(max_num_object, num_object_t)] = target_loc_object_pose_t[:min(max_num_object, num_object_t)]

        target_object_out = np.zeros((5, max_num_object, 5))
        target_object_category_out = np.zeros((5, max_num_object))
        target_object_mask_out = np.zeros((5, max_num_object))
        target_object_pose_out = np.zeros((5, max_num_object, 3))
        target_object_id_out = np.zeros((5, max_num_object))

        if 'target_object' in input_data:
            target_object = input_data['target_object']
            target_object_category = input_data['target_object_category']
            if 'target_object_id' in input_data:
                target_object_id = input_data['target_object_id']
            if 'target_object_pose' in input_data:
                target_object_pose = input_data['target_object_pose']
            for i in range(len(target_object)):
                if len(target_object[i]) > 0:
                    target_object_t = np.array(target_object[i]).reshape(-1, 4)
                    target_object_t[:, 0] = np.maximum(0, target_object_t[:, 0]-5)
                    target_object_t[:, 1] = np.maximum(0, target_object_t[:, 1]-5)
                    target_object_t[:, 2] = np.minimum(W, target_object_t[:, 2]+5)
                    target_object_t[:, 3] = np.minimum(H, target_object_t[:, 3]+5)
                    target_object_category_t = np.array(target_object_category[i]).reshape(-1)
                    if 'target_object_id' in input_data:
                        target_object_id_t = np.array(target_object_id[i]).reshape(-1)
                    if 'target_object_pose' in input_data:
                        target_object_pose_t = np.array(target_object_pose[i]).reshape(-1, 3)
                    num_object_t = len(target_object_t)
                    if self.config.scene_data == "mp3d":
                        size = (target_object_t[:, 2]-target_object_t[:, 0])*(target_object_t[:, 3]-target_object_t[:, 1])
                        idx = np.argsort(-size)
                        target_object_out[i, :min(max_num_object, num_object_t), 1:] = target_object_t[idx][:min(max_num_object, num_object_t), :4]
                        target_object_category_out[i, :min(max_num_object, num_object_t)] = target_object_category_t[idx][:min(max_num_object, num_object_t)]
                        target_object_mask_out[i, :min(max_num_object, num_object_t)] = 1
                        if 'target_object_pose' in input_data:
                            target_object_pose_out[i, :min(max_num_object, num_object_t)] = target_object_pose_t[:min(max_num_object, num_object_t)]
                    else:
                        target_object_out[i, :min(max_num_object, num_object_t), 1:] = target_object_t[:min(max_num_object, num_object_t), :4]
                        target_object_category_out[i, :min(max_num_object, num_object_t)] = target_object_category_t[:min(max_num_object, num_object_t)]
                        target_object_mask_out[i, :min(max_num_object, num_object_t)] = 1
                        if 'target_object_id' in input_data:
                            target_object_id_out[i, :min(max_num_object, num_object_t)] = target_object_id_t[:min(max_num_object, num_object_t)]
                        if 'target_object_pose' in input_data:
                            target_object_pose_out[i, :min(max_num_object, num_object_t)] = target_object_pose_t[:min(max_num_object, num_object_t)]

        positions = np.zeros([self.max_input_length,3])
        positions[:input_length] = input_data['position'][start_idx:start_idx+input_length]

        rotation = q.as_euler_angles(np.array(q.from_float_array(input_data['rotation'])))[:,1]
        rotations = np.zeros([self.max_input_length])
        rotations[:input_length] = rotation[start_idx:start_idx+input_length]

        vis_rotations = np.zeros([self.max_input_length, 4])
        vis_rotations[:input_length] = np.stack(input_data['rotation'][start_idx:start_idx+input_length])

        have_been = np.zeros([self.max_input_length])
        pp = input_data['position'][start_idx:end_idx][:input_length]
        for idx, pos_t in enumerate(pp):
            if idx == 0:
                have_been[idx] = 0
            else:
                dists = np.linalg.norm(pp[:idx] - pos_t, axis=1)
                if len(dists) > 10:
                    far = np.where(dists > 1.0)[0]
                    near = np.where(dists[:-10] < 1.0)[0]
                    if len(far) > 0 and len(near) > 0 and (near < far.max()).any():
                        have_been[idx] = 1
                    else:
                        have_been[idx] = 0
                else:
                    have_been[idx] = 0

        aux_info['distance'] = np.zeros([self.max_input_length])
        distances = np.stack(input_data['distance'][start_idx:start_idx+input_length])
        aux_info['distance'][:input_length] = torch.from_numpy(distances).float()
        episode_length = np.stack([input_data['distance'][0]] + [np.stack(input_data['distance'])[i+1] for i in np.where(np.stack(input_data['action'])==0)[0][:-1]])
        episode_idx = np.stack([i+1 for i in np.where(np.stack(input_data['action'])==0)[0]])[:-1]
        max_length = []
        epi_idx = 0
        for i in range(len(input_data['distance'])):
            if i in episode_idx:
                epi_idx += 1
            max_length.append(episode_length[epi_idx])
        progress = np.clip(1 - np.array(np.stack(input_data['distance'])/np.stack(max_length))[start_idx:start_idx+input_length],0,1)
        is_img_target = np.stack(input_data['distance'])[start_idx:start_idx+input_length] < 1

        train_info = {}
        train_info["panoramic_rgb"] = torch.from_numpy(input_rgb_out).float()
        train_info["panoramic_rgb_trans"] = torch.from_numpy(input_rgb_transformed_out).float()
        train_info["panoramic_rgb_eval"] = torch.from_numpy(input_rgb_eval_transformed_out).float()
        train_info["panoramic_depth"] = torch.from_numpy(input_dep_out).float()
        train_info["relpose"] = torch.from_numpy(relpose).float()
        train_info["action"] = torch.from_numpy(input_act_out).float()
        train_info["object"] = torch.from_numpy(input_object_out).float()
        train_info["object_mask"] = torch.from_numpy(input_object_mask_out).float()
        train_info["object_score"] = torch.from_numpy(input_object_score_out).float()
        train_info["object_category"] = torch.from_numpy(input_object_category_out).float()
        train_info["object_id"] = torch.from_numpy(input_object_id_out).float()
        train_info["object_pose"] = torch.from_numpy(input_object_pose_out).float()
        train_info["object_relpose"] = torch.from_numpy(input_object_relpose_out).float()
        train_info["position"] = torch.from_numpy(positions).float()
        train_info["rotation"] = torch.from_numpy(rotations).float()
        train_info["target"] = targets
        # train_info["target_goal_orig"] = torch.from_numpy(target_img_orig_out[targets.astype(np.int32)]).float()
        train_info["target_goal"] = torch.from_numpy(target_img_out[targets.astype(np.long)]).float()
        train_info["target_goal_trans"] = torch.from_numpy(target_img_transformed_out[targets.astype(np.int32)]).float()

        train_info["target_object"] = torch.from_numpy(target_object_out[targets.astype(np.int32)]).float()
        train_info["target_object_category"] = torch.from_numpy(target_object_category_out[targets.astype(np.int32)]).float()
        train_info["target_object_mask"] = torch.from_numpy(target_object_mask_out[targets.astype(np.int32)]).float()
        train_info["target_object_pose"] = torch.from_numpy(target_object_pose_out[targets.astype(np.int32)]).float()
        train_info["target_object_id"] = torch.from_numpy(target_object_id_out[targets.astype(np.int32)]).float()
        train_info["target_loc_object"] = torch.from_numpy(target_loc_object_out[targets.astype(np.int32)]).float()
        train_info["target_loc_object_category"] = torch.from_numpy(target_loc_object_category_out[targets.astype(np.int32)]).float()
        train_info["target_loc_object_score"] = torch.from_numpy(target_loc_object_score_out[targets.astype(np.int32)]).float()
        train_info["target_loc_object_mask"] = torch.from_numpy(target_loc_object_mask_out[targets.astype(np.int32)]).float()
        train_info["target_loc_object_pose"] = torch.from_numpy(target_loc_object_pose_out[targets.astype(np.int32)]).float()
        train_info["target_loc_object_id"] = torch.from_numpy(target_loc_object_id_out[targets.astype(np.int32)]).float()
        train_info["scene"] = scene
        is_target = np.stack([(target_object_id_out[targets.astype(np.int32)][i] == input_object_id_out[i]).astype(np.float32) for i in range(input_object_id_out.shape[0])])
        aux_info['progress'] = np.zeros([self.max_input_length])
        aux_info['progress'][:input_length] = torch.from_numpy(progress).float()
        aux_info['have_been'] = torch.from_numpy(have_been).float()
        aux_info['have_seen'] = torch.from_numpy(have_seen).float()
        aux_info['is_obj_target'] = torch.from_numpy(is_target).float()
        aux_info['is_img_target'] = np.zeros([self.max_input_length])
        aux_info['is_img_target'][:input_length] = torch.from_numpy(is_img_target).float()

        vis_info = {}
        vis_info["target_position"] = target_pose_out
        vis_info["start_position"] = input_data['position'][start_idx]
        vis_info["house"] = scene
        vis_info["position"] = positions
        vis_info["rotation"] = vis_rotations
        vis_info["len_data"] = len(input_data['position'][start_idx:start_idx+input_length])
        vis_info["input_image"] = input_rgb_out
        vis_info["target_goal"] = target_img_orig_out[targets.astype(np.int32)]
        vis_info["have_been"] = aux_info['have_been']
        vis_info["progress"] = aux_info['progress']

        if self.config.TASK_CONFIG.TASK.TASK_NAME == "ImgGoal" and train_info['target_loc_object'].sum() == 0:
            train_info['target_loc_object'] = train_info['target_object']
            train_info['target_loc_object_mask'] = train_info['target_object_mask']
            train_info['target_loc_object_category'] = train_info['target_object_category']
            train_info["target_loc_object_pose"] = train_info['target_object_pose']
            train_info["target_loc_object_id"] = train_info['target_object_id']

        vis_info["object"] = torch.from_numpy(input_object_out).float()
        if TIME_DEBUG : s, get_step_t = log_time(s, 'process data', return_time=True)
        train_info['data_path'] = self.data_list[index]
        return [train_info, aux_info, vis_info]
