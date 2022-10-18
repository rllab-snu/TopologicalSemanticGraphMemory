import os
import numpy as np
from PIL import Image
from typing import Optional
from habitat import Config, Dataset
from gym.spaces.box import Box
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from env_utils.imagegoal_env import ImageGoalEnv
# from model.Graph.resnet_img import resnet18 as resnet18_img
from torchvision.models import resnet18 as resnet18_img
from model.Graph.resnet_obj import resnet18 as resnet18_obj
from model.Graph.graph import ImgGraph, ObjGraph
from model.Graph.graph_update import update_image_graph, update_object_graph
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ImageGoalGraphEnv(ImageGoalEnv):
    metadata = {'render.modes': ['rgb_array']}
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config, dataset)
        self.config = config
        self.scene_data = config.scene_data
        self.input_shape = config.IMG_SHAPE
        self.object_feature_dim = config.features.object_feature_dim
        self.num_objects = config.memory.num_objects
        self.feature_dim = config.memory.img_embedding_dim
        self.torch_device = 'cuda:' + str(config.TORCH_GPU_ID) if torch.cuda.device_count() > 0 else 'cpu'
        self.transform_eval = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.img_node_th = config.TASK_CONFIG.img_node_th
        self.obj_node_th = config.TASK_CONFIG.obj_node_th
        self.imggraph = ImgGraph(config)
        self.objgraph = ObjGraph(config)

        self.img_encoder = self.load_img_encoder(config.memory.img_embedding_dim)
        self.obj_encoder = self.load_obj_encoder(config.features.object_feature_dim)

        self.reset_all_memory()
        
        if self.args.record > 0:
            global VIDEO_DIR
            VIDEO_DIR = os.path.join(self.args.record_dir, self.args.version, 'video')
            os.makedirs(VIDEO_DIR, exist_ok=True)
            self.record_iter = -1

        if self.args.record > 1:
            global OTHER_DIR
            OTHER_DIR = os.path.join(self.args.record_dir, self.args.version, 'others')
            os.makedirs(OTHER_DIR, exist_ok=True)

        if self.args.mode != "train_rl" and self.args.mode != "eval" and self.args.mode != "collect":
            return

        self.dn = config.TASK_CONFIG.DATASET.DATASET_NAME.split("_")[0]
        self.observation_space.spaces.update({
            'img_memory_feat': Box(low=-np.Inf, high=np.Inf, shape=(self.imggraph.M, self.feature_dim), dtype=np.float32),
            'img_memory_pose': Box(low=-np.Inf, high=np.Inf, shape=(self.imggraph.M, 3), dtype=np.float32),
            'img_memory_mask': Box(low=0, high=1, shape=(self.imggraph.M,), dtype=np.bool),
            'img_memory_A': Box(low=0, high=1, shape=(self.imggraph.M, self.imggraph.M), dtype=np.bool),
            'img_memory_idx': Box(low=-np.Inf, high=np.Inf, shape=(), dtype=np.float32),
            'img_memory_time': Box(low=-np.Inf, high=np.Inf, shape=(self.imggraph.M,), dtype=np.float32)
        })
        self.observation_space.spaces.update({
            'obj_memory_feat': Box(low=-np.Inf, high=np.Inf, shape=(self.objgraph.M, self.object_feature_dim), dtype=np.float32),
            'obj_memory_score': Box(low=-np.Inf, high=np.Inf, shape=(self.objgraph.M,), dtype=np.float32),
            'obj_memory_pose': Box(low=-np.Inf, high=np.Inf, shape=(self.objgraph.M, 3), dtype=np.float32),
            'obj_memory_category': Box(low=-np.Inf, high=np.Inf, shape=(self.objgraph.M,), dtype=np.float32),
            'obj_memory_mask': Box(low=0, high=1, shape=(self.objgraph.M,), dtype=np.bool),
            'obj_memory_A_OV': Box(low=0, high=1, shape=(self.objgraph.M, self.objgraph.MV), dtype=np.bool),
            'obj_memory_time': Box(low=-np.Inf, high=np.Inf, shape=(self.objgraph.M,), dtype=np.float32)
        })

    def load_img_encoder(self, feature_dim):
        img_encoder = resnet18_img(num_classes=feature_dim)
        dim_mlp = img_encoder.fc.weight.shape[1]
        img_encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), img_encoder.fc)
        ckpt_pth = os.path.join(project_dir, 'data/graph', self.dn, 'Img_encoder.pth.tar')
        ckpt = torch.load(ckpt_pth, map_location='cpu')
        state_dict = {k[len('module.encoder_q.'):]: v for k, v in ckpt['state_dict'].items() if 'module.encoder_q.' in k}
        img_encoder.load_state_dict(state_dict)
        img_encoder.eval().to(self.torch_device)

        # img_encoder = resnet18_img(num_classes=feature_dim)
        # dim_mlp = img_encoder.fc.weight.shape[1]
        # img_encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), img_encoder.fc)
        # ckpt_pth = os.path.join(project_dir, 'data/graph', self.dn, 'Img_encoder.pth.tar')
        # ckpt = torch.load(ckpt_pth, map_location='cpu')
        # img_encoder.load_state_dict(ckpt)
        # img_encoder.eval().to(self.torch_device)
        return img_encoder

    def load_obj_encoder(self, feature_dim):
        obj_encoder = resnet18_obj(num_classes=feature_dim)
        dim_mlp = obj_encoder.fc.weight.shape[1]
        obj_encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), obj_encoder.fc)
        ckpt_pth = os.path.join(project_dir, 'data/graph', self.dn, f'Obj_encoder.pth.tar')
        ckpt = torch.load(ckpt_pth, map_location='cpu')
        state_dict = {k[len('module.encoder_q.'):]: v for k, v in ckpt['state_dict'].items() if 'module.encoder_q.' in k}
        obj_encoder.load_state_dict(state_dict)
        obj_encoder.eval().to(self.torch_device)
        return obj_encoder

    def reset_all_memory(self):
        self.imggraph.reset()
        self.objgraph.reset()

    def is_close(self, embed_a, embed_b, return_prob=False, th=0.75):
        logits = np.matmul(embed_a, embed_b.transpose(1, 0))
        close = (logits > th)
        if return_prob:
            return close, logits
        else:
            return close
    #
    # def update_object_graph(self, object_embedding, obs, done):
    #     object_score, object_category, object_mask, object_position, object_bboxes, time = \
    #         obs['object_score'], obs['object_category'], obs['object_mask'], obs['object_pose'], obs['object'], obs['step']
    #     # The position is only used for visualizations. Remove if object features are similar
    #     # object masking
    #     object_score = object_score[object_mask==1]
    #     object_category = object_category[object_mask==1]
    #     object_position = object_position[object_mask==1]
    #     object_embedding = object_embedding[object_mask==1]
    #     object_mask = object_mask[object_mask==1]
    #     if done:
    #         self.objgraph.reset()
    #         self.objgraph.initialize_graph(object_embedding, object_score, object_category, object_mask, object_position)
    #
    #     # not_found = not done  # Dense
    #     to_add = [True] * int(sum(object_mask))
    #
    #     if self.config.OBJECTGRAPH.SPARSE:
    #         not_found = ~self.found  # Sparse
    #     else:
    #         not_found = not done  # Dense
    #     if not_found:
    #         hop1_vis_node = self.imggraph.A[self.imggraph.last_localized_node_idx]
    #         hop1_obj_node_mask = np.sum(self.objgraph.A_OV.transpose(1, 0)[hop1_vis_node], 0) > 0
    #         curr_obj_node_mask = self.objgraph.A_OV[:, self.imggraph.last_localized_node_idx]
    #         neighbor_obj_node_mask = (hop1_obj_node_mask + curr_obj_node_mask) > 0
    #         neighbor_node_embedding = self.objgraph.graph_memory[neighbor_obj_node_mask]
    #         neighbor_obj_memory_idx = np.where(neighbor_obj_node_mask)[0]
    #         neighbor_obj_memory_score = self.objgraph.graph_score[neighbor_obj_memory_idx]
    #         neighbor_obj_memory_cat = self.objgraph.graph_category[neighbor_obj_memory_idx]
    #
    #         close, prob = self.is_close(neighbor_node_embedding, object_embedding, return_prob=True, th=self.obj_node_th)
    #         for c_i in range(prob.shape[1]):
    #             close_mem_indices = np.where(close[:, c_i] == 1)[0]
    #             # detection score 높은 순으로 체크
    #             for m_i in close_mem_indices:
    #                 is_same = False
    #                 to_update = False
    #                 # m_i = neighbor_obj_memory_idx[close_idx]
    #                 if (object_category[c_i] == neighbor_obj_memory_cat[m_i]) and object_category[c_i] != -1:
    #                     is_same = True
    #                     if object_score[c_i] > neighbor_obj_memory_score[m_i]:
    #                         to_update = True
    #
    #                 if is_same:
    #                     # 만약 새로 detect한 물체가 이미 메모리에 있는 물체라면 새로 추가하지 않는다
    #                     to_add[c_i] = False
    #
    #                 if to_update:
    #                     # 만약 새로 detect한 물체가 이미 메모리에 있는 물체고 새로 detect한 물체의 score가 높다면 메모리를 새 물체로 업데이트 해준다
    #                     self.objgraph.update_node(m_i, time, object_score[c_i], object_category[c_i], int(self.imggraph.last_localized_node_idx), object_embedding[c_i])
    #                     break
    #
    #         # Add new objects to graph
    #         if sum(to_add) > 0:
    #             start_node_idx = self.objgraph.num_node()
    #             new_idx = np.where(np.stack(to_add))[0]
    #             self.objgraph.add_node(start_node_idx, object_embedding[new_idx], object_score[new_idx],
    #                                       object_category[new_idx], object_mask[new_idx], time,
    #                                       object_position[new_idx],  int(self.imggraph.last_localized_node_idx))
    #
    # def update_image_graph(self, new_embedding, curr_obj_embeding, obs, done):
    #     # The position is only used for visualizations.
    #     position, rotation, time = obs['position'], obs['rotation'],  obs['step']
    #     if done:
    #         self.imggraph.reset()
    #         self.imggraph.initialize_graph(new_embedding, position, rotation)
    #
    #     obj_close = True
    #     obj_graph_mask = self.objgraph.graph_score[self.objgraph.A_OV[:, self.imggraph.last_localized_node_idx]] > 0.5
    #     if len(obj_graph_mask) > 0:
    #         curr_obj_mask = obs['object_score'] > 0.5
    #         if np.sum(curr_obj_mask) / len(curr_obj_mask) >= 0.5:
    #             close_obj, prob_obj = self.is_close(self.objgraph.graph_memory[self.objgraph.A_OV[:, self.imggraph.last_localized_node_idx]], curr_obj_embeding, return_prob=True, th=self.obj_node_th)
    #             close_obj = close_obj[obj_graph_mask, :][:, curr_obj_mask]
    #             category_mask = self.objgraph.graph_category[self.objgraph.A_OV[:, self.imggraph.last_localized_node_idx]][obj_graph_mask][:, None] == obs['object_category'][curr_obj_mask]
    #             close_obj[~category_mask] = False
    #             if len(close_obj) >= 3:
    #                 clos_obj_p = close_obj.any(1).sum() / (close_obj.shape[0])
    #                 if clos_obj_p < 0.1:  # Fail to localize (find the same object) with the last localized frame
    #                     obj_close = False
    #
    #     close, prob = self.is_close(self.imggraph.last_localized_node_embedding[None], new_embedding[None], return_prob=True, th=self.img_node_th)
    #     # print("image prob", prob[0])
    #
    #     found = (np.array(done) + close.squeeze()) & np.array(obj_close).squeeze()
    #     # found = np.array(done) + close.squeeze()  # (T,T): is in 0 state, (T,F): not much moved, (F,T): impossible, (F,F): moved much
    #     self.found_prev = False
    #     self.found = found
    #     self.found_in_memory = False
    #     to_add = False
    #     if found:
    #         self.imggraph.update_nodes(self.imggraph.last_localized_node_idx, time)
    #         self.found_prev = True
    #     else:
    #         # 모든 메모리 노드 체크
    #         check_list = 1 - self.imggraph.graph_mask[:self.imggraph.num_node()]
    #         # 바로 직전 노드는 체크하지 않는다.
    #         check_list[self.imggraph.last_localized_node_idx] = 1.0
    #         while not found:
    #             not_checked_yet = np.where((1 - check_list))[0]
    #             neighbor_embedding = self.imggraph.graph_memory[not_checked_yet]
    #             num_to_check = len(not_checked_yet)
    #             if num_to_check == 0:
    #                 # 과거의 노드와도 다르고, 메모리와도 모두 다르다면 새로운 노드로 추가
    #                 to_add = True
    #                 break
    #             else:
    #                 # 메모리 노드에 존재하는지 체크
    #                 close, prob = self.is_close(new_embedding[None], neighbor_embedding, return_prob=True, th=self.img_node_th)
    #                 close = close[0];  prob = prob[0]
    #                 close_idx = np.where(close)[0]
    #                 if len(close_idx) >= 1:
    #                     found_node = not_checked_yet[prob.argmax()]
    #                 else:
    #                     found_node = None
    #                 if found_node is not None:
    #                     found = True
    #                     if abs(time - self.imggraph.graph_time[found_node]) > 20:
    #                         self.found_in_memory = True #만약 새롭게 찾은 노드가 오랜만에 돌아온 노드라면 found_in_memory를 True로 바꿔준다
    #                     self.imggraph.update_node(found_node, time, new_embedding)
    #                     self.imggraph.add_edge(found_node, self.imggraph.last_localized_node_idx)
    #                     self.imggraph.record_localized_state(found_node, new_embedding)
    #                 check_list[found_node] = 1.0
    #
    #     if to_add:
    #         new_node_idx = self.imggraph.num_node()
    #         self.imggraph.add_node(new_node_idx, new_embedding, time, position, rotation)
    #         self.imggraph.add_edge(new_node_idx, self.imggraph.last_localized_node_idx)
    #         self.imggraph.record_localized_state(new_node_idx, new_embedding)
    #     self.last_localized_node_idx = self.imggraph.last_localized_node_idx

    def embed_obs(self, obs):
        with torch.no_grad():
            img_tensor = (torch.tensor(obs['panoramic_rgb'][None]).to(self.torch_device).float() / 255).permute(0, 3, 1, 2)
            img_embedding = nn.functional.normalize(self.img_encoder(img_tensor).view(-1, self.feature_dim), dim=1)
        return img_embedding[0].cpu().detach().numpy()

    def embed_target(self, obs):
        with torch.no_grad():
            img_tensor = obs['target_goal'][...,:3].permute(0, 3, 1, 2)
            img_embedding = nn.functional.normalize(self.img_encoder(img_tensor).view(-1, self.feature_dim), dim=1)
        return img_embedding[0].cpu().detach().numpy()

    def embed_object(self, obs):
        with torch.no_grad():
            im = Image.fromarray(np.uint8(obs['panoramic_rgb']))
            img_tensor = self.transform_eval(im)[None].to(self.torch_device)
            feat = self.obj_encoder(img_tensor, torch.tensor(obs['object']).to(self.torch_device).float()[None])
            obj_embedding = nn.functional.normalize(feat, dim=-1)
        return obj_embedding[0].cpu().detach().numpy()

    def build_graph(self, obs, reset=False):
        if not reset:
            obs, reward, done, info = obs
        curr_img_embeddings = self.embed_obs(obs)
        curr_object_embedding = self.embed_object(obs)
        if reset:
            self.imggraph.reset()
            self.imggraph.initialize_graph(curr_img_embeddings, obs['position'], obs['rotation'])
        else:
            self.imggraph = update_image_graph(self.imggraph, self.objgraph, curr_img_embeddings, curr_object_embedding, obs, done=False)
        img_memory_dict = self.get_img_memory()
        if reset:
            self.objgraph.reset()
            self.objgraph.initialize_graph(curr_object_embedding, obs['object_score'], obs['object_category'], obs['object_mask'], obs['object_pose'])
        else:
            self.objgraph = update_object_graph(self.imggraph, self.objgraph, curr_object_embedding, obs, done=False)
        obj_memory_dict = self.get_obj_memory()
        obs = self.add_memory_in_obs(obs, img_memory_dict, obj_memory_dict)
        if self.args.render:
            # self.draw_graphs()
            self.render('human')
        return obs

    def reset(self):
        if self.args.record > 0:
            self.record_pose_action = []
            self.record_graphs = []
            self.record_maps = []
            self.record_objects = []
            self.record_imgs = []
            self.record_iter += 1
        obs_list = super().reset()
        obs = self.build_graph(obs_list, reset=True)
        return obs

    def draw_semantic_map_(self):
        input_args = {'xyz': self.xyz, 'category': self.obs['object_seg']}
        self.draw_semantic_map(**input_args)

    def draw_graphs(self):
        input_args = {'node_list': self.objgraph.node_position_list, 'node_category': self.objgraph.graph_category, 'node_score': self.objgraph.graph_score,
                      'vis_node_list': self.imggraph.node_position_list, 'affinity': self.objgraph.A_OV, 'graph_mask': self.objgraph.graph_mask,
                      'curr_info': {'curr_node': self.objgraph.last_localized_node_idx}}
        self.draw_object_graph_on_map(**input_args)
        input_args = {'node_list': self.imggraph.node_position_list, 'affinity': self.imggraph.A, 'graph_mask': self.imggraph.graph_mask,
                      'curr_info': {'curr_node': self.imggraph.last_localized_node_idx}}
        self.draw_image_graph_on_map(**input_args)

    def add_memory_in_obs(self, obs, memory_dict, obj_memory_dict):
        """
        Add memory in observation
        """
        obs.update(memory_dict)
        obs.update(obj_memory_dict)
        obs.update({'object_localized_idx': self.objgraph.last_localized_node_idx})
        obs.update({'localized_idx': self.imggraph.last_localized_node_idx})
        if 'distance' in obs.keys():
            obs['distance'] = obs['distance']
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        obs = self.build_graph((obs, reward, done, info))
        if self.args.render:
            self.draw_graphs()
            self.draw_semantic_map_()
            self.render('human')
        return obs, reward, done, info

    def get_img_memory(self):
        img_memory_dict = {
            'img_memory_feat': self.imggraph.graph_memory,
            'img_memory_mask': self.imggraph.graph_mask,
            'img_memory_A': self.imggraph.A,
            'img_memory_idx': self.imggraph.last_localized_node_idx,
            'img_memory_time': self.imggraph.graph_time
        }
        return img_memory_dict

    def get_obj_memory(self):
        obj_memory_dict = {
            'obj_memory_feat': self.objgraph.graph_memory,
            'obj_memory_score': self.objgraph.graph_score,
            'obj_memory_category': self.objgraph.graph_category,
            'obj_memory_mask': self.objgraph.graph_mask,
            'obj_memory_A_OV': self.objgraph.A_OV,
            'obj_memory_time': self.objgraph.graph_time
        }
        return obj_memory_dict