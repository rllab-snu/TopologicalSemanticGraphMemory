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