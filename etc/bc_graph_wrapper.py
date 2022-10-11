from gym.wrappers.monitor import Wrapper
from gym.spaces.box import Box
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from NuriUtils.habitat_utils import batch_obs
from NuriUtils.debug_utils import log_time
from model.PCL.resnet_pcl import resnet18
from model.PCL.resnet_obj_pcl import resnet18 as resnet18_obj_pcl
from torchvision.models import resnet18 as resnet18_rgb
import os
from env_utils.env_wrapper.graph_batch import Graph, ObjectGraph
import torchvision.transforms as transforms
import time
from habitat.core.vector_env import VectorEnv
from NuriUtils.statics import CATEGORIES
from types import SimpleNamespace

TIME_DEBUG = False


class GraphWrapper(Wrapper):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, envs, exp_config):
        self.envs = envs
        self.env = self.envs
        try:
            self.project_dir = exp_config['ARGS']['project_dir']
        except:
            self.project_dir = '.'
        if isinstance(envs, VectorEnv):
            self.is_vector_env = True
            self.num_envs = self.envs.num_envs
            self.action_spaces = self.envs.action_spaces
            self.observation_spaces = self.envs.observation_spaces
        else:
            self.is_vector_env = False
            self.num_envs = 1
        self.args = SimpleNamespace(**exp_config['ARGS'])
        self.train_bc = exp_config.TASK_CONFIG.TRAIN_BC
        self.B = self.num_envs
        self.exp_config = exp_config
        self.scene_data = exp_config.scene_data
        self.input_shape = exp_config.IMG_SHAPE
        self.object_feature_dim = exp_config.features.object_feature_dim  # + exp_config.features.object_category_num
        self.num_objects = exp_config.memory.num_objects
        self.feature_dim = exp_config.memory.embedding_size
        self.torch = exp_config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.GPU_GPU
        self.torch_device = 'cuda:' + str(exp_config.TORCH_GPU_ID) if torch.cuda.device_count() > 0 else 'cpu'

        self.visual_encoder_type = 'unsupervised'
        self.visual_encoder = self.load_visual_encoder(self.visual_encoder_type, self.input_shape, self.feature_dim).to(self.torch_device)
        self.object_encoder = self.load_object_encoder(exp_config.TASK_CONFIG.DATASET.DATASET_NAME, exp_config.features.object_feature_dim)  # .to(self.torch_device)
        self.img_node_th = exp_config.TASK_CONFIG.img_node_th
        self.object_th = exp_config.TASK_CONFIG.obj_node_th
        try:
            self.stop_th = exp_config.TASK_CONFIG.stop_th
        except:
            self.stop_th = 0.9
        self.graph = Graph(exp_config, self.B)
        self.objectgraph = ObjectGraph(exp_config, self.B)
        self.need_goal_embedding = 'wo_Fvis' in exp_config.POLICY
        self.img_height = float(exp_config.IMG_SHAPE[0])
        self.num_of_camera = exp_config.NUM_CAMERA
        self.img_width = float(exp_config.IMG_SHAPE[0] * 4 // self.num_of_camera * self.num_of_camera)
        angles = [2 * np.pi * idx / self.num_of_camera for idx in range(self.num_of_camera - 1, -1, -1)]
        half = self.num_of_camera // 2
        self.angles = angles[half:] + angles[:half]
        try:
            self.dn = exp_config.TASK_CONFIG.DATASET.DATASET_NAME.split("_")[0]
        except:
            self.dn = self.args.dataset.split("_")[0]
        self.img_ranges = np.arange(self.num_of_camera + 1) * (exp_config.IMG_SHAPE[0] * 4 // self.num_of_camera)
        if isinstance(envs, VectorEnv):
            for obs_space in self.observation_spaces:
                obs_space.spaces.update(
                    {'global_memory': Box(low=-np.Inf, high=np.Inf, shape=(self.graph.M, self.feature_dim), dtype=np.float32),
                     'global_memory_relpose': Box(low=-np.Inf, high=np.Inf, shape=(self.graph.M, 3), dtype=np.float32),
                     'global_act_memory': Box(low=-np.Inf, high=np.Inf, shape=(self.graph.M,), dtype=np.float32),
                     'global_mask': Box(low=-np.Inf, high=np.Inf, shape=(self.graph.M,), dtype=np.bool),
                     'global_A': Box(low=-np.Inf, high=np.Inf, shape=(self.graph.M, self.graph.M), dtype=np.bool),
                     'global_idx': Box(low=-np.Inf, high=np.Inf, shape=(), dtype=np.float32),
                     'global_time': Box(low=-np.Inf, high=np.Inf, shape=(self.graph.M,), dtype=np.float32)
                     }
                )
                obs_space.spaces.update(
                    {'object_memory': Box(low=-np.Inf, high=np.Inf, shape=(self.objectgraph.M, self.object_feature_dim), dtype=np.float32),
                     'object_memory_score': Box(low=-np.Inf, high=np.Inf, shape=(self.objectgraph.M,), dtype=np.float32),
                     'object_memory_relpose': Box(low=-np.Inf, high=np.Inf, shape=(self.objectgraph.M, 2), dtype=np.float32),
                     'object_memory_category': Box(low=-np.Inf, high=np.Inf, shape=(self.objectgraph.M,), dtype=np.float32),
                     'object_memory_mask': Box(low=-np.Inf, high=np.Inf, shape=(self.objectgraph.M,), dtype=np.bool),
                     'object_memory_A_OV': Box(low=-np.Inf, high=np.Inf, shape=(self.objectgraph.M, self.objectgraph.MV), dtype=np.bool),
                     'object_memory_time': Box(low=-np.Inf, high=np.Inf, shape=(self.objectgraph.M,), dtype=np.float32)
                     }
                )
                if self.need_goal_embedding:
                    obs_space.spaces.update(
                        {'goal_embedding': Box(low=-np.Inf, high=np.Inf, shape=(self.feature_dim,), dtype=np.float32)}
                    )
        self.num_agents = exp_config.NUM_AGENTS

        self.localize_mode = 'predict'
        self.reset_all_memory()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        eval_augmentation = [
            transforms.ToTensor(),
            normalize
        ]
        self.transform_eval = transforms.Compose(eval_augmentation)

    def load_visual_encoder(self, type, input_shape, feature_dim):
        if self.args.dataset == "mp3d":
            visual_encoder = resnet18_rgb(num_classes=feature_dim)
            dim_mlp = visual_encoder.fc.weight.shape[1]
            visual_encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), visual_encoder.fc)
            ckpt_pth = os.path.join(self.project_dir, 'model/PCL', f'PCL_encoder_mp3d.pth.tar')
            ckpt = torch.load(ckpt_pth, map_location='cpu')
            visual_encoder.load_state_dict({k[len('module.encoder_q.'):]: v for k, v in ckpt['state_dict'].items() if 'module.encoder_q.' in k})
        else:
            if self.args.nodepth:
                visual_encoder = resnet18_rgb(num_classes=feature_dim)
                dim_mlp = visual_encoder.fc.weight.shape[1]
                visual_encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), visual_encoder.fc)
                ckpt_pth = os.path.join(self.project_dir, 'model/PCL', f'PCL_encoder_nodepth.pth.tar')
                ckpt = torch.load(ckpt_pth, map_location='cpu')
                visual_encoder.load_state_dict({k[len('module.encoder_q.'):]: v for k, v in ckpt['state_dict'].items() if 'module.encoder_q.' in k})
            else:
                visual_encoder = resnet18(num_classes=feature_dim)
                dim_mlp = visual_encoder.fc.weight.shape[1]
                visual_encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), visual_encoder.fc)
                ckpt_pth = os.path.join(self.project_dir, 'model/PCL', 'PCL_encoder.pth')
                ckpt = torch.load(ckpt_pth, map_location='cpu')
                visual_encoder.load_state_dict(ckpt)
        visual_encoder.eval().to(self.torch_device)
        return visual_encoder

    def load_object_encoder(self, dataset, feature_dim):
        object_encoder = resnet18_obj_pcl(num_classes=feature_dim)
        dim_mlp = object_encoder.fc.weight.shape[1]
        object_encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), object_encoder.fc)
        ckpt_pth = os.path.join(self.project_dir, 'model/PCL', f'ObjPCL_gibson.pth.tar')
        ckpt = torch.load(ckpt_pth, map_location='cpu')
        state_dict = {k[len('module.encoder_k.'):]: v for k, v in ckpt['state_dict'].items() if 'module.encoder_k.' in k}
        object_encoder.load_state_dict(state_dict)
        object_encoder.eval().to(self.torch_device)
        return object_encoder

    def reset_all_memory(self, B=None):
        self.graph.reset(B)
        self.objectgraph.reset(B)

    def is_close(self, embed_a, embed_b, return_prob=False, th=0.75):
        with torch.no_grad():
            logits = torch.matmul(embed_a, embed_b.transpose(2, 1))  # .squeeze()
            close = (logits > th).detach().cpu()
        if return_prob: return close, logits
        else: return close

    def update_object_graph(self, object_embedding, obs_batch, done_list):
        object_score, object_category, object_mask, object_position, object_pose, time = obs_batch['object_score'], obs_batch['object_category'], obs_batch['object_mask'].detach().cpu().numpy(),\
                                                                     obs_batch['object_pose'].detach().cpu().numpy(), obs_batch['object_relpose'].detach().cpu().numpy(), obs_batch['step']
        # The position is only used for visualizations. Remove if object features are similar
        # initialize graph if an episode is finished
        done = np.where(done_list)[0]
        if len(done) > 0:
            for b in done:
                self.objectgraph.reset_at(b)
                self.objectgraph.initialize_graph(b, object_embedding, object_score, object_category, object_mask, object_position)

        not_found_batch_indicies = np.where([not i for i in done_list])[0] #Dense
        for b in not_found_batch_indicies:
            hop1_vis_node = self.graph.A[b, self.graph.last_localized_node_idx[b].long()]
            hop1_obj_node_mask = torch.sum(self.objectgraph.A_OV.permute(0, 2, 1)[b][hop1_vis_node], 0) > 0
            curr_obj_node_mask = self.objectgraph.A_OV[b, :, self.graph.last_localized_node_idx[b].long()]
            neighbor_obj_node_mask = (hop1_obj_node_mask + curr_obj_node_mask) > 0
            neighbor_node_embedding = self.objectgraph.graph_memory[b, neighbor_obj_node_mask]
            neighbor_obj_memory_idx = torch.where(neighbor_obj_node_mask)[0]
            neighbor_obj_memory_score = self.objectgraph.graph_score[b, neighbor_obj_memory_idx]

            exist_in_memory = [] #current objects
            close, prob = self.is_close(neighbor_node_embedding[None], object_embedding[b][object_mask[b]==1][None], return_prob=True, th=self.object_th)
            for c_i in range(prob[0].shape[1]):
                close_mem_indices = torch.where(close[0, :, c_i] == 1)[0]
                idx = torch.argsort(prob[0][close_mem_indices, c_i], descending=True)
                # idx = torch.argsort(neighbor_obj_memory_score[close_mem_indices], descending=True)
                close_mem_indices = close_mem_indices[idx]
                for close_idx in close_mem_indices:
                    m_i = neighbor_obj_memory_idx[close_idx].item()
                    is_same, node_idx_a, node_idx_b, node_score = self.objectgraph.update_node(b, m_i, time[b], object_score[b][c_i], object_category[b][c_i], int(self.graph.last_localized_node_idx[b]),
                                                                                               object_position[b][c_i],
                                                                                               object_embedding[b][c_i])
                    if is_same:
                        # if node_score > 0.75:
                        #     self.graph.add_edges(b, node_idx_a, node_idx_b)
                        exist_in_memory.append(c_i)
                        break

            start_node_idx = self.objectgraph.num_node(b)
            new_idx = np.array(list(set(np.arange(sum(object_mask[b]))).difference(exist_in_memory)), dtype=np.int32)

            # Add new objects to graph
            if len(new_idx) > 0:
                self.objectgraph.add_node(b, start_node_idx, object_embedding[b][new_idx], object_score[b][new_idx],
                                          object_category[b][new_idx], object_mask[b][new_idx], time[b], object_position[b][new_idx], int(self.graph.last_localized_node_idx[b]))

    # assume memory index == node index
    def localize(self, new_embedding, position, relpose, time, done_list, curr_obj_embeding, obs_batch):
        # The position is only used for visualizations.
        done = np.where(done_list)[0]
        if len(done) > 0:
            for b in done:
                self.graph.reset_at(b)
                self.graph.initialize_graph(b, new_embedding, position, relpose)

        close, prob = self.is_close(self.graph.last_localized_node_embedding.unsqueeze(1), new_embedding.unsqueeze(1), return_prob=True, th=self.img_node_th)
        # print(prob)
        obj_close = np.ones_like(close).astype(np.bool)
        for b in range(new_embedding.shape[0]):
            obj_graph_mask = self.objectgraph.graph_score[b, self.objectgraph.A_OV[b, :, self.graph.last_localized_node_idx[b]]] > 0.5
            if len(obj_graph_mask) > 0:
                curr_obj_mask = obs_batch['object_score'] > 0.5
                if torch.sum(curr_obj_mask) / curr_obj_mask.shape[1] >= 0.5:
                    close_obj, prob_obj = self.is_close(self.objectgraph.graph_memory[b, self.objectgraph.A_OV[b, :, self.graph.last_localized_node_idx[b]]][None], curr_obj_embeding[b][None], return_prob=True, th=0.9)
                    close_obj = close_obj[0][obj_graph_mask, :][:, curr_obj_mask[0]]
                    category_mask = self.objectgraph.graph_category[b, self.objectgraph.A_OV[b, :, self.graph.last_localized_node_idx[b]]][obj_graph_mask][:, None].cpu() == \
                                    obs_batch['object_category'][b][curr_obj_mask[0]][None].cpu()
                    close_obj[~category_mask] = False
                    if len(close_obj) >= 3:
                        clos_obj_p = close_obj.any(1).sum() / (close_obj.shape[0])
                        # print(clos_obj_p)
                        if clos_obj_p < 0.1:  # Fail to localize (find the same object) with the last localized frame
                            obj_close[b] = False
                        else:
                            obj_close[b] = True

        found = (torch.tensor(done_list) + close.squeeze()) & torch.tensor(obj_close).squeeze()  # (T,T): is in 0 state, (T,F): not much moved, (F,T): impossible, (F,F): moved much
        self.found = found.clone()
        found_batch_indices = torch.where(found)[0]

        localized_node_indices = torch.ones([self.B], dtype=torch.int32) * -1
        localized_node_indices[found_batch_indices] = self.graph.last_localized_node_idx[found_batch_indices]
        self.graph.update_nodes(found_batch_indices, localized_node_indices[found_batch_indices], time[found_batch_indices])

        check_list = 1 - self.graph.graph_mask[:, :self.graph.num_node_max()]
        check_list[range(self.B), self.graph.last_localized_node_idx.long()] = 1.0
        # for b in range(self.B):
        #     check_list[b, self.graph.num_node(b) - 1:] = 1.0
        check_list[found_batch_indices] = 1.0
        to_add = torch.zeros(self.B)
        hop = 1
        max_hop = 0
        while not found.all():
            if hop <= max_hop: k_hop_A = self.graph.calculate_multihop(hop)
            not_found_batch_indicies = torch.where(~found)[0]
            neighbor_embedding = []
            batch_new_embedding = []
            num_neighbors = []
            neighbor_indices = []
            for b in not_found_batch_indicies:
                if hop <= max_hop:
                    neighbor_mask = k_hop_A[b, self.graph.last_localized_node_idx[b]] == 1
                    not_checked_yet = torch.where((1 - check_list[b]) * neighbor_mask[:len(check_list[b])])[0]
                else:
                    not_checked_yet = torch.where((1 - check_list[b]))[0]
                neighbor_indices.append(not_checked_yet)
                neighbor_embedding.append(self.graph.graph_memory[b, not_checked_yet])
                num_neighbors.append(len(not_checked_yet))
                if len(not_checked_yet) > 0:
                    batch_new_embedding.append(new_embedding[b:b+1].repeat(len(not_checked_yet),1))
                else:
                    found[b] = True
                    to_add[b] = True
            if torch.sum(torch.tensor(num_neighbors)) > 0:
                neighbor_embedding = torch.cat(neighbor_embedding)
                batch_new_embedding = torch.cat(batch_new_embedding)
                bs = neighbor_embedding.shape[0]
                batch_close, batch_prob = self.is_close(neighbor_embedding.unsqueeze(1), batch_new_embedding.unsqueeze(1), return_prob=True, th=self.img_node_th)
                close = batch_close.reshape(bs).split(num_neighbors)
                prob = batch_prob.reshape(bs).split(num_neighbors)
                for ii in range(len(close)):
                    is_close = torch.where(close[ii] == True)[0]
                    if len(is_close) == 1:
                        found_node = neighbor_indices[ii][is_close.item()].item()
                    elif len(is_close) > 1:
                        found_node = neighbor_indices[ii][prob[ii].argmax().item()].item()
                    else:
                        found_node = None
                    b = not_found_batch_indicies[ii]
                    if found_node is not None:
                        found[b] = True
                        localized_node_indices[b] = found_node
                        if found_node != self.graph.last_localized_node_idx[b].item():
                            self.graph.update_node(b, found_node, time[b], new_embedding[b])
                            self.graph.add_edge(b, found_node, self.graph.last_localized_node_idx[b])
                            self.graph.record_localized_state(b, found_node, new_embedding[b])
                    check_list[b, neighbor_indices[ii]] = 1.0
            hop += 1

        batch_indices_to_add_new_node = torch.where(to_add)[0]
        for b in batch_indices_to_add_new_node:
            new_node_idx = self.graph.num_node(b)
            self.graph.add_node(b, new_node_idx, new_embedding[b], time[b], position[b], relpose[b])
            self.graph.add_edge(b, new_node_idx, self.graph.last_localized_node_idx[b])
            self.graph.record_localized_state(b, new_node_idx, new_embedding[b])

    # assume memory index == node index
    def localize_only_image(self, new_embedding, position, relpose, time, done_list, action, curr_obj_embeding, obs_batch):
        # The position is only used for visualizations.
        done = np.where(done_list)[0]
        if len(done) > 0:
            for b in done:
                self.graph.reset_at(b)
                self.graph.initialize_graph(b, new_embedding, position, relpose, action)

        close, prob = self.is_close(self.graph.last_localized_node_embedding.unsqueeze(1), new_embedding.unsqueeze(1), return_prob=True, th=self.img_node_th)

        found = (torch.tensor(done_list) + close.squeeze())# & torch.tensor(obj_close).squeeze()  # (T,T): is in 0 state, (T,F): not much moved, (F,T): impossible, (F,F): moved much
        self.found = found.clone()
        found_batch_indices = torch.where(found)[0]

        localized_node_indices = torch.ones([self.B], dtype=torch.int32) * -1
        localized_node_indices[found_batch_indices] = self.graph.last_localized_node_idx[found_batch_indices]
        self.graph.update_nodes(found_batch_indices, localized_node_indices[found_batch_indices], time[found_batch_indices])

        check_list = 1 - self.graph.graph_mask[:, :self.graph.num_node_max()]
        check_list[range(self.B), self.graph.last_localized_node_idx.long()] = 1.0
        # for b in range(self.B):
        #     check_list[b, self.graph.num_node(b) - 1:] = 1.0
        check_list[found_batch_indices] = 1.0
        to_add = torch.zeros(self.B)
        hop = 1
        max_hop = 0
        while not found.all():
            if hop <= max_hop: k_hop_A = self.graph.calculate_multihop(hop)
            not_found_batch_indicies = torch.where(~found)[0]
            neighbor_embedding = []
            batch_new_embedding = []
            num_neighbors = []
            neighbor_indices = []
            for b in not_found_batch_indicies:
                if hop <= max_hop:
                    neighbor_mask = k_hop_A[b, self.graph.last_localized_node_idx[b]] == 1
                    not_checked_yet = torch.where((1 - check_list[b]) * neighbor_mask[:len(check_list[b])])[0]
                else:
                    not_checked_yet = torch.where((1 - check_list[b]))[0]
                neighbor_indices.append(not_checked_yet)
                neighbor_embedding.append(self.graph.graph_memory[b, not_checked_yet])
                num_neighbors.append(len(not_checked_yet))
                if len(not_checked_yet) > 0:
                    batch_new_embedding.append(new_embedding[b:b+1].repeat(len(not_checked_yet),1))
                else:
                    found[b] = True
                    to_add[b] = True
            if torch.sum(torch.tensor(num_neighbors)) > 0:
                neighbor_embedding = torch.cat(neighbor_embedding)
                batch_new_embedding = torch.cat(batch_new_embedding)
                bs = neighbor_embedding.shape[0]
                batch_close, batch_prob = self.is_close(neighbor_embedding.unsqueeze(1), batch_new_embedding.unsqueeze(1), return_prob=True, th=self.img_node_th)
                close = batch_close.reshape(bs).split(num_neighbors)
                prob = batch_prob.reshape(bs).split(num_neighbors)
                for ii in range(len(close)):
                    is_close = torch.where(close[ii] == True)[0]
                    if len(is_close) == 1:
                        found_node = neighbor_indices[ii][is_close.item()].item()
                    elif len(is_close) > 1:
                        found_node = neighbor_indices[ii][prob[ii].argmax().item()].item()
                    else:
                        found_node = None
                    b = not_found_batch_indicies[ii]
                    if found_node is not None:
                        found[b] = True
                        localized_node_indices[b] = found_node
                        if found_node != self.graph.last_localized_node_idx[b].item():
                            self.graph.update_node(b, found_node, time[b], new_embedding[b])
                            self.graph.add_edge(b, found_node, self.graph.last_localized_node_idx[b])
                            self.graph.record_localized_state(b, found_node, new_embedding[b])
                    check_list[b, neighbor_indices[ii]] = 1.0
            hop += 1

        batch_indices_to_add_new_node = torch.where(to_add)[0]
        for b in batch_indices_to_add_new_node:
            new_node_idx = self.graph.num_node(b)
            self.graph.add_node(b, new_node_idx, new_embedding[b], time[b], position[b], relpose[b], action[b])
            self.graph.add_edge(b, new_node_idx, self.graph.last_localized_node_idx[b])
            self.graph.record_localized_state(b, new_node_idx, new_embedding[b])

    # assume memory index == node index
    def localize_using_obj(self, new_embedding, position, relpose, time, done_list, action, curr_obj_embeding, obs_batch):
        # The position is only used for visualizations.
        done = np.where(done_list)[0]
        if len(done) > 0:
            for b in done:
                self.graph.reset_at(b)
                self.graph.initialize_graph(b, new_embedding, position, relpose, action)

        close, prob = self.is_close(self.graph.last_localized_node_embedding.unsqueeze(1), new_embedding.unsqueeze(1), return_prob=True, th=self.img_node_th)
        # print("image prob", prob[0])
        obj_close = np.ones_like(close).astype(np.bool)
        for b in range(new_embedding.shape[0]):
            obj_graph_mask = self.objectgraph.graph_score[b, self.objectgraph.A_OV[b, :, self.graph.last_localized_node_idx[b]]] > 0.5
            if len(obj_graph_mask) > 0:
                curr_obj_mask = obs_batch['object_score'] > 0.5
                if torch.sum(curr_obj_mask) / curr_obj_mask.shape[1] >= 0.5:
                    close_obj, prob_obj = self.is_close(self.objectgraph.graph_memory[b, self.objectgraph.A_OV[b, :, self.graph.last_localized_node_idx[b]]][None], curr_obj_embeding[b][None], return_prob=True, th=0.9)
                    close_obj = close_obj[0][obj_graph_mask, :][:, curr_obj_mask[0]]
                    category_mask = self.objectgraph.graph_category[b, self.objectgraph.A_OV[b, :, self.graph.last_localized_node_idx[b]]][obj_graph_mask][:, None].cpu() == \
                                    obs_batch['object_category'][b][curr_obj_mask[0]][None].cpu()
                    close_obj[~category_mask] = False
                    if len(close_obj) >= 3:
                        clos_obj_p = close_obj.any(1).sum() / (close_obj.shape[0])
                        # print(clos_obj_p)
                        if clos_obj_p < 0.1:  # Fail to localize (find the same object) with the last localized frame
                            obj_close[b] = False
                        else:
                            obj_close[b] = True

        found = (torch.tensor(done_list) + close.squeeze()) & torch.tensor(obj_close).squeeze()  # (T,T): is in 0 state, (T,F): not much moved, (F,T): impossible, (F,F): moved much
        self.found = found.clone()
        found_batch_indices = torch.where(found)[0]

        localized_node_indices = torch.ones([self.B], dtype=torch.int32) * -1
        localized_node_indices[found_batch_indices] = self.graph.last_localized_node_idx[found_batch_indices]
        self.graph.update_nodes(found_batch_indices, localized_node_indices[found_batch_indices], time[found_batch_indices])

        check_list = 1 - self.graph.graph_mask[:, :self.graph.num_node_max()]
        check_list[range(self.B), self.graph.last_localized_node_idx.long()] = 1.0
        for b in range(self.B):
            check_list[b, self.graph.num_node(b) - 1:] = 1.0
        check_list[found_batch_indices] = 1.0
        to_add = torch.zeros(self.B)
        hop = 1
        max_hop = 0
        while not found.all():
            if hop <= max_hop: k_hop_A = self.graph.calculate_multihop(hop)
            not_found_batch_indicies = torch.where(~found)[0]
            neighbor_embedding = []
            neighbor_obj_embedding = []
            neighbor_obj_category = []
            neighbor_obj_score = []
            batch_new_embedding = []
            batch_new_obj_embedding = []
            batch_new_obj_category = []
            batch_new_obj_score = []
            num_neighbors = []
            neighbor_indices = []
            for b in not_found_batch_indicies:
                if hop <= max_hop:
                    neighbor_mask = k_hop_A[b, self.graph.last_localized_node_idx[b]] == 1
                    not_checked_yet = torch.where((1 - check_list[b]) * neighbor_mask[:len(check_list[b])])[0]
                else:
                    not_checked_yet = torch.where((1 - check_list[b]))[0]
                neighbor_indices.append(not_checked_yet)
                neighbor_embedding.append(self.graph.graph_memory[b, not_checked_yet])
                for not_checked_yet_ in not_checked_yet:
                    neighbor_obj_embedding.append(self.objectgraph.graph_memory[b][self.objectgraph.A_OV[b, :, [not_checked_yet_]].sum(-1) == 1])
                    neighbor_obj_category.append(self.objectgraph.graph_category[b][self.objectgraph.A_OV[b, :, [not_checked_yet_]].sum(-1) == 1])
                    neighbor_obj_score.append(self.objectgraph.graph_score[b][self.objectgraph.A_OV[b, :, [not_checked_yet_]].sum(-1) == 1])
                num_neighbors.append(len(not_checked_yet))
                if len(not_checked_yet) > 0:
                    batch_new_embedding.append(new_embedding[b:b + 1].repeat(len(not_checked_yet), 1))
                    batch_new_obj_embedding.append(curr_obj_embeding[b:b + 1].repeat(len(not_checked_yet), 1, 1))
                    batch_new_obj_score.append(obs_batch['object_score'][b:b + 1].repeat(len(not_checked_yet), 1))
                    batch_new_obj_category.append(obs_batch['object_category'][b:b + 1].repeat(len(not_checked_yet), 1))
                else:
                    found[b] = True
                    to_add[b] = True
            if torch.sum(torch.tensor(num_neighbors)) > 0:
                neighbor_embedding = torch.cat(neighbor_embedding)
                batch_new_embedding = torch.cat(batch_new_embedding)
                batch_new_obj_embedding = torch.cat(batch_new_obj_embedding)
                batch_new_obj_category = torch.cat(batch_new_obj_category)
                batch_new_obj_score = torch.cat(batch_new_obj_score)
                # neighbor_indices = torch.cat(neighbor_indices)
                bs = neighbor_embedding.shape[0]
                # batch_close, batch_prob = self.is_close(neighbor_embedding, batch_new_embedding, return_prob=True)
                batch_close, batch_prob = self.is_close(neighbor_embedding.unsqueeze(1), batch_new_embedding.unsqueeze(1), return_prob=True, th=self.img_node_th)
                close = batch_close.reshape(bs).split(num_neighbors)
                prob = batch_prob.reshape(bs).split(num_neighbors)
                batch_obj_prob = torch.zeros(bs)
                for b_ in range(bs):
                    obj_graph_mask = neighbor_obj_score[b_] > 0.5
                    if len(obj_graph_mask) > 0:
                        curr_obj_mask = batch_new_obj_score[b_] > 0.5
                        close_obj, prob_obj = self.is_close(neighbor_obj_embedding[b_][None], batch_new_obj_embedding[b_][None], return_prob=True, th=0.9)
                        close_obj = close_obj[0][obj_graph_mask, :][:, curr_obj_mask]
                        category_mask = neighbor_obj_category[b_][obj_graph_mask][:, None].cpu() == batch_new_obj_category[b_][curr_obj_mask][None].cpu()
                        close_obj[~category_mask] = False
                        clos_obj_p = close_obj.any(1).sum() / (close_obj.shape[0])
                        batch_obj_prob[b_] = clos_obj_p
                batch_obj_prob = batch_obj_prob.reshape(bs).split(num_neighbors)
                for ii in range(len(close)):
                    close_neighbor = (batch_obj_prob[ii] >= 0.2) * close[ii]
                    is_close = torch.where(close_neighbor == True)[0]
                    if len(is_close) == 1:
                        found_node = neighbor_indices[ii][is_close.item()].item()
                    elif len(is_close) > 1:
                        found_node = neighbor_indices[ii][prob[ii].argmax().item()].item()
                    else:
                        found_node = None
                    b = not_found_batch_indicies[ii]
                    if found_node is not None:
                        found[b] = True
                        localized_node_indices[b] = found_node
                        if found_node != self.graph.last_localized_node_idx[b].item():
                            self.graph.update_node(b, found_node, time[b], new_embedding[b])
                            self.graph.add_edge(b, found_node, self.graph.last_localized_node_idx[b])
                            self.graph.record_localized_state(b, found_node, new_embedding[b])
                    check_list[b, neighbor_indices[ii]] = 1.0
            hop += 1

        batch_indices_to_add_new_node = torch.where(to_add)[0]
        for b in batch_indices_to_add_new_node:
            new_node_idx = self.graph.num_node(b)
            self.graph.add_node(b, new_node_idx, new_embedding[b], time[b], position[b], relpose[b], action[b])
            self.graph.add_edge(b, new_node_idx, self.graph.last_localized_node_idx[b])
            self.graph.record_localized_state(b, new_node_idx, new_embedding[b])

    def update_graph(self):
        if self.is_vector_env:
            args_list = [{'node_list': self.objectgraph.node_position_list[b], 'node_category': self.objectgraph.graph_category[b],
                          'vis_node_list': self.graph.node_position_list[b], 'affinity': self.objectgraph.A_OV[b],
                          'graph_mask': self.objectgraph.graph_mask[b], 'curr_info': {'curr_node': self.objectgraph.last_localized_node_idx[b]},
                          } for b in range(self.B)]
            self.envs.call(['update_vo_graph'] * self.B, args_list)
            args_list = [{'node_list': self.graph.node_position_list[b], 'affinity': self.graph.A[b], 'graph_mask': self.graph.graph_mask[b],
                          'curr_info': {'curr_node': self.graph.last_localized_node_idx[b]},
                          } for b in range(self.B)]
            self.envs.call(['update_graph'] * self.B, args_list)
        else:
            b = 0
            input_args = {'node_list': self.objectgraph.node_position_list[b], 'node_category': self.objectgraph.graph_category[b], 'node_score': self.objectgraph.graph_score[b],
                          'vis_node_list': self.graph.node_position_list[b], 'affinity': self.objectgraph.A_OV[b],
                          'graph_mask': self.objectgraph.graph_mask[b], 'curr_info': {'curr_node': self.objectgraph.last_localized_node_idx[b]}}
            self.envs.update_vo_graph(**input_args)
            input_args = {'node_list': self.graph.node_position_list[b], 'affinity': self.graph.A[b], 'graph_mask': self.graph.graph_mask[b],
                          'curr_info': {'curr_node': self.graph.last_localized_node_idx[b]}}
            self.envs.update_graph(**input_args)

    def update_goal(self, object_goal_position):
        if self.is_vector_env:
            args_list = [{'node_list': object_goal_position[b]} for b in range(self.B)]
            self.envs.call(['update_goal'] * self.B, args_list)
        else:
            b = 0
            input_args = {'node_list': object_goal_position[b]}
            self.envs.update_goal(**input_args)

    def set_object_on_env(self, object_bbox, object_score, object_category, object_distance, obs_list=[], reward_list=[], done_list=[], info_list=[]):
        if self.is_vector_env:
            args_list = [{'object_bbox': object_bbox[b], 'object_score': object_score[b], 'object_category': object_category[b], 'object_distance': object_distance[b]} for b in range(self.B)]
            aa = self.envs.call(['set_object'] * self.B, args_list)
            reward = [aa[i][0] for i in range(len(aa))]
            done = [aa[i][1] for i in range(len(aa))]
            info = [aa[i][2] for i in range(len(aa))]
            for i, done_i in enumerate(done):
                if done_i:
                    obs_list[i].update(self.envs.reset_at(i)[0])
                    reward_list[i] = reward[i]
                    done_list[i] = done_i
                    info_list[i].update(info[i])
        else:
            b = 0
            input_args = {'object_bbox': object_bbox[b], 'object_score': object_score[b], 'object_category': object_category[b], 'object_distance': object_distance[b]}
            reward, done, info = self.envs.set_object(**input_args)
            done = [done]
            for i, done_i in enumerate(done):
                obs_list[i].update(self.envs.reset_at(i)[0])
                reward_list[i].update(reward)
                done_list[i].update(done_i)
                info_list[i].update(info)
        return obs_list, reward_list, done_list, info_list

    def embed_obs(self, obs_batch):
        with torch.no_grad():
            if self.args.nodepth:
                img_tensor = (obs_batch['panoramic_rgb'] / 255.).permute(0, 3, 1, 2)
            else:
                img_tensor = torch.cat((obs_batch['panoramic_rgb'] / 255., obs_batch['panoramic_depth']), 3).permute(0, 3, 1, 2)
            vis_embedding = nn.functional.normalize(self.visual_encoder(img_tensor).view(self.B, -1), dim=1)
        return vis_embedding.detach().cpu()

    def embed_target(self, obs_batch):
        with torch.no_grad():
            if self.args.nodepth:
                img_tensor = obs_batch['target_goal'][...,:3].permute(0, 3, 1, 2)
            else:
                img_tensor = obs_batch['target_goal'].permute(0, 3, 1, 2)
            vis_embedding = nn.functional.normalize(self.visual_encoder(img_tensor).view(self.B, -1), dim=1)
        return vis_embedding.detach().cpu()

    def embed_object(self, obs_batch):
        with torch.no_grad():
            feat = self.object_encoder(obs_batch['panoramic_rgb'].permute(0,3,1,2) / 255., obs_batch['object'])
            obj_embedding = nn.functional.normalize(feat, dim=-1)
        return obj_embedding.detach().cpu()

    def update_memory(self, obs_batch, memory_dict):
        # add memory to obs
        obs_batch.update(memory_dict)
        obs_batch.update({'localized_idx': self.graph.last_localized_node_idx.unsqueeze(1)})
        if 'distance' in obs_batch.keys():
            obs_batch['distance'] = obs_batch['distance']  # .unsqueeze(1)
        if self.need_goal_embedding:
            obs_batch['goal_embedding'] = self.embed_target(obs_batch)
        return obs_batch

    def step(self, actions):
        with torch.no_grad():
            if TIME_DEBUG: s = time.time()
            if self.is_vector_env:
                dict_actions = [{'action': actions[b]} for b in range(self.B)]
                outputs = self.envs.step(dict_actions)
                if TIME_DEBUG: s = log_time(s, 'step env')
            else:
                outputs = [self.envs.step(actions)]
                actions = [actions]

            obs_list, reward_list, done_list, info_list = [list(x) for x in zip(*outputs)]
            obs_batch = batch_obs(obs_list, device=self.torch_device)

            curr_vis_embedding = self.embed_obs(obs_batch)
            curr_object_embedding = self.embed_object(obs_batch)
            if TIME_DEBUG: s = log_time(s, 'encode observations')
            self.localize(curr_vis_embedding, obs_batch['position'].detach().cpu().numpy(), obs_batch['relpose'].detach().cpu().numpy(),
                          obs_batch['step'].detach().cpu(), done_list, curr_object_embedding, obs_batch)
            global_memory_dict = self.get_global_memory()
            obs_batch = self.update_memory(obs_batch, global_memory_dict)
            if TIME_DEBUG: s = log_time(s, 'update image memory')
            self.update_object_graph(curr_object_embedding, obs_batch, done_list)
            object_memory_dict = self.get_object_memory()
            obs_batch.update(object_memory_dict)
            obs_batch.update({'object_localized_idx': self.objectgraph.last_localized_node_idx})
            if TIME_DEBUG: s = log_time(s, 'update object memory')
            if self.args.render > 0 or self.args.record > 0:
                self.update_graph()
                if TIME_DEBUG: s = log_time(s, 'update graph vis')

        if self.is_vector_env:
            return obs_batch, reward_list, done_list, info_list
        else:
            return obs_batch, reward_list[0], done_list[0], info_list[0]

    def reset_graph(self, obs_list):
        with torch.no_grad():
            if TIME_DEBUG: s = time.time()
            obs_batch = batch_obs([obs_list], device=self.torch_device)
            curr_vis_embeddings = self.embed_obs(obs_batch)
            curr_object_embedding = self.embed_object(obs_batch)
            if self.need_goal_embedding: obs_batch['curr_embedding'] = curr_vis_embeddings
            self.localize(curr_vis_embeddings, obs_batch['position'].detach().cpu().numpy(), obs_batch['relpose'].detach().cpu().numpy(),
                          obs_batch['step'].detach().cpu(), [True] * self.B, curr_object_embedding, obs_batch)
            global_memory_dict = self.get_global_memory()
            obs_batch = self.update_memory(obs_batch, global_memory_dict)
            self.update_object_graph(curr_object_embedding, obs_batch, [True] * self.B)
            object_memory_dict = self.get_object_memory()
            obs_batch.update(object_memory_dict)
            obs_batch.update({'object_localized_idx': self.objectgraph.last_localized_node_idx})
            if TIME_DEBUG: s = log_time(s, 'reset')
        return obs_batch

    def step_graph(self, obs_list, done_list=[False]):
        with torch.no_grad():
            obs_batch = batch_obs([obs_list], device=self.torch_device)
            curr_vis_embedding = self.embed_obs(obs_batch)
            curr_object_embedding = self.embed_object(obs_batch)
            self.localize(curr_vis_embedding, obs_batch['position'].detach().cpu().numpy(), obs_batch['relpose'].detach().cpu().numpy(),
                          obs_batch['step'].detach().cpu(), done_list, curr_object_embedding, obs_batch)
            global_memory_dict = self.get_global_memory()
            obs_batch = self.update_memory(obs_batch, global_memory_dict)
            self.update_object_graph(curr_object_embedding, obs_batch, done_list)
            object_memory_dict = self.get_object_memory()
            obs_batch.update(object_memory_dict)
            obs_batch.update({'object_localized_idx': self.objectgraph.last_localized_node_idx})

    def reset(self):
        with torch.no_grad():
            if TIME_DEBUG: s = time.time()
            obs_list = self.envs.reset()
            if not self.is_vector_env:
                obs_list = [obs_list]
            self.object_positions = [[]] * self.B
            self.object_categories = [[]] * self.B
            self.goal_position = [[]] * self.B

            obs_batch = batch_obs(obs_list, device=self.torch_device)
            curr_vis_embeddings = self.embed_obs(obs_batch)
            curr_object_embedding = self.embed_object(obs_batch)
            if self.need_goal_embedding: obs_batch['curr_embedding'] = curr_vis_embeddings
            self.localize(curr_vis_embeddings, obs_batch['position'].detach().cpu().numpy(), obs_batch['relpose'].detach().cpu().numpy(),
                          obs_batch['step'].detach().cpu(), [True] * self.B, curr_object_embedding, obs_batch)
            global_memory_dict = self.get_global_memory()
            obs_batch = self.update_memory(obs_batch, global_memory_dict)
            self.update_object_graph(curr_object_embedding, obs_batch, [True] * self.B)
            object_memory_dict = self.get_object_memory()
            obs_batch.update(object_memory_dict)
            obs_batch.update({'object_localized_idx': self.objectgraph.last_localized_node_idx})
            if self.args.render > 0 or self.args.record > 0:
                self.update_graph()
            if TIME_DEBUG: s = log_time(s, 'reset')

        return obs_batch

    def get_global_memory(self, mode='feature'):
        max_num_node = self.graph.num_node_max()
        global_memory_dict = {
            'global_memory': self.graph.graph_memory[:, :max_num_node],
            'global_memory_relpose': self.graph.graph_memory_pose[:, :max_num_node],
            # 'global_act_memory': self.graph.graph_act_memory[:, :max_num_node],
            'global_mask': self.graph.graph_mask[:, :max_num_node],
            'global_A': self.graph.A[:, :max_num_node, :max_num_node],
            'global_idx': self.graph.last_localized_node_idx,
            'global_time': self.graph.graph_time[:, :max_num_node]
        }
        for k, v in global_memory_dict.items():
            global_memory_dict[k] = v.to(self.torch_device)
        return global_memory_dict

    def get_object_memory(self, mode='feature'):
        max_num_vis_node = self.graph.num_node_max()
        max_num_node = self.objectgraph.num_node_max()
        object_memory_dict = {
            'object_memory': self.objectgraph.graph_memory[:, :max_num_node],
            'object_memory_score': self.objectgraph.graph_score[:, :max_num_node],
            'object_memory_relpose': self.objectgraph.graph_memory_pose[:, :max_num_node],
            # 'object_memory_gtpose': self.objectgraph.graph_memory_gt_pose[:, :max_num_node],
            'object_memory_category': self.objectgraph.graph_category[:, :max_num_node],
            'object_memory_mask': self.objectgraph.graph_mask[:, :max_num_node],
            'object_memory_A_OV': self.objectgraph.A_OV[:, :max_num_node, :max_num_vis_node],
            # 'object_memory_A_VV': self.graph.A[:, :max_num_vis_node, :max_num_vis_node],
            'object_memory_time': self.objectgraph.graph_time[:, :max_num_node]
        }
        for k, v in object_memory_dict.items():
            object_memory_dict[k] = v.to(self.torch_device)
        return object_memory_dict

    def call(self, aa, bb):
        return self.envs.call(aa, bb)

    def log_info(self, log_type='str', info=None):
        return self.envs.log_info(log_type, info)

    @property
    def habitat_env(self):
        return self.envs.habitat_env

    @property
    def noise(self):
        return self.envs.noise

    @property
    def current_episode(self):
        if self.is_vector_env:
            return self.envs.current_episodes
        else:
            return self.envs.current_episode

    @property
    def current_episodes(self):
        return self.envs.current_episodes

    def add_sensor(self, obs):
        obs_dict = {}
        obs_dict['target_object_category'] = []
        # if 'mp3d' in self.exp_config.DATASET.DATA_PATH:
        obs_dict['target_object_category'].append(CATEGORIES[self.dn].index(obs['goal_category']))
        # else:
        #     obs_dict['target_object_category'].append(CATEGORIES[self.dn].index(self.goal_name))
        if self.args.mode == "eval":
            goal_obj_idx = np.where(obs_dict['target_object_category'][0] == obs['object_category'][obs['object_score'] > 0.75])[0]
            if len(goal_obj_idx) > 0:
                obs_dict['goal_sensor'] = [np.clip(float(np.max(1. / (obs['object_depth'][obs['object_score'] > 0.75][goal_obj_idx] + 0.001))), 0, 1.)]
            else:  # np.maximum(1-np.array(obs_dict['distance'])/2.,0.0)
                obs_dict['goal_sensor'] = [0.]
        else:
            obs_dict['goal_sensor'] = [0.]
            try:
                gt_object_category = obs['object_category'][obs['object_score'] == 1]
                if obs_dict['target_object_category'][0] in gt_object_category:
                    goal_obj_idx = np.where([obs_dict['target_object_category'][0] == i for i in gt_object_category])[0]
                    x = obs['object_depth'][obs['object_score'] == 1][goal_obj_idx]
                    obs_dict['goal_sensor'] = [np.max(np.clip(-(x - 2) * 2 / 3, 0, 1))]
            except:
                obs_dict['goal_sensor'] = [0.]
        return obs_dict


