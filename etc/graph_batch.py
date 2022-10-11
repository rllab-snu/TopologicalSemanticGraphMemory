import torch
import numpy as np


class Node(object):
    def __init__(self, info=None):
        self.node_num = None
        self.time_t = None
        self.neighbors = []
        self.neighbors_node_num = []
        self.embedding = None
        self.misc_info = None
        self.action = -1
        self.visited_time = []
        self.visited_memory = []
        if info is not None:
            for k, v in info.items():
                setattr(self, k, v)


class Graph(object):
    def __init__(self, cfg, B):
        self.B = B
        self.memory = None
        self.memory_mask = None
        self.memory_time = None
        self.memory_num = 0
        self.input_shape = cfg.IMG_SHAPE
        self.feature_dim = cfg.memory.embedding_size
        self.M = cfg.memory.memory_size
        self.torch_device = "cpu"#device

    def num_node(self, b):
        return len(self.node_position_list[b])

    def num_node_max(self):
        return self.graph_mask.sum(dim=1).max().long().item()

    def reset(self, B):
        if B: self.B = B
        self.node_position_list = [[] for _ in range(self.B)]  # This position list is only for visualizations

        self.graph_memory = torch.zeros([self.B, self.M, self.feature_dim]).to(self.torch_device)
        # self.graph_act_memory = torch.zeros([self.B, self.M], dtype=torch.uint8).to(self.torch_device)
        self.graph_memory_pose = torch.zeros([self.B, self.M, 3], dtype=torch.uint8).to(self.torch_device)

        self.A = torch.zeros([self.B, self.M, self.M], dtype=torch.bool).to(self.torch_device)
        self.distance_mat = torch.full([self.B, self.M, self.M], fill_value=float('inf'), dtype=torch.float32).to(self.torch_device)
        self.connectivity_mat = torch.full([self.B, self.M, self.M], fill_value=0, dtype=torch.float32)

        self.graph_mask = torch.zeros(self.B, self.M).to(self.torch_device)
        self.graph_time = torch.zeros(self.B, self.M).to(self.torch_device)

        self.pre_last_localized_node_idx = torch.zeros([self.B], dtype=torch.int32)
        self.last_localized_node_idx = torch.zeros([self.B], dtype=torch.int32)
        self.last_local_node_num = torch.zeros([self.B])
        self.last_localized_node_embedding = torch.zeros([self.B, self.feature_dim], dtype=torch.float32).to(self.torch_device)

    def reset_at(self, b):
        self.graph_memory[b] = 0
        # self.graph_act_memory[b] = 0
        self.graph_memory_pose[b] = 0
        self.A[b] = 0
        self.graph_mask[b] = 0
        self.graph_time[b] = 0
        self.pre_last_localized_node_idx[b] = 0
        self.last_localized_node_idx[b] = 0
        self.last_local_node_num[b] = 0
        self.node_position_list[b] = []
        self.distance_mat[b] = float('inf')
        self.connectivity_mat[b] = 0.0

    def initialize_graph(self, b, new_embeddings, positions, poses):
        self.add_node(b, node_idx=0, embedding=new_embeddings[b], time_step=0, position=positions[b], relpose=poses[b])
        self.record_localized_state(b, node_idx=0, embedding=new_embeddings[b])

    def add_node(self, b, node_idx, embedding, time_step, position, relpose, dists=None, connectivity=None):
        self.node_position_list[b].append(position)
        self.graph_memory[b, node_idx] = embedding
        # self.graph_act_memory[b, node_idx] = action
        self.graph_memory_pose[b, node_idx] = torch.tensor(relpose)
        self.graph_mask[b, node_idx] = 1.0
        self.graph_time[b, node_idx] = time_step
        if dists is not None:
            self.distance_mat[b, node_idx, :node_idx] = torch.tensor(dists)
            self.distance_mat[b, :node_idx, node_idx] = torch.tensor(dists)
        if connectivity is not None:
            self.connectivity_mat[b, node_idx, :node_idx] = torch.tensor(connectivity)
            self.connectivity_mat[b, :node_idx, node_idx] = torch.tensor(connectivity)

    def record_localized_state(self, b, node_idx, embedding):
        self.pre_last_localized_node_idx[b] = self.last_localized_node_idx[b].clone()
        self.last_localized_node_idx[b] = node_idx
        self.last_localized_node_embedding[b] = embedding

    def add_edge(self, b, node_idx_a, node_idx_b):
        self.A[b, node_idx_a, node_idx_b] = 1.0
        self.A[b, node_idx_b, node_idx_a] = 1.0
        return

    def add_edges(self, b, node_idx_as, node_idx_b):
        for node_idx_a in node_idx_as:
            self.A[b, node_idx_a, node_idx_b] = 1.0
            self.A[b, node_idx_b, node_idx_a] = 1.0
        return

    def update_node(self, b, node_idx, time_info, embedding=None):
        if embedding is not None:
            self.graph_memory[b, node_idx] = embedding
        self.graph_time[b, node_idx] = time_info
        return

    def update_nodes(self, bs, node_indices, time_infos, embeddings=None):
        if embeddings is not None:
            self.graph_memory[bs, node_indices] = embeddings
        self.graph_time[bs, node_indices.long()] = time_infos

    def get_positions(self, b, a=None):
        if a is None:
            return self.node_position_list[b]
        else:
            return self.node_position_list[b][a]

    def get_neighbor(self, b, node_idx, return_mask=False):
        if return_mask:
            return self.A[b, node_idx]
        else:
            return torch.where(self.A[b, node_idx])[0]

    def calculate_multihop(self, hop):
        return torch.matrix_power(self.A[:, :self.num_node_max(), :self.num_node_max()].float(), hop)

class ObjectGraph(object):
    def __init__(self, cfg, B):
        self.B = B
        self.memory = None
        self.memory_mask = None
        self.memory_time = None
        self.memory_num = 0
        self.feature_dim = cfg.features.object_feature_dim
        if cfg.TASK_CONFIG['ARGS']['mode'] == "train_bc":
            self.M = cfg.memory.memory_size * cfg.memory.num_objects
        else:
            self.M = cfg.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS * cfg.memory.num_objects
        self.MV = cfg.memory.memory_size
        self.num_obj = cfg.memory.num_objects
        self.torch_device = "cpu"#device
        self.task = cfg['ARGS']['task']

    def num_node(self, b):
        return len(self.node_position_list[b])

    def num_node_max(self):
        return self.graph_mask.sum(dim=1).max().long().item()

    def reset(self, B):
        if B: self.B = B
        self.node_position_list = [[] for _ in range(self.B)]  # This position list is only for visualizations

        self.graph_memory = torch.zeros([self.B, self.M, self.feature_dim]).to(self.torch_device)
        self.graph_memory_pose = torch.zeros([self.B, self.M, 3]).to(self.torch_device)
        self.graph_category = torch.ones([self.B, self.M]).to(self.torch_device) * (-1)
        self.graph_score = torch.zeros([self.B, self.M]).to(self.torch_device)

        self.A_OV = torch.zeros([self.B, self.M, self.MV], dtype=torch.bool).to(self.torch_device)

        self.graph_mask = torch.zeros(self.B, self.M).to(self.torch_device)
        self.graph_time = torch.zeros([self.B, self.M], dtype=torch.int32).to(self.torch_device)

        self.last_localized_node_idx = [[] for _ in range(self.B)]

    def reset_at(self, b):
        self.graph_memory[b] = 0
        self.graph_memory_pose[b] = 0
        self.graph_category[b] = -1
        self.graph_score[b] = 0

        self.A_OV[b] = 0

        self.graph_mask[b] = 0
        self.graph_time[b] = 0

        self.last_localized_node_idx[b] = []
        self.node_position_list[b] = []

    def initialize_graph(self, b, new_embeddings, object_scores, object_categories, masks, positions):
        if sum(masks[b] == 1) == 0:
            masks[b][0] = 1
        self.add_node(b, node_idx=0, embedding=new_embeddings[b], object_score=object_scores[b], object_category=object_categories[b], time_step=0, mask=masks[b],
                      position=positions[b], vis_node_idx=0)

    def add_node(self, b, node_idx, embedding, object_score, object_category, mask, time_step, position, vis_node_idx):
        node_idx_ = node_idx
        i = 0
        while True:
            if self.task == "objgoalnav":
                cond = mask[i] == 1 and np.all(np.sqrt(np.sum((position[i] - self.graph_memory_pose[b].cpu().detach().numpy()) ** 2, 1)) > 0.5)
            else:
                cond = mask[i] == 1
            if cond:
                self.node_position_list[b].append(position[i])
                self.graph_memory[b, node_idx_] = embedding[i].to(self.torch_device)
                self.graph_memory_pose[b, node_idx_] = torch.tensor(position[i]).to(self.torch_device)
                self.graph_score[b, node_idx_] = object_score[i].to(self.torch_device)
                self.graph_category[b, node_idx_] = object_category[i].to(self.torch_device)
                self.graph_mask[b, node_idx_] = 1.0
                self.graph_time[b, node_idx_] = time_step
                self.add_vo_edge(b, [node_idx_], vis_node_idx)
                node_idx_ += 1
            i += 1
            if i == len(position):
                break

    def add_vo_edge(self, b, node_idx_obj, curr_vis_node_idx):
        for node_idx_obj_i in node_idx_obj:
            self.A_OV[b, node_idx_obj_i, curr_vis_node_idx] = 1.0

    def update_node(self, b, node_idx, time_info, node_score, node_category, curr_vis_node_idx, relpose, embedding=None):
        if self.task == "objgoalnav":
            dist = np.sqrt(np.sum((relpose - self.graph_memory_pose[b, node_idx].cpu().detach().numpy())**2))
        else:
            dist = 10
        if (node_category == self.graph_category[b, node_idx] or dist < 0.5) and node_category != -1:
            orig_vis_node_idx = torch.where(self.A_OV[b, node_idx])[0]
            if node_score >= self.graph_score[b, node_idx]:
                # Update
                if embedding is not None:
                    self.graph_memory[b, node_idx] = embedding.cpu()
                self.graph_memory_pose[b, node_idx] = torch.tensor(relpose)
                self.graph_time[b, node_idx] = time_info.cpu()
                self.graph_score[b, node_idx] = node_score.cpu()
                self.graph_category[b, node_idx] = node_category.cpu()
                self.A_OV[b, node_idx, curr_vis_node_idx] = 1
            return True, orig_vis_node_idx, curr_vis_node_idx, node_score
        return False, None, None, None

    def get_positions(self, b, a=None):
        if a is None:
            return self.node_position_list[b]
        else:
            return self.node_position_list[b][a]
