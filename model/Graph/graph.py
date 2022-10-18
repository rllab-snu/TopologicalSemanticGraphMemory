import numpy as np


class ImgGraph(object):
    def __init__(self, cfg):
        self.memory = None
        self.memory_mask = None
        self.memory_time = None
        self.memory_num = 0
        self.input_shape = cfg.IMG_SHAPE
        self.feature_dim = cfg.memory.img_embedding_dim
        self.M = cfg.memory.memory_size
        self.node_th = cfg.TASK_CONFIG.img_node_th

    def num_node(self):
        return len(self.node_position_list)

    def reset(self):
        self.node_position_list = []  # This position list is only for visualizations
        self.node_rotation_list = []  # This position list is only for visualizations
        self.graph_memory = np.zeros([self.M, self.feature_dim])
        self.A = np.zeros([self.M, self.M], dtype=np.bool)
        self.distance_mat = np.full([self.M, self.M], fill_value=float('inf'), dtype=np.float32)
        self.connectivity_mat = np.full([self.M, self.M], fill_value=0, dtype=np.float32)
        self.graph_mask = np.zeros(self.M)
        self.graph_time = np.zeros(self.M)
        self.pre_last_localized_node_idx = np.zeros([1], dtype=np.int32)
        self.last_localized_node_idx = np.zeros([1], dtype=np.int32)
        self.last_local_node_num = np.zeros([1])
        self.last_localized_node_embedding = np.zeros([self.feature_dim], dtype=np.float32)
        self.found = True

    def initialize_graph(self, new_embeddings, positions, rotations):
        self.add_node(node_idx=0, embedding=new_embeddings, time_step=0, position=positions, rotation=rotations)
        self.record_localized_state(node_idx=0, embedding=new_embeddings)

    def add_node(self, node_idx, embedding, time_step, position, rotation, dists=None, connectivity=None):
        self.node_position_list.append(position)
        self.node_rotation_list.append(rotation)
        self.graph_memory[node_idx] = embedding
        self.graph_mask[node_idx] = 1.0
        self.graph_time[node_idx] = time_step
        if dists is not None:
            self.distance_mat[node_idx, :node_idx] = dists
            self.distance_mat[:node_idx, node_idx] = dists
        if connectivity is not None:
            self.connectivity_mat[node_idx, :node_idx] = connectivity
            self.connectivity_mat[:node_idx, node_idx] = connectivity

    def record_localized_state(self, node_idx, embedding):
        self.pre_last_localized_node_idx = self.last_localized_node_idx
        self.last_localized_node_idx = node_idx
        self.last_localized_node_embedding = embedding

    def add_edge(self, node_idx_a, node_idx_b):
        self.A[node_idx_a, node_idx_b] = True
        self.A[node_idx_b, node_idx_a] = True

    def update_node(self, node_idx, time_info, embedding=None):
        if embedding is not None:
            self.graph_memory[node_idx] = embedding
        self.graph_time[node_idx] = time_info


class ObjGraph(object):
    def __init__(self, cfg):
        self.memory = None
        self.memory_mask = None
        self.memory_time = None
        self.memory_num = 0
        self.feature_dim = cfg.features.object_feature_dim
        self.M = (cfg.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS * cfg.memory.num_objects) // 2
        self.MV = cfg.memory.memory_size
        self.num_obj = cfg.memory.num_objects
        self.sparse = cfg.OBJECTGRAPH.SPARSE
        self.node_th = cfg.TASK_CONFIG.obj_node_th

    def num_node(self):
        return len(self.node_position_list)

    def reset(self):
        self.node_position_list = []  # This position list is only for visualizations
        self.graph_memory = np.zeros([self.M, self.feature_dim])
        self.graph_category = np.zeros([self.M])
        self.graph_score = np.zeros([self.M])
        self.A_OV = np.zeros([self.M, self.MV], dtype=np.bool)
        self.graph_mask = np.zeros(self.M)
        self.graph_time = np.zeros([self.M], dtype=np.int32)
        self.last_localized_node_idx = 0

    def initialize_graph(self, new_embeddings, object_scores, object_categories, masks, positions):
        if sum(masks == 1) == 0:
            masks[0] = 1
        self.add_node(node_idx=0, embedding=new_embeddings, object_score=object_scores, object_category=object_categories, time_step=0, mask=masks, position=positions, vis_node_idx=0)

    def add_node(self, node_idx, embedding, object_score, object_category, mask, time_step, position, vis_node_idx):
        node_idx_ = node_idx
        i = 0
        while True:
            if mask[i] == 1:
                self.node_position_list.append(position[i])
                self.graph_memory[node_idx_] = embedding[i]
                self.graph_score[node_idx_] = object_score[i]
                self.graph_category[node_idx_] = object_category[i]
                self.graph_mask[node_idx_] = 1.0
                self.graph_time[node_idx_] = time_step
                self.add_vo_edge([node_idx_], vis_node_idx)
                node_idx_ += 1
            i += 1
            if i == len(position):
                break

    def add_vo_edge(self, node_idx_obj, curr_vis_node_idx):
        for node_idx_obj_i in node_idx_obj:
            self.A_OV[node_idx_obj_i, curr_vis_node_idx] = True

    def update_node(self, node_idx, time_info, node_score, node_category, curr_vis_node_idx, embedding=None):
        if embedding is not None:
            self.graph_memory[node_idx] = embedding
        self.graph_score[node_idx] = node_score
        self.graph_category[node_idx] = node_category
        self.graph_time[node_idx] = time_info
        self.A_OV[node_idx, :] = False
        self.A_OV[node_idx, curr_vis_node_idx] = True