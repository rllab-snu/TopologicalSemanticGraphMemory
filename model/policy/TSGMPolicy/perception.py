import torch
import torch.nn.functional as F
from .graph_layer import GraphConvolution, GraphAttention
import torch.nn as nn
import math
from model.utils import CategoryEncoding

class Attblock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.nhead = nhead
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, trg, src_mask):
        # q = k = self.with_pos_embed(src, pos)
        q = src.permute(1, 0, 2)
        k = trg.permute(1, 0, 2)
        src_mask = ~src_mask.bool()
        src2, attention = self.attn(q, k, value=k, key_padding_mask=src_mask)
        src2 = src2.permute(1, 0, 2)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attention


class CrossGCN(nn.Module):
    def __init__(self, cfg, dropout=0.1, obj_dropout=0.2, hidden_dim=4, init='xavier'):
        super(CrossGCN, self).__init__()
        self.cfg = cfg
        self.gc_img1 = GraphAttention(cfg.features.visual_feature_dim, hidden_dim, init=init, dropout=dropout)
        self.gc_img2 = GraphAttention(hidden_dim, hidden_dim, init=init, dropout=dropout)

        self.gc_obj1 = GraphAttention(cfg.features.object_feature_dim, hidden_dim, init=init, dropout=obj_dropout)
        self.gc_obj2 = GraphAttention(hidden_dim, hidden_dim, init=init, dropout=obj_dropout)

        self.globalTobject = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hidden_dim, hidden_dim))
        self.objectTglobal = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hidden_dim, hidden_dim))
        self.img_out = nn.Linear(hidden_dim, cfg.features.visual_feature_dim)
        self.obj_out = nn.Linear(hidden_dim, cfg.features.object_feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, vis_memory, obj_memory, vis_adj, obj_adj, object_A_OV):
        BV, NV = vis_memory.shape[0], vis_memory.shape[1]
        BO, NO = obj_memory.shape[0], obj_memory.shape[1]

        # construct big version of graph
        with torch.no_grad():
            cross_vo_adj = (object_A_OV.permute(0,2,1)) / ((object_A_OV.permute(0,2,1)).sum(2, True) + 0.00001)
            big_vo_adj = torch.zeros(BV * NV, BO * NO).to(vis_memory.device)
            for b in range(BV):
                big_vo_adj[b * NV:(b + 1) * NV, b * NO:(b + 1) * NO] = cross_vo_adj[b]
            big_ov_adj = big_vo_adj.t() / (big_vo_adj.t().sum(1, True)+0.00001)
            big_obj_graph = torch.cat([graph for graph in obj_memory], 0)
            big_obj_adj = torch.zeros(BO * NO, BO * NO).to(obj_memory.device)
            for b in range(BO):
                big_obj_adj[b * NO:(b + 1) * NO, b * NO:(b + 1) * NO] = obj_adj[b]
            big_vis_graph = torch.cat([graph for graph in vis_memory], 0)
            big_vis_adj = torch.zeros(BV * NV, BV * NV).to(vis_memory.device)
            for b in range(BV):
                big_vis_adj[b * NV:(b + 1) * NV, b * NV:(b + 1) * NV] = vis_adj[b]

        # graph convolution on image graph / object graph
        big_vis_graph_ = self.dropout(F.relu(self.gc_img1(big_vis_graph, big_vis_adj))) #message passing function
        big_obj_graph_ = self.dropout(F.relu(self.gc_obj1(big_obj_graph, big_obj_adj)))

        # update image node with object node
        big_vis_graph = big_vis_graph_ + self.objectTglobal(torch.matmul(big_vo_adj, big_obj_graph_)) #update function
        # update object node with image node
        big_obj_graph = big_obj_graph_ + self.globalTobject(torch.matmul(big_ov_adj, big_vis_graph_))

        big_vis_graph_ = self.dropout(F.relu(self.gc_img2(big_vis_graph, big_vis_adj)))
        big_vis_adj = None
        big_obj_graph_ = self.dropout(F.relu(self.gc_obj2(big_obj_graph, big_obj_adj)))
        big_obj_adj = None

        # update image node with object node
        big_vis_graph = big_vis_graph_ + self.objectTglobal(torch.matmul(big_vo_adj, big_obj_graph_))
        big_vo_adj = None
        # update object node with image node
        big_obj_graph = big_obj_graph_ + self.globalTobject(torch.matmul(big_ov_adj, big_vis_graph_))
        big_ov_adj = None

        big_vis_output = torch.stack(big_vis_graph.split(NV))
        big_vis_output = self.img_out(big_vis_output)
        big_obj_output = torch.stack(big_obj_graph.split(NO))
        big_obj_output = self.obj_out(big_obj_output)

        return big_vis_output, big_obj_output


class PositionEncoding(nn.Module):
    def __init__(self, n_filters=512, max_len=2000):
        """
        :param n_filters: same with input hidden size
        :param max_len: maximum sequence length
        """
        super(PositionEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_filters)  # (L, D)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_filters, 2).float() * - (math.log(10000.0) / n_filters))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # buffer is a tensor, not a variable, (L, D)

    def forward(self, x, times):
        """
        :Input: (*, L, D)
        :Output: (*, L, D) the same size as input
        """
        pe = []
        for b in range(x.shape[0]):
            pe.append(self.pe.data[times[b].long()])  # (#x.size(-2), n_filters)
        pe_tensor = torch.stack(pe)
        x = x + pe_tensor
        return x

class Perception(nn.Module):
    def __init__(self, cfg, device):
        super(Perception, self).__init__()
        self.cfg = cfg
        self.torch_device = device
        self.pe_method = 'pe'  # or exp(-t)
        self.time_embedd_size = cfg.features.time_dim
        self.max_time_steps = cfg.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS
        self.goal_time_embedd_index = self.max_time_steps
        memory_dim = cfg.features.visual_feature_dim
        if self.pe_method == 'embedding':
            self.time_embedding = nn.Embedding(self.max_time_steps + 2, self.time_embedd_size)
        elif self.pe_method == 'pe':
            self.time_embedding = PositionEncoding(memory_dim, self.max_time_steps + 10)
            self.obj_time_embedding = PositionEncoding(cfg.features.object_feature_dim, self.max_time_steps + 10)
        else:
            self.time_embedding = lambda t: torch.exp(-t.unsqueeze(-1) / 5)

        feature_dim = cfg.features.visual_feature_dim
        self.feature_embedding = nn.Sequential(nn.Linear(feature_dim + cfg.features.visual_feature_dim, memory_dim),
                                               nn.ReLU(),
                                               nn.Linear(memory_dim, memory_dim))
        self.object_embedding = nn.Sequential(nn.Linear((cfg.features.object_feature_dim + cfg.features.visual_feature_dim), cfg.features.object_feature_dim),
                                              nn.ReLU(),
                                              nn.Linear(cfg.features.object_feature_dim, cfg.features.object_feature_dim))

        self.goal_Decoder = Attblock(cfg.transformer.hidden_dim,
                                     cfg.transformer.nheads,
                                     cfg.transformer.dim_feedforward,
                                     cfg.transformer.dropout)
        self.cGCN = CrossGCN(cfg)
        self.vis_Decoder = Attblock(cfg.transformer.hidden_dim,
                                     cfg.transformer.nheads,
                                     cfg.transformer.dim_feedforward,
                                     cfg.transformer.dropout)
        self.obj_Decoder = Attblock(cfg.features.object_feature_dim,
                                    cfg.transformer.nheads,
                                    cfg.transformer.dim_feedforward,
                                    cfg.transformer.dropout)
        self.goal_obj_Decoder = Attblock(cfg.features.object_feature_dim,
                                         cfg.transformer.nheads,
                                         cfg.transformer.dim_feedforward,
                                         cfg.transformer.dropout)
        self.obj_category_embedding = CategoryEncoding(cfg.features.object_feature_dim, cfg.features.object_category_num)
        self.Cat = nn.Sequential(nn.Linear((cfg.features.object_feature_dim * 2), cfg.features.object_feature_dim),
                                              nn.ReLU(),
                                              nn.Linear(cfg.features.object_feature_dim, cfg.features.object_feature_dim))
        self.output_size = feature_dim

    def forward(self, observations, embeddings):
        B = observations['img_memory_mask'].shape[0]
        max_node_num = observations['img_memory_mask'].sum(dim=1).max().long()
        global_relative_time = observations['step'].unsqueeze(1) - observations['img_memory_time'][:, :max_node_num]
        img_memory = self.time_embedding(observations['img_memory_feat'][:, :max_node_num], global_relative_time)
        img_memory_mask = observations['img_memory_mask'][:, :max_node_num]
        I = torch.eye(max_node_num).unsqueeze(0).repeat(B, 1, 1).cuda()
        img_memory_A = observations['img_memory_A'][:, :max_node_num, :max_node_num] + I

        max_obj_node_num = observations['obj_memory_mask'].sum(dim=1).max().long()
        object_relative_time = observations['step'].unsqueeze(1) - observations['obj_memory_time'][:, :max_obj_node_num]
        obj_memory = observations['obj_memory_feat'][:, :max_obj_node_num]
        obj_memory = self.Cat(torch.cat([obj_memory, self.obj_category_embedding(observations['obj_memory_category'][:, :max_obj_node_num])], -1))
        obj_memory = self.obj_time_embedding(obj_memory, object_relative_time)
        object_mask = observations['obj_memory_mask'][:, :max_obj_node_num]
        object_A_OV = observations['obj_memory_A_OV'][:, :max_obj_node_num, :max_node_num].float()
        object_A = (torch.matmul(torch.matmul(object_A_OV, img_memory_A), object_A_OV.permute(0, 2, 1))>0).float()

        img_memory_with_goal = self.feature_embedding(torch.cat((img_memory[:,:max_node_num], embeddings['goal_embedding'].unsqueeze(1).repeat(1,max_node_num,1)),-1))
        obj_memory_with_goal = self.object_embedding(torch.cat((obj_memory[:,:max_obj_node_num], embeddings['goal_embedding'].unsqueeze(1).repeat(1,max_obj_node_num,1)),-1))

        global_context, object_context = self.cGCN(img_memory_with_goal, obj_memory_with_goal, img_memory_A, object_A, object_A_OV)
        global_context = self.time_embedding(global_context, global_relative_time)
        object_context = self.obj_time_embedding(object_context, object_relative_time)

        object_mask[object_mask.sum(1) == 0] = 1
        curr_obj_context, curr_obj_attn = self.obj_Decoder(embeddings['curr_obj_embedding'], object_context, object_mask)
        goal_obj_context, goal_obj_attn = self.goal_obj_Decoder(embeddings['target_obj_embedding'], object_context, object_mask)
        goal_obj_context = goal_obj_context.squeeze(1)
        goal_context, goal_attn = self.goal_Decoder(embeddings['goal_embedding'].unsqueeze(1), global_context, img_memory_mask)
        goal_context = goal_context.squeeze(1)
        curr_context, curr_attn = self.vis_Decoder(embeddings['curr_embedding'].unsqueeze(1), global_context, img_memory_mask)
        curr_context = curr_context.squeeze(1)

        return_f = {'goal_attn': goal_attn, 'curr_attn': curr_attn, 'curr_obj_attn': curr_obj_attn, 'goal_obj_attn': goal_obj_attn}
        return curr_context, goal_context, curr_obj_context, goal_obj_context, return_f
