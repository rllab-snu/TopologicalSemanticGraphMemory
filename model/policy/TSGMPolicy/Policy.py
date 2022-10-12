import torch
import torch.nn as nn
from model.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.common.utils import CategoricalNet
from model.resnet import resnet
from model.resnet.resnet import ResNetEncoder
from .perception import Perception
from env_utils import *
from torch.autograd import Variable
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from utils.augmentations import GaussianBlur
from .perception import CategoryEncoding


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class TSGMPolicy(nn.Module):
    def __init__(
            self,
            observation_space,
            action_space,
            goal_sensor_uuid="pointgoal_with_gps_compass",
            hidden_size=512,
            num_recurrent_layers=2,
            rnn_type="GRU",
            resnet_baseplanes=32,
            backbone="resnet50",
            normalize_visual_inputs=True,
            cfg=None
    ):
        super().__init__()
        self.cfg = cfg
        self.net = TSGMNet(
            observation_space=observation_space,
            action_space=action_space,
            goal_sensor_uuid=goal_sensor_uuid,
            hidden_size=hidden_size,
            num_recurrent_layers=num_recurrent_layers,
            rnn_type=rnn_type,
            backbone=backbone,
            resnet_baseplanes=resnet_baseplanes,
            normalize_visual_inputs=normalize_visual_inputs,
            cfg=cfg
        )
        self.dim_actions = action_space.n

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.net.output_size)

    def act(
            self,
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            deterministic=False,
            return_features=False,
    ):

        features, rnn_hidden_states, preds, ffeatures, = self.net(
            observations, rnn_hidden_states, prev_actions, masks, return_features=return_features
        )
        features = torch.where(torch.isnan(features), Variable(torch.ones_like(features)*0.0001, requires_grad=False).to(features.device), features)
        distribution, x = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()
        action_log_probs = distribution.log_probs(action)
        # The shape of the output should be B * N * (shapes)
        if return_features:
            return value, action, action_log_probs, rnn_hidden_states, x, preds, ffeatures
        return value, action, action_log_probs, rnn_hidden_states, x, preds, None

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, *_ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        value = self.critic(features)
        return value

    def evaluate_actions(
            self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, rnn_hidden_states, preds, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution, x = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states, preds


class TSGMNet(nn.Module):
    def __init__(
            self,
            observation_space,
            action_space,
            goal_sensor_uuid,
            hidden_size,
            num_recurrent_layers,
            rnn_type,
            backbone,
            resnet_baseplanes,
            normalize_visual_inputs,
            cfg
    ):
        super().__init__()
        self.cfg = cfg
        self.goal_sensor_uuid = goal_sensor_uuid
        self.train_il = self.cfg.TASK_CONFIG.TRAIN_IL

        self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
        self._n_prev_action = 32
        self._n_input_goal = 0
        self._hidden_size = hidden_size

        rnn_input_size = self._n_input_goal + self._n_prev_action
        self.img_encoder = ResNetEncoder(
            cfg,
            observation_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
            feature_dim=cfg.features.visual_feature_dim
        )

        self.torch_device = "cuda:" + str(self.cfg.TORCH_GPU_ID) if torch.cuda.device_count() > 0 else 'cpu'
        self.perception_unit = Perception(cfg, self.torch_device)
        self.visual_fc = nn.Sequential(
            nn.Linear(
                cfg.features.visual_feature_dim * 3, hidden_size * 2
            ),
            nn.ReLU(True),
            nn.Linear(
                hidden_size * 2, hidden_size
            ),
            nn.ReLU(True),
        )
        self.obj_fc = nn.Sequential(
            nn.Linear(
                cfg.features.object_feature_dim * 3, hidden_size * 2
            ),
            nn.ReLU(True),
            nn.Linear(
                hidden_size * 2, cfg.features.object_feature_dim
            )
        )

        self.pred_aux2 = nn.Sequential(nn.Linear(cfg.features.visual_feature_dim * 2, cfg.features.visual_feature_dim),
                                       nn.ReLU(),
                                       nn.Linear(cfg.features.visual_feature_dim, 1))
        self.pred_aux4 = nn.Sequential(nn.Linear(cfg.features.visual_feature_dim * 2, cfg.features.visual_feature_dim),
                                       nn.ReLU(),
                                       nn.Linear(cfg.features.visual_feature_dim, 1))

        self.state_encoder_vis = RNNStateEncoder(
            (0 if self.is_blind else self._hidden_size) + rnn_input_size + 1,
            int(self._hidden_size/2),
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )
        self.state_encoder_obj = RNNStateEncoder(
            (0 if self.is_blind else cfg.memory.num_objects*cfg.features.object_feature_dim) + rnn_input_size,
            int(self._hidden_size/2),
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()
        if self.training:
            self.initialize()

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            normalize
        ])
        self.transform_eval = transforms.Compose([
            transforms.ToTensor(),
            normalize
            ])
        self.obj_category_embedding = CategoryEncoding(cfg.features.object_feature_dim, cfg.features.object_category_num)
        self.Cat = nn.Sequential(nn.Linear((cfg.features.object_feature_dim * 2), cfg.features.object_feature_dim),
                                              nn.ReLU(),
                                              nn.Linear(cfg.features.object_feature_dim, cfg.features.object_feature_dim))

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.img_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder_vis.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks, mode='', return_features=False):
        prev_actions = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().squeeze(-1)
        )
        depth = observations['panoramic_depth']
        with torch.no_grad():
            if self.train_il:
                panoramic_rgb = observations['panoramic_rgb_trans']
            else:
                panoramic_rgb = []
                for i in range(len(observations['panoramic_rgb'])):
                    if self.training:
                        im = observations['panoramic_rgb'][i].cpu().detach().numpy()
                        im = Image.fromarray(np.uint8(im))
                        panoramic_rgb.append(self.transform(im))
                    else:
                        im = Image.fromarray(np.uint8(observations['panoramic_rgb'][i].cpu().detach().numpy()))
                        panoramic_rgb.append(self.transform_eval(im))
                panoramic_rgb = torch.stack(panoramic_rgb).to(observations['panoramic_depth'].device)
            input_list = [panoramic_rgb, depth.permute(0, 3, 1, 2)]
            curr_tensor = torch.cat(input_list, 1)
            curr_tensor = curr_tensor.to(self.torch_device)

            if self.train_il:
                goal_tensor = observations['target_goal_trans']
            else:
                goal_tensor = []
                for i in range(len(observations['target_goal'])):
                    im = Image.fromarray(np.uint8(observations['target_goal'][i][:,:,:3].cpu().detach().numpy() * 255.))
                    goal_tensor.append(torch.cat([self.transform_eval(im).to(observations['target_goal'].device), observations['target_goal'][i][:,:,3][None]], 0))
                goal_tensor = torch.stack(goal_tensor, 0)
            goal_tensor = goal_tensor.to(self.torch_device)
        embeddings = {}
        embeddings['curr_embedding'] = self.img_encoder(curr_tensor).view(curr_tensor.shape[0], -1)
        embeddings['goal_embedding'] = self.img_encoder(goal_tensor).view(goal_tensor.shape[0], -1)
        curr_obj_embedding = self.img_encoder.embed_object(curr_tensor, observations['object'].to(self.torch_device)).view(curr_tensor.shape[0], observations['object'].shape[1], -1)
        embeddings['curr_obj_embedding'] = self.Cat(torch.cat([curr_obj_embedding, self.obj_category_embedding(observations['object_category'])], -1))
        target_obj_embedding = self.img_encoder.embed_object(goal_tensor, observations['target_loc_object'].to(self.torch_device)).view(goal_tensor.shape[0], observations['target_loc_object'].shape[1], -1)
        embeddings['target_obj_embedding'] = self.Cat(torch.cat([target_obj_embedding, self.obj_category_embedding(observations['target_loc_object_category'].to(self.torch_device))], -1))

        curr_context, goal_context, curr_obj_context, goal_obj_context, ffeatures = self.perception_unit(observations, embeddings)

        vis_contexts = torch.cat((curr_context, goal_context), -1)
        obj_context = torch.cat((curr_obj_context, goal_obj_context), -1)

        vis_feats = self.visual_fc(torch.cat((vis_contexts, embeddings['curr_embedding']), 1))
        obj_feats = self.obj_fc(torch.cat((obj_context, embeddings['curr_obj_embedding']), -1)).flatten(1)
        rnn_hidden_states_vis, rnn_hidden_states_obj = rnn_hidden_states.split(int(rnn_hidden_states.shape[-1]/2), dim=-1)

        progress = self.pred_aux2(vis_contexts) #progress
        goal = self.pred_aux4(vis_contexts) #is target
        preds = (progress, goal)

        vis_x, rnn_hidden_states_vis = self.state_encoder_vis(torch.cat([vis_feats, torch.sigmoid(goal), prev_actions], dim=1), rnn_hidden_states_vis, masks)
        obj_x, rnn_hidden_states_obj = self.state_encoder_obj(torch.cat([obj_feats, prev_actions], dim=1), rnn_hidden_states_obj, masks)
        x = torch.cat([vis_x, obj_x], -1)
        rnn_hidden_states = torch.cat([rnn_hidden_states_vis, rnn_hidden_states_obj], -1)

        return x, rnn_hidden_states, preds, ffeatures

    def initialize(self):
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
                try:
                    m.bias.data.fill_(0.001)
                except:
                    pass
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
            elif classname.find('Linear') != -1:
                m.weight.data.normal_(0.0, 0.02)
                try:
                    m.bias.data.fill_(0.001)
                except:
                    pass
            elif classname.find('LSTM') != -1:
                m.weight_hh_l0.data.normal_(0.0, 0.02)
                m.weight_ih_l0.data.normal_(0.0, 0.02)
                m.weight_hh_l1.data.normal_(0.0, 0.02)
                m.weight_ih_l1.data.normal_(0.0, 0.02)
                try:
                    m.bias_hh_l0.data.fill_(0.001)
                    m.bias_ih_l0.data.fill_(0.001)
                    m.bias_hh_l1.data.fill_(0.001)
                    m.bias_ih_l1.data.fill_(0.001)
                except:
                    pass
            elif classname.find('GRU') != -1 and classname.find('GRUNet') == -1:
                m.weight_hh_l0.data.normal_(0.0, 0.02)
                m.weight_ih_l0.data.normal_(0.0, 0.02)
                try:
                    m.bias_hh_l0.data.fill_(0.001)
                    m.bias_ih_l0.data.fill_(0.001)
                except:
                    pass

        self.perception_unit.apply(weights_init)
        self.visual_fc.apply(weights_init)
        self.obj_fc.apply(weights_init)
        self.state_encoder_vis.apply(weights_init)
        self.state_encoder_obj.apply(weights_init)
        self.pred_aux2.apply(weights_init)
        self.pred_aux4.apply(weights_init)
