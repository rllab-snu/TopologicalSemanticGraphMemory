from runner.base_runner import BaseRunner
import torch

from gym.spaces.dict import Dict as SpaceDict
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
import numpy as np
from env_utils.env_wrapper import *
from model.policy import *
from env_utils import *


class TSGMRunner(BaseRunner):
    def __init__(self, args, config, return_features=False):
        super().__init__(args, config)
        observation_space = SpaceDict({
            'panoramic_rgb': Box(low=0, high=256, shape=(64, 256, 3), dtype=np.float32),
            'panoramic_depth': Box(low=0, high=256, shape=(64, 256, 1), dtype=np.float32),
            'panoramic_semantic': Box(low=0, high=256, shape=(64, 256, 1), dtype=np.float32),
            'target_goal': Box(low=0, high=256, shape=(64, 256, 3), dtype=np.float32),
            'step': Box(low=0, high=500, shape=(1,), dtype=np.float32),
            'prev_act': Box(low=0, high=3, shape=(1,), dtype=np.int32),
            'gt_action': Box(low=0, high=3, shape=(1,), dtype=np.int32)
        })
        action_space = Discrete(config.ACTION_DIM)
        print(config.POLICY, 'using ', eval(config.POLICY))
        agent = eval(config.POLICY)(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.features.hidden_size,
            rnn_type=config.features.rnn_type,
            num_recurrent_layers=config.features.num_recurrent_layers,
            backbone=config.features.backbone,
            goal_sensor_uuid=config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
            normalize_visual_inputs=True,
            cfg=config
        )
        self.agent = agent
        self.torch_device = 'cpu' if args.gpu == '-1' else 'cuda:{}'.format(args.gpu)
        # self.torch_device = 'cuda'
        self.return_features = return_features
        self.need_env_wrapper = True
        self.num_agents = 1
        self.config = config

    def reset(self):
        self.B = 1
        self.hidden_states = torch.zeros(self.agent.net.num_recurrent_layers, self.B,
                                         self.agent.net._hidden_size).to(self.torch_device)
        self.actions = torch.zeros([self.B]).to(self.torch_device)
        self.time_t = 0

    def step(self, obs, reward, done, info, env=None):
        new_obs = {}
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                new_obs[k] = torch.from_numpy(v).float().to(self.torch_device).unsqueeze(0)
            if not isinstance(v, torch.Tensor):
                new_obs[k] = torch.tensor(v).float().to(self.torch_device).unsqueeze(0)
            else:
                new_obs[k] = v.to(self.torch_device)
        obs = new_obs
        (
            values,
            actions,
            actions_log_probs,
            hidden_states,
            actions_logits,
            preds,
            act_features
        ) = self.agent.act(
            obs,
            self.hidden_states,
            self.actions,
            torch.ones(self.B).unsqueeze(1).to(self.torch_device) * (1-done),
            deterministic=False,
            return_features=self.return_features
        )
        self.features = act_features
        if preds[0] is not None:
            have_been = torch.sigmoid(preds[0][0])
            have_been_str = 'have_been: '
            have_been_str += '%.3f '%(have_been.item())
        else: have_been_str = ''
        if preds[1] is not None:
            pred_target_distance = torch.sigmoid(preds[1][0])
            pred_dist_str = 'pred_prog: '
            pred_dist_str += '%.3f '%(pred_target_distance.item())
        else: pred_dist_str = ''
        try:
            if preds[3] is not None:
                is_target = torch.max(torch.sigmoid(preds[3][0]))
                is_target_str = 'pred_target: '
                is_target_str += '%.3f '%(is_target.item())
            else: is_target_str = ''
        except:
            is_target_str = ''

        log_str = have_been_str + ' ' + pred_dist_str +  ' ' + is_target_str# + ' ' + have_seen_str +  ' ' + is_target_str
        self.env.log_info(log_type='str', info=log_str)
        self.hidden_states = hidden_states
        self.actions = actions
        self.time_t += 1
        return self.actions.item()

    def visualize(self, env_img):
        return NotImplementedError

    def setup_env(self):
        return

    def wrap_env(self, env, config):
        self.env = eval(config.WRAPPER)(env, config)
        return self.env

    def get_mean_dist_btw_nodes(self):
        # assume batch size is 1
        dists = []
        for node_idx in range(len(self.node_list[0])):
            neighbors = torch.where(self.A[0, node_idx])[0]
            curr_node_position = self.node_list[0][node_idx].cpu().numpy()
            curr_dists = []
            for neighbor in neighbors:
                if neighbor <= node_idx: continue
                dist = self.env.habitat_env._sim.geodesic_distance(curr_node_position,
                                                                   self.node_list[0][neighbor].cpu().numpy())
                if np.isnan(dist):
                    dist = np.linalg.norm(curr_node_position - self.node_list[0][neighbor].cpu().numpy())
                curr_dists.append(dist)
            if len(curr_dists) > 0:
                dists.append(min(curr_dists))
        return dists

    def save(self, file_name=None, epoch=0, step=0):
        if file_name is not None:
            save_dict = {}
            save_dict['config'] = self.config
            save_dict['trained'] = [epoch, step]
            save_dict['state_dict'] = self.agent.state_dict()
            torch.save(save_dict, file_name)
