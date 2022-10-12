try:
     from gym.wrappers.monitor import Wrapper
except:
     from gym.wrappers.record_video import RecordVideo as Wrapper
import torch
import numpy as np
from utils.debug_utils import log_time

TIME_DEBUG = False
from utils.habitat_utils import batch_obs

# this wrapper comes after vectorenv
from habitat.core.vector_env import VectorEnv


class EnvWrapper(Wrapper):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, envs, exp_config):
        self.envs = envs
        self.env = self.envs
        if isinstance(envs, VectorEnv):
            self.is_vector_env = True
            self.num_envs = self.envs.num_envs
            self.action_spaces = self.envs.action_spaces
            self.observation_spaces = self.envs.observation_spaces
        else:
            self.is_vector_env = False
            self.num_envs = 1

        self.B = self.num_envs
        self.torch = exp_config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.GPU_GPU
        self.torch_device = 'cuda:' + str(exp_config.TORCH_GPU_ID) if torch.cuda.device_count() > 0 else 'cpu'

    def step(self, actions):
        if TIME_DEBUG: s = log_time()
        if self.is_vector_env:
            dict_actions = [{'action': actions[b]} for b in range(self.B)]
            outputs = self.envs.step(dict_actions)
        else:
            outputs = [self.envs.step(actions)]
        obs_list, reward_list, done_list, info_list = [list(x) for x in zip(*outputs)]
        obs_batch = batch_obs(obs_list, device=self.torch_device)

        if self.is_vector_env:
            return obs_batch, reward_list, done_list, info_list
        else:
            return obs_batch, reward_list[0], done_list[0], info_list[0]

    def reset(self):
        obs_list = self.envs.reset()
        if not self.is_vector_env: obs_list = [obs_list]
        obs_batch = batch_obs(obs_list, device=self.torch_device)
        return obs_batch

    def call(self, aa, bb):
        return self.envs.call(aa, bb)
    def log_info(self, log_type='str', info=None):
        return self.envs.log_info(log_type, info)
    @property
    def habitat_env(self): return self.envs.habitat_env
    @property
    def noise(self): return self.envs.noise
    @property
    def current_episode(self):
        if self.is_vector_env:
            return self.envs.current_episodes
        else:
            return self.envs.current_episode
    @property
    def current_episodes(self):
        return self.envs.current_episodes
