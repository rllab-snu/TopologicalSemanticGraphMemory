#!/usr/bin/env python3

# Copyright (without_goal+curr_emb) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union, Dict

import numpy as np

from habitat import get_config as get_task_config
from habitat.config import Config as CN
import os

DEFAULT_CONFIG_DIR = "configs/"
CONFIG_FILE_SEPARATOR = ","
# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
_C.VERSION = 'base'
_C.MAP_NAME = "TopDownGraphMap"
_C.BASE_TASK_CONFIG_PATH = "configs/imggoalnav_gibson.yaml"
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.CMD_TRAILING_OPTS = []  # store command line options as list of strings
_C.IL_TRAINER_NAME = "il"
_C.RL_TRAINER_NAME = "ppo"
_C.ENV_NAME = "NavRLEnv"
_C.SIMULATOR_GPU_ID = 0
_C.TORCH_GPU_ID = 0
_C.MANUAL = False
_C.IMG_SHAPE = (64, 252)
_C.NUM_CAMERA = 12
_C.VIDEO_OPTION = ["disk", "tensorboard"]

_C.TENSORBOARD_DIR = "data/logs/"
_C.VIDEO_DIR = "data/video_dir"
_C.EVAL_CKPT_PATH_DIR = "data/eval_checkpoints"  # path to ckpt or path to ckpts dir
_C.CHECKPOINT_FOLDER = "data/checkpoints"
_C.EPISODE_SOURCE = 'sample'

_C.NUM_PROCESSES = 2
_C.NUM_VAL_PROCESSES = 0

_C.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]

_C.NUM_UPDATES = -1
_C.TOTAL_NUM_STEPS = 1e7
_C.LOG_INTERVAL = 100
_C.LOG_FILE = "train.log"
_C.CHECKPOINT_INTERVAL = 50
_C.VIS_INTERVAL = 1000

_C.POLICY = 'PointNavResNetPolicy'
_C.img_encoder_type = 'unsupervised'
_C.WRAPPER = 'EnvWrapper'
_C.IL_WRAPPER = 'ILWrapper'
_C.DIFFICULTY = 'easy'
_C.NUM_GOALS = 1
_C.NUM_AGENTS = 1
_C.scene_data = 'gibson'
_C.OBS_TO_SAVE = ['panoramic_rgb', 'panoramic_depth', 'target_goal']
_C.noisy_actuation = True
_C.USE_AUXILIARY_INFO = True

#----------------------------------------------------------------------------
# Base architecture config
_C.features = CN()
_C.features.visual_feature_dim = 512
_C.features.object_feature_dim = 32
_C.features.object_category_num = 0
_C.features.action_feature_dim = 32
_C.features.time_dim = 8
_C.features.hidden_size = 512
_C.features.rnn_type = 'LSTM'
_C.features.num_recurrent_layers = 2
_C.features.backbone = 'resnet18'
_C.features.message_feature_dim = 32
#----------------------------------------------------------------------------
# Transformer

_C.transformer = CN()
_C.transformer.hidden_dim = 512
_C.transformer.dropout = 0.1
_C.transformer.nheads = 4
_C.transformer.dim_feedforward = 1024
_C.transformer.enc_layers = 2
_C.transformer.dec_layers = 1
_C.transformer.pre_norm = False
_C.transformer.num_queries = 1

# for memory module
_C.memory = CN()
_C.memory.num_objects = 10
# _C.memory.num_target_objects = 10
_C.memory.img_embedding_dim = 512 #512
_C.memory.memory_size = 100
_C.memory.pose_dim = 5
_C.memory.need_local_memory = False

_C.saving = CN()
_C.saving.name = 'test'
_C.saving.log_interval = 100
_C.saving.save_interval = 5000
_C.saving.eval_interval = 5000
_C.record = False
_C.render = False

_C.RL = CN()

_C.RL.SUCCESS_MEASURE = "SUCCESS"
_C.RL.SUCCESS_DISTANCE = 1.0
_C.RL.REWARD_METHOD = 'progress'

_C.RL.SLACK_REWARD = -0.001
_C.RL.SUCCESS_REWARD = 10.0
_C.RL.COLLISION_REWARD = -0.001

_C.RL.PPO = CN()

_C.RL.PPO.clip_param=0.2
_C.RL.PPO.ppo_epoch=2
_C.RL.PPO.num_mini_batch=3
_C.RL.PPO.value_loss_coef=0.5
_C.RL.PPO.entropy_coef=0.01
_C.RL.PPO.lr=0.00001
_C.RL.PPO.eps=0.00001
_C.RL.PPO.max_grad_norm=0.2
_C.RL.PPO.num_steps = 64
_C.RL.PPO.use_gae=True
_C.RL.PPO.gamma=0.99
_C.RL.PPO.tau=0.95
_C.RL.PPO.use_linear_clip_decay=True
_C.RL.PPO.use_linear_lr_decay=True
_C.RL.PPO.reward_window_size=50
_C.RL.PPO.use_normalized_advantage=True
_C.RL.PPO.hidden_size = 512
_C.RL.PPO.pretrained_weights=""
_C.RL.PPO.rl_pretrained=False
_C.RL.PPO.il_pretrained=False
_C.RL.PPO.pretrained_encoder=False
_C.RL.PPO.train_encoder=True
_C.RL.PPO.reset_critic=False
_C.RL.PPO.backbone='resnet18'
_C.RL.PPO.rnn_type='LSTM'
_C.RL.PPO.num_recurrent_layers=2

_C.IL = CN()
_C.IL.lr = 0.0001
_C.IL.eps = 0.00001
_C.IL.max_grad_norm = 0.5
_C.IL.use_linear_clip_decay=True
_C.IL.use_linear_lr_decay=True
_C.IL.backbone= 'resnet18'
_C.IL.rnn_type= 'LSTM'
_C.IL.num_recurrent_layers=2
_C.IL.batch_size = 4
_C.IL.max_epoch = 100
_C.IL.lr_decay = 0.5
_C.IL.num_workers = 4


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    base_task_config_path: str = "",
    opts: Optional[list] = None,
    arguments: Dict=None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    `config_paths` and overwritten by options from `opts`.
    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, `opts = ['FOO.BAR',
        0.5]`. Argument can be used for parameter sweeping or quick tests.
    """
    config = _C.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if base_task_config_path != "":
        config.BASE_TASK_CONFIG_PATH = base_task_config_path
    config.TASK_CONFIG = get_task_config(config.BASE_TASK_CONFIG_PATH)
    config.defrost()
    config.TASK_CONFIG.TRAIN_IL = False
    if opts:
       config.CMD_TRAILING_OPTS = opts
       config.merge_from_list(opts)

    if arguments:
        config['ARGS'] = arguments
        config.TASK_CONFIG['ARGS'] = arguments
        if 'version' in arguments:
            config.VERSION = arguments['version']
            config.TASK_CONFIG.VERSION = arguments['version']

    if not os.path.exists('data'): os.mkdir('data')
    # if not os.path.exists(config.TENSORBOARD_DIR): os.mkdir(config.TENSORBOARD_DIR)
    if not os.path.exists(config.VIDEO_DIR): os.mkdir(config.VIDEO_DIR)
    if not os.path.exists(config.EVAL_CKPT_PATH_DIR): os.mkdir(config.EVAL_CKPT_PATH_DIR)
    if not os.path.exists(config.CHECKPOINT_FOLDER): os.mkdir(config.CHECKPOINT_FOLDER)

    # config.TENSORBOARD_DIR = os.path.join(config.TENSORBOARD_DIR, config.VERSION)
    # if not os.path.exists(config.TENSORBOARD_DIR): os.mkdir(config.TENSORBOARD_DIR)

    config.freeze()
    return config
