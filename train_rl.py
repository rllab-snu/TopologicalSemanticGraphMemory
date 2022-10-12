#!/usr/bin/env python3

# Copyright (without_goal+curr_emb) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import argparse
import random

import numpy as np
from configs.default import get_config
from trainer.rl import ppo
from habitat_baselines.common.baseline_registry import baseline_registry
import env_utils
import env_utils.env_wrapper
import os

os.environ['GLOG_minloglevel'] = "2"
os.environ['MAGNUM_LOG'] = "quiet"
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    default="./configs/TSGM.yaml",
    type=str,
    # required=True,
    help="path to config yaml containing info about experiment",
)
parser.add_argument(
    "--version",
    type=str,
    required=True,
    help="version of the training experiment",
)
parser.add_argument(
    "--gpu",
    type=str,
    default="0",
    help="gpus",
)
parser.add_argument(
    "--no-noise",
    action='store_true',
    default=False,
    help="include noise or not",
)
parser.add_argument(
    "--diff",
    default='hard',
    choices=['easy', 'medium', 'hard', 'random'],
    help="episode difficulty",
)
parser.add_argument(
    "--seed",
    type=str,
    default="none"
)
parser.add_argument(
    "--render",
    action='store_true',
    default=False,
    help="This will save the episode videos, periodically",
)
parser.add_argument('--task', default='imggoalnav', type=str)

parser.add_argument('--dataset', default='gibson', type=str)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--num-object', default=5, type=int)
parser.add_argument('--multi-target', action='store_true', default=False)
parser.add_argument('--strict-stop', action='store_true', default=True)
parser.add_argument('--policy', default='TSGMPolicy', required=True, type=str)
parser.add_argument(
    "--wandb",
    action='store_true'
)
parser.add_argument('--record', choices=['0','1','2','3'], default='0') # 0: no record 1: env.render 2: pose + action numerical traj 3: features
parser.add_argument('--record-dir', type=str, default='data/video_dir')
parser.add_argument(
    "--run-type",
    choices=["train", "eval"],
    default="train",
    help="run type of the experiment ",
)
parser.add_argument('--mode', default='train_rl', type=str)
parser.add_argument('--project-dir', default='.', type=str)
parser.add_argument('--train-gt', action='store_true', default=False)
parser.add_argument('--use-detector', action='store_true', default=False)
parser.add_argument('--detector-th', default=0.01, type=float)
parser.add_argument('--resume', default='none', type=str)
parser.add_argument('--obj-score-th', default=0.2, type=float)
parser.add_argument('--img-node-th', type=str, default='0.75')
parser.add_argument('--obj-node-th', type=str, default='0.8')
parser.add_argument('--global-policy', action='store_true', default=False)
parser.add_argument('--fd', action='store_true', default=False)

# Noise settings
parser.add_argument('--depth_noise', action='store_true', default=False)
parser.add_argument('--actua_noise', action='store_true', default=False)
parser.add_argument('--sensor_noise', action='store_true', default=False)
parser.add_argument('--depth-noise-level', default=4.0, type=float)
parser.add_argument('--actua-noise-level', default=4.0, type=float)
parser.add_argument('--sensor-noise-level', default=4.0, type=float)
parser.add_argument('--num-procs', default=0, type=int)

arguments = parser.parse_args()
arguments.record = int(arguments.record)
arguments.img_node_th = float(arguments.img_node_th)
arguments.obj_node_th = float(arguments.obj_node_th)
arguments.num_procs = int(arguments.num_procs)


def main():
    run_exp(**vars(arguments))


def run_exp(config: str, opts=None, *args, **kwargs) -> None:
    config = get_config(config, base_task_config_path="./configs/{}_{}.yaml".format(arguments.task, arguments.dataset), opts=opts, arguments=kwargs)
    config.defrost()
    config.POLICY = arguments.policy
    config.RUN_TYPE = arguments.run_type
    config.memory.num_objects = arguments.num_object
    config.render_map = arguments.record > 0 or arguments.render

    config.noisy_actuation = not arguments.no_noise
    config.DIFFICULTY = arguments.diff

    if arguments.num_procs > 0:
        config.NUM_PROCESSES = arguments.num_procs
    config.USE_DETECTOR = config.TASK_CONFIG.USE_DETECTOR = arguments.use_detector
    config.detector_th = config.TASK_CONFIG.detector_th = arguments.detector_th
    config.render = arguments.render
    config.DATASET_NAME = arguments.dataset  # .split("_")[0]
    config.TASK_CONFIG.DATASET.DATASET_NAME = arguments.dataset  # .split("_")[0]

    if arguments.debug:
        config.RL.LOG_INTERVAL = 1
        config.RL.PPO.num_mini_batch = 1
        config.RL.PPO.num_steps = 16
        config.NUM_PROCESSES = 2

    config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    if arguments.seed != 'none':
        config.TASK_CONFIG.SEED = int(arguments.seed)
    config.TASK_CONFIG.TASK.MEASUREMENTS = ["GOAL_INDEX"] + config.TASK_CONFIG.TASK.MEASUREMENTS
    config.TASK_CONFIG.TASK.GOAL_INDEX = config.TASK_CONFIG.TASK.SPL.clone()
    config.TASK_CONFIG.TASK.GOAL_INDEX.TYPE = 'GoalIndex'
    if arguments.strict_stop:
        config.TASK_CONFIG.TASK.SUCCESS_DISTANCE = float(np.clip(float(config.TASK_CONFIG.TASK.SUCCESS_DISTANCE) - 0.5, 0.0, 1.0))
        config.RL.SUCCESS_DISTANCE = float(np.clip(float(config.RL.SUCCESS_DISTANCE) - 0.5, 0.0, 1.0))
    print(config.TASK_CONFIG.TASK.SUCCESS_DISTANCE)
    print(config.RL.SUCCESS_DISTANCE)
    config.TRAINER_NAME = config.RL_TRAINER_NAME
    config.features.object_category_num = 80
    config.memory.num_objects = arguments.num_object
    config.ENV_NAME = "ImageGoalGraphEnv"
    config.img_node_th = arguments.img_node_th
    config.TASK_CONFIG.img_node_th = arguments.img_node_th
    config.TASK_CONFIG.obj_node_th = arguments.obj_node_th
    config.TASK_CONFIG.TRAIN_IL = False
    config.TASK_CONFIG.DATASET.DATASET_NAME = arguments.dataset
    config.TASK_CONFIG.PROC_ID = 0
    config.IMG_SHAPE = (64, 252) #config.TASK_CONFIG.IMG_SHAPE
    config.CHECKPOINT_FOLDER = os.path.join(arguments.project_dir, config.CHECKPOINT_FOLDER)
    config.record = arguments.record > 0
    config.OBJECTGRAPH.SPARSE = True
    arguments.gpu = [int(g) for g in arguments.gpu]
    config.TORCH_GPU_ID = arguments.gpu[0]
    config.SIMULATOR_GPU_ID = arguments.gpu[1]
    config.freeze()
    np.random.seed(config.TASK_CONFIG.SEED)
    random.seed(config.TASK_CONFIG.SEED)

    SAVE_DIR = os.path.join(config.CHECKPOINT_FOLDER, arguments.version)
    if not os.path.exists(SAVE_DIR): os.mkdir(SAVE_DIR)

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)
    trainer.train()


if __name__ == "__main__":
    main()
