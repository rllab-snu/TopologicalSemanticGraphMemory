#!/usr/bin/env python3

# Copyright (without_goal+curr_emb) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Habitat environment without Dataset
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union

import habitat_sim
import gym
import numpy as np
from gym.spaces.dict import Dict as SpaceDict
from gym.spaces.discrete import Discrete
import torch
from habitat.config import Config
from habitat.core.dataset import Dataset, Episode, EpisodeIterator
from habitat.core.embodied_task import EmbodiedTask, Metrics
from habitat.core.simulator import Observations, Simulator
from habitat.datasets import make_dataset
from habitat.sims import make_sim
from habitat.tasks import make_task
import os
from habitat_sim.utils.common import quat_to_coeffs
import quaternion as q
import time
import json
MIN_DIST = 1.5
MAX_DIST = 10.0
from env_utils.custom_habitat_map import get_topdown_map
from env_utils import *
from NuriUtils.statics import CATEGORIES, COI_INDEX
from habitat.core.utils import not_none_validator, try_cv2_import
import attr
import gzip
from NuriUtils.ncutils import append_to_dict
from env_utils.detector_wrapper import VisualizationDemo
from detectron2.config import get_cfg
SURFNORM_KERNEL = None
from habitat.tasks.utils import cartesian_to_polar, quaternion_to_rotation
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    NavigationTask,
)
from types import SimpleNamespace
TIME_DEBUG = False
from torchvision.ops import roi_align
from NuriUtils.ncutils import cam_to_world, get_point_cloud_from_z_panoramic, get_camera_matrix
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@attr.s(auto_attribs=True, kw_only=True)
class VisualNavigationGoal:
    r"""Base class for a goal specification hierarchy."""

    position: List[float] = attr.ib(default=None, validator=not_none_validator)
    rotation: List[float] = attr.ib(default=None, validator=not_none_validator)
    radius: Optional[float] = None


@attr.s(auto_attribs=True, kw_only=True)
class ObjectNavigationGoal:
    r"""Base class for a goal specification hierarchy."""

    position: List[float] = attr.ib(default=None, validator=not_none_validator)
    rotation: List[float] = attr.ib(default=None, validator=not_none_validator)
    object_position: List[float] = attr.ib(default=None, validator=not_none_validator)
    object_target_score: float = attr.ib(default=None)
    object_target_bbox: List[float] = attr.ib(default=None)
    object_target_category: int = attr.ib(default=None)
    object_target_id: int = attr.ib(default=None)
    radius: Optional[float] = None


class Env:
    observation_space: SpaceDict
    action_space: SpaceDict
    _config: Config
    _dataset: Optional[Dataset]
    _episodes: List[Type[Episode]]
    _current_episode_index: Optional[int]
    _current_episode: Optional[Type[Episode]]
    _episode_iterator: Optional[Iterator]
    _sim: Simulator
    _task: EmbodiedTask
    _max_episode_seconds: int
    _max_episode_steps: int
    _elapsed_steps: int
    _episode_start_time: Optional[float]
    _episode_over: bool

    def __init__(
            self, config: Config
    ) -> None:
        """Constructor

        :param config: config for the environment. Should contain id for
            simulator and ``task_name`` which are passed into ``make_sim`` and
            ``make_task``.
        :param dataset: reference to dataset for task instance level
            information. Can be defined as :py:`None` in which case
            ``_episodes`` should be populated from outside.
        """

        assert config.is_frozen(), (
            "Freeze the config before creating the "
            "environment, use config.freeze()."
        )
        self._config = config
        self.task_type = config.TASK.TASK_NAME
        self._current_episode_index = None
        self._current_episode = None
        self.dn = self._config.DATASET.DATASET_NAME.split("_")[0]

        self._scenes = config.DATASET.CONTENT_SCENES
        self._swap_building_every = config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES
        try:
            print('[HabitatEnv][Proc {}] Total {} scenes: '.format(config.PROC_ID, len(self._scenes)), self._scenes)
            print('[HabitatEnv][Proc {}] swap building every '.format(config.PROC_ID), self._swap_building_every)
        except:
            print('[HabitatEnv] Total {} scenes: '.format(len(self._scenes)), self._scenes)
            print('[HabitatEnv] swap building every ', self._swap_building_every)
        self._current_scene_episode_idx = 0
        self._current_scene_idx = 0

        self._config.defrost()
        self._config.SIMULATOR.SCENE = os.path.join(config.DATASET.SCENES_DIR,
                                                    '{}/{}.glb'.format(config.DATASET.DATASET_NAME, self._scenes[0]))
        if not os.path.exists(self._config.SIMULATOR.SCENE):
            self._config.SIMULATOR.SCENE = os.path.join(config.DATASET.SCENES_DIR,
                                                        '{}/{}/{}.glb'.format(config.DATASET.DATASET_NAME, self._scenes[0], self._scenes[0]))
        try:
            if "PROC_ID" in self._config.TASK.GOAL_INDEX:
                self._config.TASK.GOAL_INDEX.PROC_ID = self._config.PROC_ID
            else:
                self._config.PROC_ID = 0
                self._config.TASK.GOAL_INDEX.PROC_ID = self._config.PROC_ID
        except:
            self._config.PROC_ID = 0
        self._config.freeze()

        self._sim = make_sim(
            id_sim=self._config.SIMULATOR.TYPE, config=self._config.SIMULATOR
        )
        self._task = make_task(
            self._config.TASK.TYPE,
            config=self._config.TASK,
            sim=self._sim
        )
        self.observation_space = SpaceDict(
            {
                **self._sim.sensor_suite.observation_spaces.spaces,
                **self._task.sensor_suite.observation_spaces.spaces,
            }
        )
        self.action_space = Discrete(len(self._task.actions))
        self._max_episode_seconds = (
            self._config.ENVIRONMENT.MAX_EPISODE_SECONDS
        )
        self._max_episode_steps = self._config.ENVIRONMENT.MAX_EPISODE_STEPS
        self._elapsed_steps = 0
        self._episode_start_time: Optional[float] = None
        self._episode_over = False

        self.MAX_DIST = MAX_DIST
        self.MIN_DIST = MIN_DIST
        self.difficulty = "random"
        self.args = SimpleNamespace(**config['ARGS'])
        self.episode_name = "VGM"
        try:
            self.episode_name = self.args.episode_name
        except:
            pass

        self._num_goals = getattr(self._config.ENVIRONMENT, 'NUM_GOALS', 1)
        self._episode_iterator = {}
        self._episode_datasets = {}
        self._current_scene_iter = 0
        self.num_agents = len(self._config.SIMULATOR.AGENTS)
        self._total_episode_id = -1
        self._eval_dataset_idx = 0
        self.num_of_camera = config.SIMULATOR.PANORAMIC_SENSOR.NUM_CAMERA
        self.img_height = float(config.IMG_SHAPE[0])
        self.cam_width = float(config.IMG_SHAPE[0] * 4 // self.num_of_camera)
        self.img_width = float(config.IMG_SHAPE[0] * 4 // self.num_of_camera * self.num_of_camera)
        angles = [2 * np.pi * idx / self.num_of_camera for idx in range(self.num_of_camera - 1, -1, -1)]
        half = self.num_of_camera // 2
        self.angles = angles[half:] + angles[:half]
        self.img_ranges = np.arange(self.num_of_camera + 1) * (config.IMG_SHAPE[0] * 4 // self.num_of_camera)

        self.split = config.DATASET.SPLIT
        self.mapping = {}
        if self._current_scene_iter == 0:
            self.get_semantic_mapping()

        if self._config.USE_DETECTOR:
            cfg_detector = self.detector_setup_cfg("model/Detector/mask_rcnn_R_50_FPN_3x.yaml", "data/detector/model_final_f10217.pkl")
            self.detector = VisualizationDemo(cfg_detector, self._config)

        """CAMERA TO WORLD MAPPING"""
        self.camera_matrix = get_camera_matrix(self.cam_width, self.img_height, 360/self.num_of_camera, 90)

    @property
    def current_position(self):
        return self.sim.get_agent_state().position

    @property
    def current_rotation(self):
        return self.sim.get_agent_state().rotation

    def detector_setup_cfg(self, yaml_dir, pkl_dir):
        cfg = get_cfg()
        cfg.merge_from_file(os.path.join(project_dir, yaml_dir))
        cfg.merge_from_list(["MODEL.WEIGHTS", os.path.join(project_dir, pkl_dir), "INPUT.FORMAT", "RGB"])
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = self._config.detector_th
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self._config.detector_th
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = self._config.detector_th
        try:
            print('detector gpu id', cfg.MODEL.DEVICE)
            cfg.MODEL.DEVICE = self._config.DETECTOR_GPU_ID
        except:
            pass
        cfg.freeze()
        return cfg

    @property
    def current_episode(self) -> Type[Episode]:
        assert self._current_episode is not None
        return self._current_episode

    @current_episode.setter
    def current_episode(self, episode: Type[Episode]) -> None:
        self._current_episode = episode

    @property
    def episode_iterator(self) -> Iterator:
        return None

    @episode_iterator.setter
    def episode_iterator(self, new_iter: Iterator) -> None:
        self._episode_iterator = new_iter

    @property
    def episodes(self) -> List[Type[Episode]]:
        return self._episodes

    @episodes.setter
    def episodes(self, episodes: List[Type[Episode]]) -> None:
        assert (
                len(episodes) > 0
        ), "Environment doesn't accept empty episodes list."
        self._episodes = episodes

    @property
    def sim(self) -> Simulator:
        return self._sim

    @property
    def episode_start_time(self) -> Optional[float]:
        return self._episode_start_time

    @property
    def episode_over(self) -> bool:
        return self._episode_over

    @property
    def task(self) -> EmbodiedTask:
        return self._task

    @property
    def _elapsed_seconds(self) -> float:
        assert (
            self._episode_start_time
        ), "Elapsed seconds requested before episode was started."
        return time.time() - self._episode_start_time

    def get_metrics(self) -> Metrics:
        return self._task.measurements.get_metrics()

    def _past_limit(self) -> bool:
        if (
                self._max_episode_steps != 0
                and self._max_episode_steps <= self._elapsed_steps
        ):
            return True
        elif (
                self._max_episode_seconds != 0
                and self._max_episode_seconds <= self._elapsed_seconds
        ):
            return True
        return False

    def _reset_stats(self) -> None:
        self._episode_start_time = time.time()
        self._elapsed_steps = 0
        self._episode_over = False
        self.target_obs = {}
        self.tdv_cnt = 0
        self.changed_list = []

    def get_polar_angle(self, ref_rotation=None):
        if ref_rotation is None:
            agent_state = self._sim.get_agent_state()
            # quaternion is in x, y, z, w format
            ref_rotation = agent_state.rotation

        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        z_neg_z_flip = np.pi
        return np.array(phi) + z_neg_z_flip

    def generate_next_episode(self, scene_id):
        found = False
        while True:
            init_start_position = self._sim.sample_navigable_point()
            random_angle = np.random.rand() * 2 * np.pi
            init_start_rotation = q.from_rotation_vector([0, random_angle, 0])
            self.start_position, self.start_rotation = init_start_position, init_start_rotation
            while True:
                random_dist = np.random.rand() * 0.3 + 0.2
                random_angle = np.random.rand() * 2 * np.pi
                new_start_position = [init_start_position[0] + random_dist * np.cos(random_angle),
                                      init_start_position[1],
                                      init_start_position[2] + random_dist * np.sin(random_angle)]
                random_angle = np.random.rand() * 2 * np.pi
                new_start_rotation = q.from_rotation_vector([0, random_angle, 0])
                if not self._sim.is_navigable(new_start_position):
                    continue
                else:
                    self.start_position = new_start_position
                    self.start_rotation = new_start_rotation
                    self._sim.set_agent_state(new_start_position, new_start_rotation)
                    break

            num_try = 0
            goals = []
            checkpoints = [self.start_position]
            while True:
                goal_position = self._sim.sample_navigable_point()
                euler = [0, 2 * np.pi * np.random.rand(), 0]
                goal_rotation = q.from_rotation_vector(euler)
                if abs(goal_position[1] - init_start_position[1]) > 0.5: continue
                geodesic_dists = [self._sim.geodesic_distance(checkpoint, goal_position) for checkpoint in checkpoints]
                valid_dist = np.stack([(geodesic_dist < self.MAX_DIST) and (geodesic_dist > self.MIN_DIST) for geodesic_dist in geodesic_dists]).all()
                # valid_dist = (geodesic_dists[-1] < self.MAX_DIST) and (geodesic_dists[-1] > self.MIN_DIST)
                if self._sim.is_navigable(goal_position) and valid_dist:
                    goal = VisualNavigationGoal(**{'position': list(goal_position),
                                                   'rotation': list(goal_rotation.components)})
                    goals.append(goal)
                    checkpoints.append(goal_position)
                    self.get_target_objects(goal_position, goal_rotation)
                if len(goals) >= self._num_goals or (num_try > 2000 and len(goals) >= 1):
                    found = True
                    break
                num_try += 1
                if num_try > 30 * self._num_goals and len(goals) == 0:
                    found = False
                    break
            if found: break

        episode_info = {'episode_id': self._current_scene_episode_idx,
                        'scene_id': scene_id,
                        'start_position': list(self.start_position),
                        'start_rotation': list(self.start_rotation.components),
                        'goals': goals,
                        'start_room': None,
                        'shortest_paths': None,
                        'info': {
                            'geodesic_distance': geodesic_dists,
                            'difficulty': self.difficulty
                        }}
        episode = NavigationEpisode(**episode_info)
        return episode, found

    def get_target_objects(self, goal_position, goal_rotation):
        goal_state = self._sim.get_observations_at(goal_position, goal_rotation, keep_agent_at_new_pose=True)
        if self._config.USE_DETECTOR:
            object_bbox, object_score, object_category, object_seg = self.detector.run_on_image(goal_state['panoramic_rgb'][:, :, :3])
            object_world_pose = np.empty([0, 3])
            if len(object_bbox) > 0:
                object_world_pose, object_depth = self.get_box_world(object_bbox, goal_state['panoramic_depth'])
        else:
            object_bbox, object_category, object_id, object_pose, object_score, object_depth = self.get_objects(goal_state['panoramic_semantic'],
                                                                                                     # semantic
                                                                                                     goal_state['panoramic_depth'])  # depth
            self.target_obs = append_to_dict(self.target_obs, "target_loc_object_id", object_id)
            object_world_pose = np.empty([0, 3])
            if len(object_bbox) > 0:
                object_world_pose, object_depth = self.get_box_world(object_bbox, goal_state['panoramic_depth'])
        self.target_obs = append_to_dict(self.target_obs, "target_loc_object", object_bbox)
        self.target_obs = append_to_dict(self.target_obs, "target_loc_object_score", object_score)
        self.target_obs = append_to_dict(self.target_obs, "target_loc_object_category", object_category)
        self.target_obs = append_to_dict(self.target_obs, "target_loc_object_pose", object_world_pose)

    def get_box_world(self, bbox, depth):
        if np.max(depth) <= 1:
            panoramic_depth = depth.squeeze(-1) * 10.
        else:
            panoramic_depth = depth.squeeze(-1)
        xyz = get_point_cloud_from_z_panoramic(panoramic_depth, self.img_width, self.num_of_camera, self.angles, self.camera_matrix)
        xyz = xyz + np.array([0, 0.88, 0])
        Tcw = cam_to_world(q.as_rotation_matrix(self.current_rotation), self.current_position)
        xyz = np.concatenate([xyz.reshape(-1, 3).transpose(), np.ones([1, len(xyz.reshape(-1, 3))])], 0)
        xyz = np.matmul(Tcw, xyz)
        xyz = xyz[:3].reshape(3, panoramic_depth.shape[0], panoramic_depth.shape[1]).transpose(1,2,0)
        bbox = np.concatenate([np.zeros([len(bbox), 1]), bbox], -1)
        bbox = torch.tensor(bbox).float()
        bbox_world = roi_align(torch.tensor(xyz[None]).permute(0,3,1,2).float(), bbox, (7, 7), 1.0).reshape(len(bbox), 3, -1).median(-1)[0].numpy()
        box_depth = roi_align(torch.tensor(depth[None]).permute(0,3,1,2).float(), bbox, (7, 7), 1.0).reshape(len(bbox), 1, -1).median(-1)[0].numpy()
        if np.max(box_depth) <= 1:
            box_depth = box_depth * 10.
        return bbox_world, box_depth.reshape(-1)

    def get_objects(self, semantic, depth=[]):
        semantic = semantic.astype(np.int32)
        max_key = np.max(np.array(list(self.mapping.keys())))
        replace_values = []
        for i in np.arange(max_key + 1):
            try:
                replace_values.append(self.mapping[i])
            except:
                replace_values.append(-1)
        semantic_obs_class = np.take(replace_values, semantic)
        COI_MASK = [(semantic_obs_class == ci).astype(np.int32) for ci in COI_INDEX[self.dn]]  # class mask
        unique_instances = np.unique(semantic * np.sum(np.stack(COI_MASK), 0))[1:]
        semantic_obs_class = semantic_obs_class * np.sum(np.stack(COI_MASK), 0)
        semantic_obs_class[semantic_obs_class == 0] = -1
        bbox_ids = unique_instances
        instance_segment = [(semantic == i).astype(np.int32) for i in unique_instances]
        box_categories = [np.unique(semantic_obs_class[semantic == i])[0] for i in unique_instances]
        if len(instance_segment) > 0:
            object_size = np.stack([np.sum(instance_segman) for instance_segman in instance_segment])
            # if self._config.VERSION == "vis":
            #     mask = (object_size > 200)
            # elif self._config.VERSION == "collect":
            #     mask = (object_size > 200)
            # else:
            mask = (object_size > self.img_height * self.img_width / 200)
            instance_segment = [instance_segman for i, instance_segman in enumerate(instance_segment) if mask[i] == 1]
            box_categories = np.stack(box_categories)[mask == 1]
            bbox_ids = np.array(bbox_ids)[mask == 1]

        x1s = [np.min(np.where(instance_segment[i])[1]) for i in range(len(instance_segment))]
        y1s = [np.min(np.where(instance_segment[i])[0]) for i in range(len(instance_segment))]
        x2s = [np.max(np.where(instance_segment[i])[1]) for i in range(len(instance_segment))]
        y2s = [np.max(np.where(instance_segment[i])[0]) for i in range(len(instance_segment))]
        bboxes = np.stack((x1s, y1s, x2s, y2s), 1)
        if len(bboxes) > 0:
            edge_box_idx = np.where(bboxes[:, 2] - bboxes[:, 0] > self.img_width * 0.8)[0]
            not_edge_box_idx = np.where(bboxes[:, 2] - bboxes[:, 0] <= self.img_width * 0.8)[0]
            # if len(edge_box_idx) > 0:
            #     bboxes = bboxes[not_edge_box_idx]
            #     box_categories = box_categories[not_edge_box_idx]
            #     bbox_ids = bbox_ids[not_edge_box_idx]
            if len(edge_box_idx) > 0:
                x1s1 = [np.min(np.where(instance_segment[i][:, :int(instance_segment[i].shape[1] / 2)])[1]) for i in edge_box_idx]
                y1s1 = [np.min(np.where(instance_segment[i][:, :int(instance_segment[i].shape[1] / 2)])[0]) for i in edge_box_idx]
                x2s1 = [np.max(np.where(instance_segment[i][:, :int(instance_segment[i].shape[1] / 2)])[1]) for i in edge_box_idx]
                y2s1 = [np.max(np.where(instance_segment[i][:, :int(instance_segment[i].shape[1] / 2)])[0]) for i in edge_box_idx]
                bboxes_1 = np.stack((x1s1, y1s1, x2s1, y2s1), 1)
                bboxes_1_categories = box_categories[edge_box_idx]
                bboxes_1_ids = bbox_ids[edge_box_idx]
                x1s2 = [int(instance_segment[i].shape[1] / 2) + np.min(np.where(instance_segment[i][:, int(instance_segment[i].shape[1] / 2):])[1]) for i in \
                        edge_box_idx]
                y1s2 = [np.min(np.where(instance_segment[i][:, int(instance_segment[i].shape[1] / 2):])[0]) for i in edge_box_idx]
                x2s2 = [int(instance_segment[i].shape[1] / 2) + np.max(np.where(instance_segment[i][:, int(instance_segment[i].shape[1] / 2):])[1]) for i in \
                        edge_box_idx]
                y2s2 = [np.max(np.where(instance_segment[i][:, int(instance_segment[i].shape[1] / 2):])[0]) for i in edge_box_idx]
                bboxes_2 = np.stack((x1s2, y1s2, x2s2, y2s2), 1)
                bboxes_2_categories = box_categories[edge_box_idx]
                bboxes_2_ids = bbox_ids[edge_box_idx]
                bboxes_ = bboxes[not_edge_box_idx]
                box_categories_ = box_categories[not_edge_box_idx]
                bbox_ids_ = bbox_ids[not_edge_box_idx]
                bboxes = np.concatenate((bboxes_, bboxes_1, bboxes_2), 0)
                box_categories = np.concatenate((box_categories_, bboxes_1_categories, bboxes_2_categories), 0)
                bbox_ids = np.concatenate((bbox_ids_, bboxes_1_ids, bboxes_2_ids), 0)

        if len(depth) > 0 and len(bboxes) > 0:
            if len(depth.shape) > 2:
                depth = depth.squeeze(-1)
                box_world = np.stack([self.object_loc[bbox_id] for bbox_id in bbox_ids])
                box_pix_xs, box_pix_ys = ((bboxes[:, 0])).astype(np.int32), ((bboxes[:, 1])).astype(np.int32)
                # T_world_camera = self.get_cameraT(box_pix_xs)
                box_depth = np.stack([depth[box_pix_y, box_pix_x] for box_pix_x, box_pix_y in zip(box_pix_xs, box_pix_ys)])
                # rx, ry = (box_pix_xs - self.cam_width / 2.) / (self.cam_width / 2.), (self.img_height / 2. - box_pix_ys) / (self.img_height / 2.)
                # xys = np.array([rx * box_depth, ry * box_depth, -box_depth, np.ones_like(box_depth)])
                # box_world_min = np.stack([np.matmul(T_world_camera[i], xys[:, i]) for i in range(xys.shape[1])])[:, 1]

                # box_pix_xs, box_pix_ys = ((bboxes[:, 2])).astype(np.int32), ((bboxes[:, 3])).astype(np.int32)
                # T_world_camera = self.get_cameraT(box_pix_xs)
                # box_depth = np.stack([depth[box_pix_y, box_pix_x] for box_pix_x, box_pix_y in zip(box_pix_xs, box_pix_ys)])
                # rx, ry = (box_pix_xs - self.cam_width / 2.) / (self.cam_width / 2.), (self.img_height / 2. - box_pix_ys) / (self.img_height / 2.)
                # xys = np.array([rx * box_depth, ry * box_depth, -box_depth, np.ones_like(box_depth)])
                # box_world_max = np.stack([np.matmul(T_world_camera[i], xys[:, i]) for i in range(xys.shape[1])])[:, 1]
            else:
                box_depth = np.ones([len(bboxes)]) * 100. #np.array([100.])
                box_world = np.ones([len(bboxes), 3]) * (-100) #np.array([-100., 3])
                # box_world_min = np.ones(len(bboxes)) * 100
                # box_world_max = np.ones(len(bboxes)) * 100
        else:
            bboxes = np.array([[0, 0, self.img_width-1, self.img_height-1]]).astype(np.float32)
            box_categories = np.array([0])
            bbox_ids = np.array([0])
            box_depth = np.array([0.])
            box_world = np.array([-100., -100., -100.])
            # box_world_min = np.array([100.])
            # box_world_max = np.array([100.])
        return bboxes, box_categories, bbox_ids, box_world, 1. / (box_depth + 0.001), box_depth

    def get_cameraT(self, xs):
        T_world_cameras = []
        sensors = self._sim.get_agent(0).get_state().sensor_states
        xs = xs % self.img_ranges[-1]
        for x in xs:
            idx = np.where(self.img_ranges - x <= 0)[0][-1]
            cam = sensors['rgb_{}'.format(idx)]
            quat, tran = cam.rotation, cam.position
            rota = q.as_rotation_matrix(quat)
            T_world_camera = np.eye(4)
            T_world_camera[0:3, 0:3] = rota
            T_world_camera[0:3, 3] = tran
            T_world_cameras.append(T_world_camera)
        return np.stack(T_world_cameras)

    def get_next_episode(self, scene_id):
        scene_name = scene_id.split('/')[-1][:-4]
        episode = []
        if len(self._episode_datasets[scene_name]) > self._current_scene_episode_idx:
            episode = self._episode_datasets[scene_name][self._current_scene_episode_idx]
        goals = []
        if self._config['ARGS']['episode_name'].split("_")[0] == "VGM":
            # goal = NavigationGoal(**{'position': episode['goal_position']})
            goal = VisualNavigationGoal(**{'position': episode['goal_position'],
                                           'rotation': list(q.from_rotation_vector([0, episode['start_poses']['0'][1], 0]).components)})
            self.start_position = episode['start_poses']['0'][0]
            self.start_rotation = q.from_rotation_vector([0, episode['start_poses']['0'][1], 0])
        elif self._config['ARGS']['episode_name'].split("_")[0] == "NRNS":
            if self._sim.is_navigable(episode['goals'][0]['position']):
                goal_position = episode['goals'][0]['position']
            else:
                goal_position = self._sim.pathfinder.snap_point(episode['goals'][0]['position'])
            # goal = NavigationGoal(**{'position': goal_position})
            goal = VisualNavigationGoal(**{'position': goal_position,
                                           'rotation': [episode['start_rotation'][3], episode['start_rotation'][0],
                                                      episode['start_rotation'][1], episode['start_rotation'][2]]})
            if self._sim.is_navigable(episode['start_position']):
                self.start_position = episode['start_position']
            else:
                self.start_position = self._sim.pathfinder.snap_point(episode['start_position'])
            print("episode distance: ", self._sim.geodesic_distance(self.start_position, goal_position))
            self.start_rotation = q.from_float_array([episode['start_rotation'][3], episode['start_rotation'][0],
                                                      episode['start_rotation'][1], episode['start_rotation'][2]])
        elif self._config['ARGS']['episode_name'] == "MARL":
            goal_position = episode['goals'][0]['position']
            # goal = NavigationGoal(**{'position': goal_position})
            goal = VisualNavigationGoal(**{'position': goal_position,
                                           'rotation': [episode['start_rotation'][3], episode['start_rotation'][0],
                                                      episode['start_rotation'][1], episode['start_rotation'][2]]})
            self.start_position = episode['start_position']
            print("episode distance: ", self._sim.geodesic_distance(self.start_position, goal_position))
            self.start_rotation = q.from_float_array([episode['start_rotation'][3], episode['start_rotation'][0],
                                                      episode['start_rotation'][1], episode['start_rotation'][2]])
        else:
            # goal = NavigationGoal(**{'position': goal_position})
            if len(episode) > 0:
                goal_position = episode['goals'][0]['position']
                goal = VisualNavigationGoal(**{'position': goal_position,
                                               'rotation': episode['goals'][0]['rotation']})
                self.start_position = episode['start_position']
                print("episode distance: ", self._sim.geodesic_distance(self.start_position, goal_position))
                self.start_rotation = q.from_float_array(episode['start_rotation'])
            else:
                done = False
                while not done:
                    episode, done = self.generate_next_episode(scene_name)
                json_file = os.path.join(project_dir, 'data/episodes/{}/{}/{}.json.gz'.format(self.episode_name,
                                                                                              self._config.DATASET.DATASET_NAME.split("_")[0],
                                                                                              scene_name))
                if os.path.exists(json_file):
                    with gzip.open(json_file, "r") as fin:
                        episodes = json.loads(fin.read().decode("utf-8"))
                else:
                    episodes = []
                goal_position = episode.goals[0].position
                goal_rotation = episode.goals[0].rotation
                goal = VisualNavigationGoal(**{'position': goal_position,
                                               'rotation': goal_rotation})
                self.start_position = episode.start_position
                print("episode distance: ", self._sim.geodesic_distance(self.start_position, goal_position))
                self.start_rotation = q.from_float_array(episode.start_rotation)
                epi_info = episode.__dict__
                goal_position = epi_info['goals'][0].position
                goal_rotation = epi_info['goals'][0].rotation
                print("episode_id", episode.episode_id)
                epi_info['goals'] = []
                epi_info['goals'].append({
                    'position': goal_position,
                    'rotation': goal_rotation,
                    'radius': None
                })
                episodes.append(epi_info)
                with gzip.open(json_file, 'wt', encoding="utf-8") as zipfile:
                    json.dump(episodes, zipfile)
        self.get_target_objects(goal.position, q.from_float_array(goal.rotation))
        goals.append(goal)
        episode_info = {'episode_id': self._current_scene_episode_idx,
                        'scene_id': scene_id,
                        'start_position': self.start_position,
                        'start_rotation': self.start_rotation.components,
                        'goals': goals,
                        'start_room': None,
                        'shortest_paths': None}
        episode = NavigationEpisode(**episode_info)
        return episode, True

    def reset(self) -> Observations:
        """Resets the environments and returns the initial observations.
        :return: initial observations from the environment.
        """
        self._reset_stats()
        scene_name = self._scenes[self._current_scene_idx]
        if self._current_scene_iter >= self._swap_building_every:
            self._episode_iterator[scene_name] = self._current_scene_episode_idx + 1
            self._current_scene_idx = (self._current_scene_idx + 1) % len(self._scenes)
            scene_name = self._scenes[self._current_scene_idx]
            if scene_name not in self._episode_iterator.keys():
                self._episode_iterator.update({scene_name: 0})
            self._current_scene_episode_idx = self._episode_iterator[scene_name]
            self._current_scene_iter = 0
            self._config.defrost()
            self._config.SIMULATOR.SCENE = os.path.join(self._config.DATASET.SCENES_DIR,
                                                        '{}/{}.glb'.format(self._config.DATASET.DATASET_NAME, scene_name))
            if not os.path.exists(self._config.SIMULATOR.SCENE):
                self._config.SIMULATOR.SCENE = os.path.join(self._config.DATASET.SCENES_DIR,
                                                            '{}/{}/{}.glb'.format(self._config.DATASET.DATASET_NAME, scene_name, scene_name))
            self._config.freeze()
            self.reconfigure(self._config)

            print('[Proc %d] swapping building %s, every episode will be sampled in : %f, %f' % (self._config.PROC_ID, scene_name, self.MIN_DIST, self.MAX_DIST))

            if self._config['ARGS']['mode'] == "eval":
                if self._config['ARGS']['episode_name'].split("_")[0] == "VGM":
                    json_file = os.path.join(project_dir, 'data/episodes/{}/{}/{}_{}.json'.format(self.episode_name,
                                                                                                  self._config.DATASET.DATASET_NAME.split("_")[0],
                                                                                                  scene_name,
                                                                                                  self.difficulty))
                    with open(json_file, 'r') as f:
                        episodes = json.load(f)
                    self._swap_building_every = len(episodes)
                elif self._config['ARGS']['episode_name'].split("_")[0] == "NRNS":
                    json_file = os.path.join(project_dir, 'data/episodes/{}/{}/{}/test_{}.json.gz'.format(self.episode_name.split("_")[0],
                                                                                                          self._config.DATASET.DATASET_NAME.split("_")[0],
                                                                                                          self.episode_name.split("_")[1], self.difficulty))

                    with gzip.open(json_file, "r") as fin:
                        episodes = json.loads(fin.read().decode("utf-8"))['episodes']
                    episodes = [episode for episode in episodes if scene_name in episode['scene_id']]
                    self._swap_building_every = len(episodes)
                elif self._config['ARGS']['episode_name'] == "MARL":
                    json_file = os.path.join(project_dir, 'data/episodes/{}/{}/{}.json.gz'.format(self.episode_name,
                                                                                                  self._config.DATASET.DATASET_NAME.split("_")[0],
                                                                                                  scene_name))
                    with gzip.open(json_file, "r") as fin:
                        episodes = json.loads(fin.read().decode("utf-8"))
                    episodes = [episode for episode in episodes if episode['info']['difficulty'] == self.difficulty]
                    self._swap_building_every = len(episodes)
                else:
                    json_file = os.path.join(project_dir, 'data/episodes/{}/{}/{}.json.gz'.format(self.episode_name,
                                                                                                  self._config.DATASET.DATASET_NAME.split("_")[0],
                                                                                                  scene_name))
                    if os.path.exists(json_file):
                        with gzip.open(json_file, "r") as fin:
                            episodes = json.loads(fin.read().decode("utf-8"))
                        episodes = [episode for episode in episodes if episode['info']['difficulty'] == self.difficulty]
                    else:
                        os.makedirs(os.path.join(project_dir, 'data/episodes/{}'.format(self.episode_name)), exist_ok=True)
                        os.makedirs(os.path.join(project_dir, 'data/episodes/{}/{}'.format(self.episode_name, self._config.DATASET.DATASET_NAME.split("_")[0])), exist_ok=True)
                        episodes = []
                    self._swap_building_every = int(np.ceil(self.args.num_episodes / len(self._scenes)))
                self._episode_datasets.update({scene_name: episodes})

        self.scene_name = scene_name

        if self._current_scene_iter == 0 and ('mp3d' in self._config.DATASET.DATA_PATH or 'tiny' in self._config.DATASET.DATASET_NAME):
            self.get_semantic_mapping()

        if TIME_DEBUG: s = time.time()

        if self.args.mode == "eval":
            self._current_episode, found_episode = self.get_next_episode(self._config.SIMULATOR.SCENE)
        else:
            self._current_episode, found_episode = self.generate_next_episode(self._config.SIMULATOR.SCENE)

        self._config.defrost()
        agent_dict = {'START_POSITION': self._current_episode.start_position,
                      'START_ROTATION': quat_to_coeffs(q.from_float_array(self._current_episode.start_rotation)).tolist(),
                      'IS_SET_START_STATE': True}
        self._config.SIMULATOR['AGENT_0'].update(agent_dict)
        self._config.freeze()
        self.reconfigure(self._config)
        self._current_scene_episode_idx += 1
        self._current_scene_iter += 1
        self._total_episode_id += 1
        current_episode = self.current_episode
        current_episode.curr_goal_idx = 0
        observations = self.task.reset(episode=current_episode)
        observations['target_idx'] = 0
        observations.update(self.target_obs)
        obs = observations.copy()
        for k, v in obs.items():
            if 'rgb_' in k or 'depth_' in k or 'semantic_' in k:
                observations.pop(k)

        self._task.measurements.reset_measures(
            episode=current_episode, task=self.task, obs=observations, position=self._sim.get_agent(0).get_state().position, dataset=self.dn
        )
        self.target_obs['target_goal'] = observations['target_goal'][..., :4]

        return observations

    def get_semantic_mapping(self):
        scene_objects = self._sim.semantic_scene.objects
        if len(scene_objects) == 0:
            print(self._config.SIMULATOR.SCENE.split("/")[-1])
        self.mapping = {int(obj.id.split("_")[-1]): int(np.where([obj.category.name() == cat for cat in CATEGORIES[self.dn]])[0][0]) for obj in scene_objects if obj != None}
        self.object_category = {int(obj.id.split("_")[-1]): obj.category.name() for obj in scene_objects if obj != None}
        self.object_loc = {int(obj.id.split("_")[-1]): obj.aabb.center for obj in scene_objects if obj != None}

    def _update_step_stats(self, obs=None) -> None:
        self._elapsed_steps += 1
        self._episode_over = not self._task.is_episode_active
        if self._past_limit():
            self._episode_over = True

        if self.episode_iterator is not None and isinstance(
                self.episode_iterator, EpisodeIterator
        ):
            self.episode_iterator.step_taken()

    def step(
            self, action: Union[int, str, Dict[str, Any]], **kwargs
    ) -> Observations:
        r"""Perform an action in the environment and return observations.

        :param action: action (belonging to `action_space`) to be performed
            inside the environment. Action is a name or index of allowed
            task's action and action arguments (belonging to action's
            `action_space`) to support parametrized and continuous actions.
        :return: observations after taking action in environment.
        """

        assert (
                self._episode_start_time is not None
        ), "Cannot call step before calling reset"
        assert (
                self._episode_over is False
        ), "Episode over, call reset before calling step"

        # Support simpler interface as well
        if isinstance(action, str) or isinstance(action, (int, np.integer)):
            action = {"action": action}

        observations = self.task.step(
            action=action, episode=self.current_episode
        )
        observations.update(self.target_obs)
        current_episode = self.current_episode
        current_episode.curr_goal_idx = self.curr_goal_idx
        observations['target_idx'] = self.curr_goal_idx
        # observations['target_goal'] = observations['target_goal'][..., :4]
        self.target_obs['target_goal'] = observations['target_goal'][..., :4]
        self._task.measurements.update_measures(
            episode=current_episode, action=action, task=self.task, obs=observations, position=self._sim.get_agent(0).get_state().position, dataset=self.dn
        )
        self._update_step_stats(observations)
        obs = observations.copy()
        for k, v in obs.items():
            if 'rgb_' in k or 'depth_' in k or 'semantic_' in k:
                observations.pop(k)
        return observations


    def seed(self, seed: int) -> None:
        self._sim.seed(seed)
        self._task.seed(seed)

    def reconfigure(self, config: Config) -> None:
        self._config = config
        self._sim.reconfigure(self._config.SIMULATOR)

    def render(self, mode="rgb") -> np.ndarray:
        return self._sim.render(mode)

    def close(self) -> None:
        self._sim.close()

    @property
    def curr_goal_idx(self):
        if 'GOAL_INDEX' in self._config.TASK.MEASUREMENTS:
            return self.get_metrics()['goal_index']['curr_goal_index']
        else:
            return 0


class RLEnv(gym.Env):
    r"""Reinforcement Learning (RL) environment class which subclasses ``gym.Env``.

    This is a wrapper over `Env` for RL users. To create custom RL
    environments users should subclass `RLEnv` and define the following
    methods: `get_reward_range()`, `get_reward()`, `get_done()`, `get_info()`.

    As this is a subclass of ``gym.Env``, it implements `reset()` and
    `step()`.
    """

    _env: Env

    def __init__(
            self, config: Config, dataset: Optional[Dataset] = None
    ) -> None:
        """Constructor

        :param config: config to construct `Env`
        :param dataset: dataset to construct `Env`.
        """

        self._env = Env(config)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.reward_range = self.get_reward_range()

    @property
    def habitat_env(self) -> Env:
        return self._env

    @property
    def episodes(self) -> List[Type[Episode]]:
        return self._env.episodes

    @property
    def current_episode(self) -> Type[Episode]:
        return self._env.current_episode

    @episodes.setter
    def episodes(self, episodes: List[Type[Episode]]) -> None:
        self._env.episodes = episodes

    def reset(self) -> Observations:
        return self._env.reset()

    def get_reward_range(self):
        r"""Get min, max range of reward.

        :return: :py:`[min, max]` range of reward.
        """
        raise NotImplementedError

    def get_reward(self, observations: Observations) -> Any:
        r"""Returns reward after action has been performed.

        :param observations: observations from simulator and task.
        :return: reward after performing the last action.

        This method is called inside the `step()` method.
        """
        raise NotImplementedError

    def get_done(self, observations: Observations) -> bool:
        r"""Returns boolean indicating whether episode is done after performing
        the last action.

        :param observations: observations from simulator and task.
        :return: done boolean after performing the last action.

        This method is called inside the step method.
        """
        raise NotImplementedError

    def get_info(self, observations) -> Dict[Any, Any]:
        r"""..

        :param observations: observations from simulator and task.
        :return: info after performing the last action.
        """
        raise NotImplementedError

    # def get_objects(self, semantic, depth=[]):
    #     objects_dict = self._env.get_objects(semantic, depth)
    #     return objects_dict

    def step(self, *args, **kwargs) -> Tuple[Observations, Any, bool, dict]:
        r"""Perform an action in the environment.

        :return: :py:`(observations, reward, done, info)`
        """

        observations = self._env.step(*args, **kwargs)
        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = self.get_info(observations)

        return observations, reward, done, info

    def seed(self, seed: Optional[int] = None) -> None:
        self._env.seed(seed)

    def render(self, mode: str = "rgb") -> np.ndarray:
        return self._env.render(mode)

    def close(self) -> None:
        self._env.close()
