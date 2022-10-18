#!/usr/bin/env python3

# Copyright (without_goal+curr_emb) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.tasks.nav.object_nav_task import ObjectGoalNavEpisode, ObjectGoal

import os
from typing import Any, List, Optional

import attr
import numpy as np
from gym import spaces

from habitat.config import Config
from habitat.core.dataset import Dataset
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, Sensor, SensorTypes, RGBSensor, DepthSensor, SemanticSensor
from habitat.core.dataset import Dataset, Episode
from habitat.core.utils import not_none_validator
from typing import Dict, List, Optional, Tuple
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    NavigationTask,
    DistanceToGoal,
    Success,
    TopDownMap,
)

#
# from habitat.tasks.nav.nav import (
#     merge_sim_episode_config,
#     EpisodicGPSSensor
# )

import quaternion as q
import torch
import habitat_sim
import magnum as mn

@registry.register_sensor(name="EquirectRGBSensor")
class EquirectRGBSensor(RGBSensor):
    def __init__(self, config, **kwargs: Any):
        self.config = config
        self.sim_sensor_type = habitat_sim.SensorType.COLOR
        self.sim_sensor_subtype = habitat_sim.SensorSubType.EQUIRECTANGULAR
        self.sensor_subtype = habitat_sim.SensorSubType.EQUIRECTANGULAR
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "equirect_rgb_sensor"

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.HEIGHT, self.config.WIDTH, 3),
            dtype=np.uint8,
        )

    def get_observation(self, obs, *args: Any, **kwargs: Any) -> Any:
        obs = obs.get(self.uuid, None)
        return obs

@registry.register_sensor(name="EquirectSemanticSensor")
class EquirectSemanticSensor(SemanticSensor):
    def __init__(self, config, **kwargs: Any):
        self.config = config
        self.sim_sensor_type = habitat_sim.SensorType.SEMANTIC
        self.sim_sensor_subtype = habitat_sim.SensorSubType.EQUIRECTANGULAR
        self.sensor_subtype = habitat_sim.SensorSubType.EQUIRECTANGULAR
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "Equirect_semantic_sensor"

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=np.Inf,
            shape=(self.config.HEIGHT, self.config.WIDTH, 1),
            dtype=np.uint8,
        )

    def get_observation(self, obs, *args: Any, **kwargs: Any) -> Any:
        obs = obs.get(self.uuid, None)
        return obs

@registry.register_sensor(name="EquirectDepthSensor")
class EquirectDepthSensor(DepthSensor):
    def __init__(self, config, **kwargs: Any):
        self.sim_sensor_type = habitat_sim.SensorType.DEPTH
        self.sim_sensor_subtype = habitat_sim.SensorSubType.EQUIRECTANGULAR
        self.sensor_subtype = habitat_sim.SensorSubType.EQUIRECTANGULAR
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "equirect_depth_sensor"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.DEPTH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=np.inf,
            shape=(self.config.HEIGHT, self.config.WIDTH, 1),
            dtype=np.float32)

    def get_observation(self, obs,*args: Any, **kwargs: Any):
        obs = obs.get(self.uuid, None)
        if isinstance(obs, np.ndarray):
            obs = np.expand_dims(obs, axis=2)
        else:
            obs = obs.unsqueeze(-1)

        return obs

@registry.register_sensor(name="OrthoRGBSensor")
class OrthoRGBSensor(RGBSensor):
    def __init__(self, config, **kwargs: Any):
        self.config = config
        self.sim_sensor_type = habitat_sim.SensorType.COLOR
        self.sim_sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
        self.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "ortho_rgb_sensor"

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.HEIGHT, self.config.WIDTH, 3),
            dtype=np.uint8,
        )

    def get_observation(self, obs, *args: Any, **kwargs: Any) -> Any:
        obs = obs.get(self.uuid, None)
        return obs

@registry.register_sensor(name="OrthoSemanticSensor")
class OrthoSemanticSensor(SemanticSensor):
    def __init__(self, config, **kwargs: Any):
        self.config = config
        self.sim_sensor_type = habitat_sim.SensorType.SEMANTIC
        self.sim_sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
        self.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "ortho_semantic_sensor"

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=np.Inf,
            shape=(self.config.HEIGHT, self.config.WIDTH, 1),
            dtype=np.uint8,
        )

    def get_observation(self, obs, *args: Any, **kwargs: Any) -> Any:
        obs = obs.get(self.uuid, None)
        return obs

@registry.register_sensor(name="OrthoDepthSensor")
class OrthoDepthSensor(DepthSensor):
    def __init__(self, config, **kwargs: Any):
        self.sim_sensor_type = habitat_sim.SensorType.DEPTH
        self.sim_sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
        # self.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "ortho_depth_sensor"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.DEPTH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=np.inf,
            shape=(self.config.HEIGHT, self.config.WIDTH, 1),
            dtype=np.float32)

    def get_observation(self, obs,*args: Any, **kwargs: Any):
        obs = obs.get(self.uuid, None)
        if isinstance(obs, np.ndarray):
            obs = np.expand_dims(obs, axis=2)
        else:
            obs = obs.unsqueeze(-1)

        return obs

@registry.register_sensor(name="PanoramicPartRGBSensor")
class PanoramicPartRGBSensor(RGBSensor):
    def __init__(self, config, **kwargs: Any):
        self.config = config
        self.angle = config.ANGLE
        self.sim_sensor_type = habitat_sim.SensorType.COLOR
        self.sim_sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        # self.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "rgb_" + self.angle

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.HEIGHT, self.config.WIDTH, 3),
            dtype=np.uint8,
        )

    def get_observation(self, obs, *args: Any, **kwargs: Any) -> Any:
        obs = obs.get(self.uuid, None)
        return obs

@registry.register_sensor(name="PanoramicPartSemanticSensor")
class PanoramicPartSemanticSensor(RGBSensor):
    def __init__(self, config, **kwargs: Any):
        self.config = config
        self.angle = config.ANGLE
        self.sim_sensor_type = habitat_sim.SensorType.SEMANTIC
        self.sim_sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        # self.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "semantic_" + self.angle

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=np.Inf,
            shape=(self.config.HEIGHT, self.config.WIDTH, 1),
            dtype=np.uint8,
        )

    def get_observation(self, obs, *args: Any, **kwargs: Any) -> Any:
        obs = obs.get(self.uuid, None)
        return obs

@registry.register_sensor(name="PanoramicPartDepthSensor")
class PanoramicPartDepthSensor(DepthSensor):
    def __init__(self, config, **kwargs: Any):
        self.sim_sensor_type = habitat_sim.SensorType.DEPTH
        self.sim_sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        self.angle = config.ANGLE
        if config.NORMALIZE_DEPTH:
            self.min_depth_value = 0
            self.max_depth_value = 1
        else:
            self.min_depth_value = config.MIN_DEPTH
            self.max_depth_value = config.MAX_DEPTH
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=self.min_depth_value,
            high=self.max_depth_value,
            shape=(self.config.HEIGHT, self.config.WIDTH, 1),
            dtype=np.float32)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "depth_" + self.angle

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.DEPTH

    def get_observation(self, obs,*args: Any, **kwargs: Any):
        obs = obs.get(self.uuid, None)
        if isinstance(obs, np.ndarray):
            obs = np.clip(obs, self.config.MIN_DEPTH, self.config.MAX_DEPTH)
            obs = np.expand_dims(
                obs, axis=2
            )
        else:
            obs = obs.clamp(self.config.MIN_DEPTH, self.config.MAX_DEPTH)

            obs = obs.unsqueeze(-1)

        if self.config.NORMALIZE_DEPTH:
            obs = (obs - self.config.MIN_DEPTH) / (
                self.config.MAX_DEPTH - self.config.MIN_DEPTH
            )
        return obs

@registry.register_sensor(name="PanoramicRGBSensor")
class PanoramicRGBSensor(Sensor):
    def __init__(self, config, **kwargs: Any):
        self.config = config
        self.torch = False #config.HABITAT_SIM_V0.GPU_GPU
        self.num_camera = config.NUM_CAMERA
        self.sim_sensor_type = habitat_sim.SensorType.COLOR
        self.sim_sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        super().__init__(config=config)

        # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "panoramic_rgb"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.HEIGHT, self.config.WIDTH, 3),
            dtype=np.uint8,
        )

    # This is called whenver reset is called or an action is taken
    def get_observation(self, observations, *args: Any, **kwargs: Any):
        if isinstance(observations['rgb_0'][:,:,:3], torch.Tensor):
            rgb_list = [observations['rgb_%d' % (i)][:, :, :3] for i in range(self.num_camera)]
            rgb_array = torch.cat(rgb_list, 1)
        else:
            rgb_list = [observations['rgb_%d' % (i)][:, :, :3] for i in range(self.num_camera)]
            rgb_array = np.concatenate(rgb_list, 1)
        if rgb_array.shape[1] > self.config.HEIGHT*4:
            left = rgb_array.shape[1] - self.config.HEIGHT*4
            slice = left//2
            rgb_array = rgb_array[:,slice:slice+self.config.HEIGHT*4]
        return rgb_array

@registry.register_sensor(name="PanoramicDepthSensor")
class PanoramicDepthSensor(DepthSensor):
    def __init__(self, config, **kwargs: Any):
        self.sim_sensor_type = habitat_sim.SensorType.DEPTH
        self.sim_sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        if config.NORMALIZE_DEPTH: self.depth_range = [0,1]
        else: self.depth_range = [config.MIN_DEPTH, config.MAX_DEPTH]
        self.min_depth_value = config.MIN_DEPTH
        self.max_depth_value = config.MAX_DEPTH
        self.num_camera = config.NUM_CAMERA
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=self.depth_range[0],
            high=self.depth_range[1],
            shape=(self.config.HEIGHT, self.config.WIDTH, 1),
            dtype=np.float32)

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "panoramic_depth"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.DEPTH

    def get_observation(self, observations,*args: Any, **kwargs: Any):
        depth_list = [observations['depth_%d' % (i)] for i in range(self.num_camera)]

        if isinstance(depth_list[0], np.ndarray):
            obs = np.concatenate(depth_list, 1)
            obs = np.clip(obs, self.config.MIN_DEPTH, self.config.MAX_DEPTH)
            obs = np.expand_dims(
                obs, axis=2
            )
        else:
            obs = torch.cat(depth_list, 1)
            obs = obs.clamp(self.config.MIN_DEPTH, self.config.MAX_DEPTH)

            obs = obs.unsqueeze(-1)

        if self.config.NORMALIZE_DEPTH:
            obs = (obs - self.config.MIN_DEPTH) / (
                self.config.MAX_DEPTH - self.config.MIN_DEPTH
            )

        if obs.shape[1] > self.config.HEIGHT*4:
            left = obs.shape[1] - self.config.HEIGHT*4
            slice = left//2
            obs = obs[:,slice:slice+self.config.HEIGHT*4]

        return obs

@registry.register_sensor(name="PanoramicSemanticSensor")
class PanoramicSemanticSensor(SemanticSensor):
    def __init__(self, config, **kwargs: Any):
        self.sim_sensor_type = habitat_sim.SensorType.SEMANTIC
        self.sim_sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        self.torch = False#sim.config.HABITAT_SIM_V0.GPU_GPU
        self.num_camera = config.NUM_CAMERA

        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=np.Inf,
            shape=(self.config.HEIGHT, self.config.WIDTH, 1),
            dtype=np.float32)

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "panoramic_semantic"
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC
    # This is called whenver reset is called or an action is taken
    def get_observation(self, observations,*args: Any, **kwargs: Any):
        depth_list = [observations['semantic_%d'%(i)] for i in range(self.num_camera)]
        if isinstance(depth_list[0], torch.Tensor):
            return torch.cat(depth_list, 1)
        else:
            return np.concatenate(depth_list,1)

@registry.register_sensor(name="CustomImgGoalSensor")
class CustomImgGoalSensor(Sensor):
    def __init__(
        self, sim, config: Config, dataset: Dataset, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._dataset = dataset
        use_depth = True#'DEPTH_SENSOR_0' in self._sim.config.sim_cfg.AGENT_0.SENSORS
        use_rgb = True#'RGB_SENSOR_0' in self._sim.config.AGENT_0.SENSORS
        self.channel = use_depth + 3 * use_rgb
        self.height = config.HEIGHT
        self.width = config.WIDTH
        self.curr_episode_id = -1
        self.curr_scene_id = ''
        self.num_camera = config.NUM_CAMERA
        self.sim_sensor_type = habitat_sim.SensorType.COLOR
        self.sim_sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "target_goal"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0, high=1.0, shape=(self.height, self.width, self.channel), dtype=np.float32)

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: NavigationEpisode,
        **kwargs: Any,
    ) -> Optional[int]:
        episode_id = episode.episode_id
        scene_id = episode.scene_id
        if (self.curr_episode_id != episode_id) or (self.curr_scene_id != scene_id):
            self.curr_episode_id = episode_id
            self.curr_scene_id = scene_id
            self.goal_obs = []
            self.goal_pose = []
            for goal in episode.goals:
                position = mn.Vector3(goal.position)
                try:
                    rotation = q.from_float_array(goal.rotation)
                except:
                    euler = [0, 2 * np.pi * np.random.rand(), 0]
                    rotation = q.from_rotation_vector(euler)
                obs = self._sim.get_observations_at(position,rotation)
                rgb_list = [obs['rgb_%d' % (i)][:, :, :3] for i in range(self.num_camera)]
                if isinstance(obs['rgb_0'], torch.Tensor):
                    rgb_array = torch.cat(rgb_list, 1) / 255.
                else:
                    rgb_array = np.concatenate(rgb_list, 1)/255.

                if rgb_array.shape[1] > self.height*4:
                    left = rgb_array.shape[1] - self.height*4
                    slice = left // 2
                    rgb_array = rgb_array[:, slice:slice + self.height*4]
                depth_list = [obs['depth_%d' % (i)] for i in range(self.num_camera)]
                if isinstance(obs['depth_0'], torch.Tensor):
                    depth_array = torch.cat(depth_list, 1)
                else:
                    depth_array = np.concatenate(depth_list, 1)
                if depth_array.shape[1] > self.height*4:
                    left = depth_array.shape[1] - self.height*4
                    slice = left // 2
                    depth_array = depth_array[:, slice:slice + self.height*4]
                if isinstance(obs['depth_0'], torch.Tensor):
                    goal_obs = torch.cat([rgb_array, depth_array],2)
                else:
                    goal_obs = np.concatenate([rgb_array, depth_array],2)
                self.goal_obs.append(goal_obs)
                self.goal_pose.append([position, rotation.components])
            if len(episode.goals) >= 1:
                if isinstance(obs['rgb_0'], torch.Tensor):
                    self.goal_obs = torch.stack(self.goal_obs,0)
                else:
                    self.goal_obs = np.array(self.goal_obs)
        return self.goal_obs


from habitat.core.embodied_task import (
    EmbodiedTask,
    Measure,
    SimulatorTaskAction,
)
from habitat.core.simulator import (
    AgentState,
    Sensor,
    SensorTypes,
    ShortestPathPoint,
    Simulator,
)
from habitat.tasks.nav.nav import Success, DistanceToGoal


@registry.register_measure(name='Success_woSTOP')
class Success_woSTOP(Success):
    r"""Whether or not the agent succeeded at its task

    This measure depends on DistanceToGoal measure.
    """
    cls_uuid: str = "success"

    def update_metric(
        self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any
    ):
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()

        if (
           distance_to_target < self._config.SUCCESS_DISTANCE
        ):
            self._metric = 1.0
        else:
            self._metric = 0.0

@registry.register_measure(name='GoalIndex')
class GoalIndex(Measure):
    cls_uuid: str = "goal_index"
    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self.num_goals = len(episode.goals)
        self.goal_index = 0
        self.all_done = False
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        self._metric = {'curr_goal_index': self.goal_index,
                        'num_goals': self.num_goals}

    def increase_goal_index(self):
        self.goal_index += 1
        self._metric = {'curr_goal_index': min(self.goal_index, self.num_goals-1),
                        'num_goals': self.num_goals}
        self.all_done = self.goal_index >= self.num_goals
        # print("goal_idx", self.goal_index)
        # print("proc_id", self._config.PROC_ID, "goal_idx", self.goal_index)
        return self.all_done



@registry.register_measure(name='Custom_DistanceToGoal')
class DistanceToGoal(Measure):
    """The measure calculates a distance towards the goal.
    """

    cls_uuid: str = "distance_to_goal"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._previous_position = None
        self._sim = sim
        self._config = config
        # self._episode_view_points = None
        self._episode_view_points: Optional[
            List[Tuple[float, float, float]]
        ] = None

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._previous_position = None
        self._metric = None
        try:
            if self._config.DISTANCE_TO == "VIEW_POINTS":
                self._episode_view_points = [
                    view_point.agent_state.position
                    for goal in episode.goals
                    for view_point in goal.view_points
                ]
        except:
            self._episode_view_points = []
        self.goal_idx = -1
        self.update_metric(episode=episode, *args, **kwargs)

    def update_metric(self, episode: Episode, task: EmbodiedTask, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position

        if GoalIndex.cls_uuid in task.measurements.measures:
            self.goal_idx = task.measurements.measures[GoalIndex.cls_uuid].get_metric()['curr_goal_index']
        else:
            self.goal_idx = 0

        if self._previous_position is None or not np.allclose(
            self._previous_position, current_position, atol=1e-4
        ):
            if self._config.DISTANCE_TO == "POINT":
                distance_to_target = self._sim.geodesic_distance(
                    current_position,
                    episode.goals[self.goal_idx].position,
                    episode,
                )
            elif self._config.DISTANCE_TO == "VIEW_POINTS":
                distance_to_target = self._sim.geodesic_distance(
                    current_position, self._episode_view_points, episode
                )
            else:
                logger.error(
                    f"Non valid DISTANCE_TO parameter was provided: {self._config.DISTANCE_TO}"
                )

            self._previous_position = current_position
            self._metric = distance_to_target

@registry.register_measure(name='Custom_SPL')
class SPL(Measure):
    r"""SPL (Success weighted by Path Length)

    ref: On Evaluation of Embodied Agents - Anderson et. al
    https://arxiv.org/pdf/1807.06757.pdf
    The measure depends on Distance to Goal measure and Success measure
    to improve computational
    performance for sophisticated goal areas.
    """

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._episode_view_points = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "spl"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        #rint(self._start_end_episode_distance)
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):

        current_position = self._sim.get_agent_state().position
        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric =(
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )


@registry.register_measure(name='Custom_SoftSPL')
class SoftSPL(SPL):
    r"""Soft SPL

    Similar to SPL with a relaxed soft-success criteria. Instead of a boolean
    success is now calculated as 1 - (ratio of distance covered to target).
    """

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "softspl"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(self, episode, task, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()

        ep_soft_success = max(
            0, (1 - distance_to_target / self._start_end_episode_distance)
        )

        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = ep_soft_success * (
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )


@registry.register_measure
class DistanceToGoalReward(Measure):
    r"""Binary success reward, sans shaping"""
    cls_uuid: str = "d2g_reward"

    def __init__(self, sim, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._previous_distance_to_target = 0
        self._metric = 0
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.cls_uuid,
            [
                Success.cls_uuid,
                DistanceToGoal.cls_uuid,
            ],
        )
        self._metric = 0
        self._previous_distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(
        self,
        episode,
        task: EmbodiedTask,
        *args: Any,
        **kwargs: Any,
    ):
        reward = 0
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self._previous_distance_to_target = distance_to_target
        reward += self._previous_distance_to_target - distance_to_target
        if task.measurements.measures[Success.cls_uuid].get_metric():
            reward += self._config.SUCCESS_REWARD
        self._metric = reward
