from typing import Optional, Type
from habitat import Config, Dataset
import cv2
from utils.vis_utils import observations_to_image, append_text_to_image
from gym.spaces.dict import Dict as SpaceDict
from gym.spaces.box import Box
import gzip
from habitat.core.spaces import ActionSpace, EmptySpace
import numpy as np
from env_utils.custom_habitat_env import RLEnv
import habitat
from env_utils.custom_habitat_map import TopDownGraphMap
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import imageio
import os
import quaternion as q
from habitat.tasks.utils import cartesian_to_polar
import torch
import json
from types import SimpleNamespace
from utils.statics import CATEGORIES, COI_INDEX
from utils.ncutils import cam_to_world, get_point_cloud_from_z_panoramic, get_camera_matrix
from torchvision.ops import roi_align
from env_utils.noisy_actions import CustomActionSpaceConfiguration
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ImageGoalEnv(RLEnv):
    metadata = {'render.modes': ['rgb_array']}
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self.noise = config.noisy_actuation
        self.record = config.record
        self.render_map = getattr(config, 'render_map', False)
        self.visualize_every = config.VIS_INTERVAL
        self.record_dir = config.VIDEO_DIR
        self.args = SimpleNamespace(**config['ARGS'])

        task_config = config.TASK_CONFIG
        task_config.defrost()
        if self.render_map:
            task_config.TASK.TOP_DOWN_MAP.DRAW_SOURCE = True
            task_config.TASK.TOP_DOWN_MAP.DRAW_SHORTEST_PATH = True
            task_config.TASK.TOP_DOWN_MAP.FOG_OF_WAR.VISIBILITY_DIST = 3.0
            task_config.TASK.TOP_DOWN_MAP.FOG_OF_WAR.FOV = 360
            task_config.TASK.TOP_DOWN_MAP.FOG_OF_WAR.DRAW = True
            task_config.TASK.TOP_DOWN_MAP.DRAW_VIEW_POINTS = False
            task_config.TASK.TOP_DOWN_MAP.DRAW_GOAL_POSITIONS = True
            task_config.TASK.TOP_DOWN_MAP.DRAW_GOAL_AABBS = False
            task_config.TASK.TOP_DOWN_GRAPH_MAP = config.TASK_CONFIG.TASK.TOP_DOWN_MAP.clone()
            task_config.TASK.TOP_DOWN_GRAPH_MAP.TYPE = config.MAP_NAME
            task_config.TASK.TOP_DOWN_GRAPH_MAP.MAP_RESOLUTION = 1250
            task_config.TASK.TOP_DOWN_GRAPH_MAP.NUM_TOPDOWN_MAP_SAMPLE_POINTS = 10000
            task_config.TASK.TOP_DOWN_GRAPH_MAP.USE_DETECTOR = config.USE_DETECTOR
            task_config.TASK.TOP_DOWN_GRAPH_MAP.PROJECT_DIR = project_dir
            task_config.TASK.TOP_DOWN_GRAPH_MAP.DATASET_NAME = config.TASK_CONFIG.DATASET.DATASET_NAME.split("_")[0]
            if getattr(config, 'map_more', False):
                task_config.TASK.TOP_DOWN_GRAPH_MAP.MAP_RESOLUTION = 2500
                task_config.TASK.TOP_DOWN_GRAPH_MAP.NUM_TOPDOWN_MAP_SAMPLE_POINTS = 10000
            task_config.TASK.TOP_DOWN_GRAPH_MAP.DRAW_CURR_LOCATION = getattr(config, 'GRAPH_LOCATION', 'point')
            task_config.TASK.MEASUREMENTS += ['TOP_DOWN_GRAPH_MAP']

        if 'TOP_DOWN_MAP' in config.TASK_CONFIG.TASK.MEASUREMENTS:
            task_config.TASK.MEASUREMENTS = [k for k in task_config.TASK.MEASUREMENTS if 'TOP_DOWN_MAP' != k]
        task_config.SIMULATOR.ACTION_SPACE_CONFIG = "CustomActionSpaceConfiguration"
        task_config.TASK.POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS + ['NOISY_FORWARD', 'NOISY_LEFT', 'NOISY_RIGHT']
        task_config.TASK.ACTIONS.NOISY_FORWARD = habitat.config.Config()
        task_config.TASK.ACTIONS.NOISY_FORWARD.TYPE = "NOISYFORWARD"
        task_config.TASK.ACTIONS.NOISY_LEFT = habitat.config.Config()
        task_config.TASK.ACTIONS.NOISY_LEFT.TYPE = "NOISYLEFT"
        task_config.TASK.ACTIONS.NOISY_RIGHT = habitat.config.Config()
        task_config.TASK.ACTIONS.NOISY_RIGHT.TYPE = "NOISYRIGHT"
        task_config.freeze()
        self.config = config
        self.dn = task_config.DATASET.DATASET_NAME.split("_")[0]
        self._core_env_config = config.TASK_CONFIG
        self.success_distance = float(config.RL.SUCCESS_DISTANCE)
        self._previous_measure = None
        self._previous_action = -1
        self.timestep = 0
        self.stuck = 0
        self.follower = None
        if 'NOISY_FORWARD' not in HabitatSimActions:
            HabitatSimActions.extend_action_space("NOISY_FORWARD")
            HabitatSimActions.extend_action_space("NOISY_LEFT")
            HabitatSimActions.extend_action_space("NOISY_RIGHT")

        if self.noise: moves = ["NOISY_FORWARD", "NOISY_LEFT", "NOISY_RIGHT"]
        else: moves = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
        if 'STOP' in task_config.TASK.POSSIBLE_ACTIONS:
            self.action_dict = {id+1: move for id, move in enumerate(moves)}
            self.action_dict[0] = "STOP"
        else:
            self.action_dict = {id: move for id, move in enumerate(moves)}

        self.SUCCESS_REWARD = self.config.RL.SUCCESS_REWARD
        self.COLLISION_REWARD = self.config.RL.COLLISION_REWARD
        self.SLACK_REWARD = self.config.RL.SLACK_REWARD
        self.number_of_episodes = 1000

        if self.args.mode != "train_rl" and self.args.mode != "eval" and self.args.mode != "collect":
            return

        print('[ImageGoalEnv] NOISY ACTUATION : ', self.noise)
        super().__init__(self._core_env_config, dataset)
        self.num_agents = self.habitat_env.num_agents

        act_dict = {
            "MOVE_FORWARD": EmptySpace(),
            'TURN_LEFT': EmptySpace(),
            'TURN_RIGHT': EmptySpace()
        }
        if 'STOP' in task_config.TASK.POSSIBLE_ACTIONS:
            act_dict.update({'STOP': EmptySpace()})
        self.action_space = ActionSpace(act_dict)
        obs_dict = {
            'panoramic_rgb': self.habitat_env._sim.sensor_suite.observation_spaces.spaces['panoramic_rgb'],
            'panoramic_depth': self.habitat_env._sim.sensor_suite.observation_spaces.spaces['panoramic_depth'],
            'target_goal': self.habitat_env._task.sensor_suite.observation_spaces.spaces['target_goal'],
            'step': Box(low=np.array(0),high=np.array(500), dtype=np.float32),
            'prev_act': Box(low=np.array(-1), high=np.array(self.action_space.n), dtype=np.int32),
            'gt_action': Box(low=np.array(-1), high=np.array(self.action_space.n), dtype=np.int32),
            'target_pose': Box(low=-np.Inf, high=np.Inf, shape=(3,), dtype=np.float32),
            'distance': Box(low=-np.Inf, high=np.Inf, shape=(1,), dtype=np.float32),
        }
        obs_dict['object'] = Box(low=-np.Inf, high=np.Inf, shape=(self.config.memory.num_objects, 5), dtype=np.float32)
        obs_dict['object_score'] = Box(low=-np.Inf, high=np.Inf, shape=(self.config.memory.num_objects,), dtype=np.float32)
        obs_dict['object_mask'] = Box(low=0, high=1, shape=(self.config.memory.num_objects,), dtype=np.bool)
        obs_dict['object_relpose'] = Box(low=-np.Inf, high=np.Inf, shape=(self.config.memory.num_objects, 2), dtype=np.float32)
        obs_dict['object_category'] = Box(low=-np.Inf, high=np.Inf, shape=(self.config.memory.num_objects,), dtype=np.float32)
        obs_dict['object_depth'] = Box(low=-np.Inf, high=np.Inf, shape=(self.config.memory.num_objects,), dtype=np.float32)
        obs_dict['target_loc_object'] = Box(low=-np.Inf, high=np.Inf, shape=(self.config.memory.num_objects, 5), dtype=np.float32)
        obs_dict['target_loc_object_score'] = Box(low=-np.Inf, high=np.Inf, shape=(self.config.memory.num_objects,), dtype=np.float32)
        obs_dict['target_loc_object_mask'] = Box(low=0, high=1, shape=(self.config.memory.num_objects,), dtype=np.bool)
        obs_dict['target_loc_object_category'] = Box(low=-np.Inf, high=np.Inf, shape=(self.config.memory.num_objects,), dtype=np.float32)
        self.mapper = self.habitat_env.task.measurements.measures['top_down_map'] if (self.render_map) else None
        if self.mapper and getattr(config, 'map_more', False):
            self.mapper.loose_check = True
            self.mapper.height_th = 0.5
        if self.config.USE_AUXILIARY_INFO:
            obs_dict.update({
                'is_goal': Box(low=0, high=1, shape=(1,), dtype=np.int32),
                'progress': Box(low=0, high=1, shape=(1,), dtype=np.float32),
                'target_dist_score': Box(low=0, high=1, shape=(1,), dtype=np.float32),
            })
        self.observation_space = SpaceDict(obs_dict)

        self.gradual_diff = getattr(config,'gradual_diff',False)
        self.habitat_env.difficulty = config.DIFFICULTY
        if config.DIFFICULTY == 'easy':
            self.habitat_env.MIN_DIST, self.habitat_env.MAX_DIST = 1.5, 3.0
        elif config.DIFFICULTY == 'medium':
            self.habitat_env.MIN_DIST, self.habitat_env.MAX_DIST = 3.0, 5.0
        elif config.DIFFICULTY == 'hard':
            self.habitat_env.MIN_DIST, self.habitat_env.MAX_DIST = 5.0, 10.0
        elif config.DIFFICULTY == 'random':
            self.habitat_env.MIN_DIST, self.habitat_env.MAX_DIST = 1.5, 10.0
        elif config.DIFFICULTY == 'collect':
            self.habitat_env.MIN_DIST, self.habitat_env.MAX_DIST = 3.0, 10.0
        elif config.DIFFICULTY == 'collect_graph':
            self.habitat_env.MIN_DIST, self.habitat_env.MAX_DIST = 3.0, 30.0
        else:
            raise NotImplementedError
        print('[ImageGoalEnv] Current difficulty %s, MIN_DIST %f, MAX_DIST %f - # goals %d'%(config.DIFFICULTY, self.habitat_env.MIN_DIST, self.habitat_env.MAX_DIST, self.habitat_env._num_goals))

        scene_name = self.config.TASK_CONFIG.SIMULATOR.SCENE.split('/')[-1][:-4]
        if self.config.TASK_CONFIG['ARGS']['mode'] == "eval":
            use_generated_one = False
            if self.config.TASK_CONFIG['ARGS']['episode_name'].split("_")[0] == "VGM":
                json_file = os.path.join(project_dir, 'data/episodes/{}/{}/{}_{}.json'.format(self.habitat_env.episode_name, self.config.TASK_CONFIG.DATASET.DATASET_NAME.split("_")[0], scene_name,
                                                                                                          self.habitat_env.difficulty))
                with open(json_file, 'r') as f:
                    episodes = json.load(f)
                self.habitat_env._swap_building_every = len(episodes)
            elif self.config.TASK_CONFIG['ARGS']['episode_name'].split("_")[0] == "NRNS":
                json_file = os.path.join(project_dir, 'data/episodes/{}/{}/{}/test_{}.json.gz'.format(self.habitat_env.episode_name.split("_")[0],
                                                                                                   self.config.TASK_CONFIG.DATASET.DATASET_NAME.split("_")[0],
                                                                                                   self.habitat_env.episode_name.split("_")[1], self.habitat_env.difficulty))

                with gzip.open(json_file, "r") as fin:
                    episodes = json.loads(fin.read().decode("utf-8"))['episodes']
                episodes = [episode for episode in episodes if scene_name in episode['scene_id']]
                self.habitat_env._swap_building_every = len(episodes)
            elif self.config.TASK_CONFIG['ARGS']['episode_name'] == "MARL":
                json_file = os.path.join(project_dir, 'data/episodes/{}/{}/{}.json.gz'.format(self.habitat_env.episode_name,
                                                                                                         self.config.TASK_CONFIG.DATASET.DATASET_NAME.split("_")[0],
                                                                                                          scene_name))
                with gzip.open(json_file, "r") as fin:
                    episodes = json.loads(fin.read().decode("utf-8"))
                episodes = [episode for episode in episodes if episode['info']['difficulty'] == self.habitat_env.difficulty]
                self.habitat_env._swap_building_every = len(episodes)
            else:
                json_file = os.path.join(project_dir, 'data/episodes/{}/{}/{}.json.gz'.format(self.habitat_env.episode_name,
                                                                                                          self.config.TASK_CONFIG.DATASET.DATASET_NAME.split("_")[0],
                                                                                                          scene_name))
                if os.path.exists(json_file):
                    with gzip.open(json_file, "r") as fin:
                        total_episodes = json.loads(fin.read().decode("utf-8"))
                    diff_episodes = [episode for episode in total_episodes if episode['info']['difficulty'] == self.habitat_env.difficulty]
                    episodes = total_episodes
                else:
                    os.makedirs(os.path.join(project_dir, 'data/episodes/{}'.format(self.habitat_env.episode_name)), exist_ok=True)
                    os.makedirs(os.path.join(project_dir, 'data/episodes/{}/{}'.format(self.habitat_env.episode_name, self.config.TASK_CONFIG.DATASET.DATASET_NAME.split("_")[0])), exist_ok=True)
                    episodes = []
                self.habitat_env._swap_building_every = int(np.ceil(self.args.num_episodes / len(self.habitat_env._scenes)))
            self.habitat_env._episode_datasets.update({scene_name: episodes})

        self.reward_method = config.RL.REWARD_METHOD
        if self.reward_method == 'progress':
            self.get_reward = self.get_progress_reward
        elif self.reward_method == 'coverage':
            self.get_reward = self.get_coverage_reward
        elif self.reward_method == 'sparse':
            self.get_reward = self.get_sparse_reward

        self.need_gt_action = False
        self.has_log_info = None
        self.curr_graph_path = None
        self.need_subgoal_obs = getattr(config, 'subgoal_obs', False)

        # self.use_noise_position = getattr(config, 'noise_position', False)
        # if self.use_noise_position:
        #     self.sensor_noise_fwd = pickle.load(open(os.path.join(project_dir, "data/noise_models/sensor_noise_fwd.pkl", 'rb')))
        #     self.sensor_noise_left = pickle.load(open(os.path.join(project_dir, "data/noise_models/sensor_noise_left.pkl", 'rb')))
        #     self.sensor_noise_right = pickle.load(open(os.path.join(project_dir, "data/noise_models/sensor_noise_right.pkl", 'rb')))
        #     self.sensor_noise_level = getattr(self.config, 'sensor_noise_level', 1.0)
        #     print('[ImageGoalEnv] Sensor noise level', self.sensor_noise_level)

        self.build_path_follower()
        self.num_of_camera = task_config.SIMULATOR.PANORAMIC_SENSOR.NUM_CAMERA
        self.img_height = float(task_config.IMG_SHAPE[0])
        self.img_width = float(task_config.IMG_SHAPE[0] * 4 // self.num_of_camera * self.num_of_camera)
        self.cam_width = float(task_config.IMG_SHAPE[0] * 4 // self.num_of_camera)
        self.camera_matrix = get_camera_matrix(self.cam_width, self.img_height, 360/self.num_of_camera, 90)
        angles = [2 * np.pi * idx / self.num_of_camera for idx in range(self.num_of_camera - 1, -1, -1)]
        half = self.num_of_camera // 2
        self.angles = angles[half:] + angles[:half]

        if self.config.USE_DETECTOR:
            self.detector = self.habitat_env.detector

    @property
    def current_position(self):
        return self.habitat_env.sim.get_agent_state().position

    @property
    def current_rotation(self):
        return self.habitat_env.sim.get_agent_state().rotation

    @property
    def curr_distance(self):
        res = self._env.get_metrics()['distance_to_goal']
        if res == None:
            res = self.starting_distance
        return res

    def draw_image_graph_on_map(self, node_list, affinity, graph_mask, curr_info, flags=None):
        if self.mapper is not None and self.render_map:
            self.mapper.draw_image_graph_on_map(node_list, affinity, graph_mask, curr_info, flags)

    def draw_object_graph_on_map(self, node_list, node_category, node_score, vis_node_list, affinity, graph_mask, curr_info, flags=None):
        if self.mapper is not None and self.render_map:
            self.mapper.draw_object_graph_on_map(node_list, node_category, node_score, vis_node_list, affinity, graph_mask, curr_info, flags)

    def get_sensor_states(self):
        return self.habitat_env._sim.get_agent(0).get_state().sensor_states

    def get_mapping(self):
        return self.habitat_env.mapping

    def get_object_loc(self):
        return self.habitat_env.object_loc

    def build_path_follower(self, each_goal=False):
        self.follower = ShortestPathFollower(self.habitat_env._sim, self.success_distance, False)

    def get_best_action(self, goal=None):
        curr_goal = goal if goal is not None else self.curr_goal.position
        act = self.follower.get_next_action(curr_goal)
        if 'STOP' not in self.habitat_env.task.actions:
            act = act - 1
            if act == -1:
                act = 1
        return act

    def get_dist(self, goal_position):
        return self.habitat_env._sim.geodesic_distance(self.current_position, goal_position)

    # def get_noisy_dist(self, goal_position):
    #     return self.habitat_env._sim.geodesic_distance(self.noise_position, goal_position)

    @property
    def recording_now(self):
        return self.record and self.habitat_env._total_episode_id % self.visualize_every == 0

    @property
    def curr_goal_idx(self): return 0

    @property
    def curr_goal(self):
        return self.current_episode.goals[self.curr_goal_idx]

    def draw_semantic_map(self, xyz, category):
        if self.mapper is not None and self.render_map:
            self.mapper.draw_semantic_map(xyz, category)

    def reset(self):
        self._previous_action = -1
        self.timestep = 0
        self.object_positions = []
        self.object_categories = []
        obs = super().reset()

        self.num_goals = len(self._env._current_episode.goals)
        self._previous_measure = self.get_dist(self.curr_goal.position)
        self.initial_pose = self.current_position
        self.start_to_goal = self.habitat_env._sim.geodesic_distance(self.initial_pose, self.curr_goal.position)
        # if self.use_noise_position:
        #     self.noise_position = [self.initial_pose[0], self.initial_pose[2], q.as_euler_angles(self.current_rotation)[1]]

        self.info = {}
        self.total_reward = 0
        self.progress = 0
        self.stuck = 0
        self.min_measure = self.habitat_env.MAX_DIST
        self.prev_coverage = 0
        self.has_log_info = None
        self.prev_position = self.current_position.copy()
        self.prev_rotation = self.current_rotation.copy()
        self.starting_distance = self.start_to_goal
        self.positions = [self.current_position]
        if self.args.mode == "collect":
            obs = self.process_obs_collect(obs)
        else:
            obs = self.process_obs(obs)
        self.obs = obs
        if self.render_map or self.record:
            self.get_xyz(obs['panoramic_depth'])
        if self.recording_now:
            self.imgs = []
            self.imgs.append(self.render('rgb'))
        return obs

    @property
    def scene_name(self): # version compatibility
        if hasattr(self.habitat_env._sim, 'habitat_config'):
            sim_scene = self.habitat_env._sim.habitat_config.SCENE
        else:
            sim_scene = self.habitat_env._sim.config.SCENE
        return sim_scene

    def get_polar_angle(self, ref_rotation=None):
        if ref_rotation is None:
            agent_state = self._env._sim.get_agent_state()
            # quaternion is in x, y, z, w format
            ref_rotation = agent_state.rotation
        vq = np.quaternion(0,0,0,0)
        vq.imag = np.array([0,0,-1])
        heading_vector = (ref_rotation.inverse() * vq * ref_rotation).imag
        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        x_y_flip = -np.pi / 2
        return np.array(phi) + x_y_flip

    def step(self, action):
        self._previous_action = action
        obs, reward, done, self.info = super().step(self.action_dict[action])

        self.timestep += 1
        self.info['length'] = self.timestep * done
        self.info['episode'] = int(self.current_episode.episode_id)
        self.info['distance_to_goal'] = self._previous_measure
        self.info['start_to_goal'] = self.start_to_goal
        self.info['step'] = self.timestep
        if 'top_down_map' in self.habitat_env.get_metrics():
            self.info['coverage'] = self.habitat_env.get_metrics()['top_down_map']['fog_of_war_mask'].mean()
        self.info['ortho_map'] = {
            'agent_loc': self.current_position,
            'agent_rot': self.get_polar_angle(self.current_rotation),
        }
        self.positions.append(self.current_position)
        self.total_reward += reward
        self.prev_position = self.current_position.copy()
        self.prev_rotation = self.current_rotation.copy()

        if self.args.mode == "collect":
            obs = self.process_obs_collect(obs)
        else:
            obs = self.process_obs(obs)
        self.obs = obs
        if self.render_map or self.record:
            self.get_xyz(obs['panoramic_depth'])
        if self.recording_now:
            self.imgs.append(self.render('rgb'))
            if done: self.save_video(self.imgs)
        return obs, reward, done, self.info

    def process_obs(self, obs):
        obs_dict = {}
        if self.args.mode != "eval":
            gt_action = self.get_best_action()
        else:
            gt_action = -1
        obs_dict['gt_action'] = gt_action

        obs_dict['step'] = self.timestep
        obs_dict['position'] = self.current_position
        obs_dict['rotation'] = self.current_rotation.components
        rotation = q.as_euler_angles(self.current_rotation)[1]
        obs_dict['pose'] = [obs_dict['position'][0], obs_dict['position'][2], rotation, obs_dict['step']]
        obs_dict["map_pose"] = self.get_sim_location_with_poserot(obs_dict['position'], q.from_float_array(obs_dict['rotation']))
        obs_dict['target_goal'] = self.habitat_env.target_obs['target_goal'][self.curr_goal_idx]
        obs.update(self.habitat_env.target_obs)

        obs_dict['distance'] = self.curr_distance
        target_dist_score = np.maximum(1 - np.array(obs_dict['distance']) / 2., 0.0)
        obs_dict['target_dist_score'] = np.array(target_dist_score).astype(np.float32).reshape(1)
        progress = np.clip(1 - self.curr_distance / self.starting_distance, 0, 1)
        obs_dict['progress'] = np.array(progress).astype(np.float32).reshape(1)

        # Transform
        obs_dict['panoramic_rgb'] = obs['panoramic_rgb'].copy()
        obs_dict['panoramic_depth'] = obs['panoramic_depth'].copy()
        obs_dict.update({"start_position": self._env._current_episode.start_position,
                         "start_rotation": self._env._current_episode.start_rotation})
        max_num_object = self.config.memory.num_objects
        obs = self.update_objects(obs)
        self.detected_object_category = [CATEGORIES[self.dn][i] for i in obs['object_category']]
        self.detected_object_score = obs['object_score'].copy()
        self.detected_object_distance = obs['object_depth'].copy()
        self.detected_object_position = obs['object_pose'].copy()
        object_out = np.zeros((max_num_object, 5))
        object_category_out = np.zeros((max_num_object))
        object_mask_out = np.zeros((max_num_object))
        object_score_out = np.zeros((max_num_object))
        object_id_out = np.zeros((max_num_object))
        object_pose_out = np.zeros((max_num_object, 3))
        object_map_pose_out = np.zeros((max_num_object, 3))
        num_object_t = obs['object'].shape[0]
        object_out[:min(max_num_object, num_object_t), 1:] = obs['object'][:min(max_num_object, num_object_t), :4]
        object_category_out[:min(max_num_object, num_object_t)] = obs['object_category'][:min(max_num_object, num_object_t)]
        object_mask_out[:min(max_num_object, num_object_t)] = 1
        object_pose_out[:min(max_num_object, num_object_t)] = obs['object_pose'][:min(max_num_object, num_object_t)]
        object_map_pose_out[:min(max_num_object, num_object_t)] = obs['object_map_pose'][:min(max_num_object, num_object_t)]
        object_score_out[:min(max_num_object, num_object_t)] = obs['object_score'][:min(max_num_object, num_object_t)]
        if 'object_id' in obs:
            object_id_out[:min(max_num_object, len(obs['object_id']))] = obs['object_id'][:min(max_num_object, len(obs['object_id']))]

        obs_dict['object'] = object_out.copy()
        if ("train" not in self.args.mode or self.render_map or self.record):
            obs_dict['object_seg'] = obs['object_seg'].copy()
        obs_dict['object_category'] = object_category_out
        obs_dict['object_mask'] = object_mask_out
        obs_dict['object_pose'] = object_pose_out
        obs_dict['object_map_pose'] = object_map_pose_out
        obs_dict['object_score'] = object_score_out
        obs_dict['object_id'] = object_id_out
        obs_dict['target_object_pose'] = self._env._current_episode.goals[self.curr_goal_idx].position
        obs_dict['is_goal'] = np.array([int(self.curr_distance < 1.0)])

        max_num_object = self.config.memory.num_objects
        target_loc_object = np.zeros((max_num_object, 5))
        target_loc_object_category_out = np.zeros((max_num_object))
        target_loc_object_mask_out = np.zeros((max_num_object))
        target_loc_object_score_out = np.zeros((max_num_object))
        target_loc_object_pose_out = np.zeros((max_num_object, 3))
        target_loc_object_id_out = np.zeros((max_num_object))
        if obs.get('target_loc_object', None) != None:
            num_object_t = obs['target_loc_object_score'][obs['target_idx']].reshape(-1).shape[0]
            if num_object_t > 0:
                target_loc_object[:min(max_num_object, num_object_t), 1:] = obs['target_loc_object'][obs['target_idx']].reshape(-1, 4)[:min(max_num_object, num_object_t)]
                target_loc_object_category_out[:min(max_num_object, num_object_t)] = obs['target_loc_object_category'][obs['target_idx']].reshape(-1)[:min(max_num_object, num_object_t)]
                target_loc_object_mask_out[:min(max_num_object, num_object_t)] = 1
                target_loc_object_pose_out[:min(max_num_object, num_object_t)] = np.array(obs['target_loc_object_pose'][obs['target_idx']]).reshape(-1, 3)[:min(max_num_object, num_object_t)]
                target_loc_object_score_out[:min(max_num_object, num_object_t)] = obs['target_loc_object_score'][obs['target_idx']].reshape(-1)[:min(max_num_object, num_object_t)]
                if 'target_loc_object_id' in obs:
                    target_loc_object_id_out[:min(max_num_object, num_object_t)] = obs['target_loc_object_id'][obs['target_idx']].reshape(-1)[:min(max_num_object, num_object_t)]

        obs_dict['target_loc_object'] = target_loc_object
        obs_dict['target_loc_object_category'] = target_loc_object_category_out
        obs_dict['target_loc_object_mask'] = target_loc_object_mask_out
        obs_dict['target_loc_object_pose'] = target_loc_object_pose_out
        obs_dict['target_loc_object_score'] = target_loc_object_score_out
        obs_dict['target_loc_object_id'] = target_loc_object_id_out

        if  obs_dict['target_loc_object'][:,1:].max() > 1:
            obs_dict['target_loc_object'][:,1] = obs_dict['target_loc_object'][:,1] / self.img_width
            obs_dict['target_loc_object'][:,2] = obs_dict['target_loc_object'][:,2] / self.img_height
            obs_dict['target_loc_object'][:,3] = obs_dict['target_loc_object'][:,3] / self.img_width
            obs_dict['target_loc_object'][:,4] = obs_dict['target_loc_object'][:,4] / self.img_height

        return obs_dict

    def process_obs_collect(self, obs):
        obs_dict = {}
        if self.args.mode != "eval":
            gt_action = self.get_best_action()
        else:
            gt_action = -1
        obs_dict['gt_action'] = gt_action

        obs_dict['step'] = self.timestep
        obs_dict['position'] = self.current_position
        obs_dict['rotation'] = self.current_rotation.components
        rotation = q.as_euler_angles(self.current_rotation)[1]
        obs_dict['pose'] = [obs_dict['position'][0], obs_dict['position'][2], rotation, obs_dict['step']]
        obs_dict["map_pose"] = self.get_sim_location_with_poserot(obs_dict['position'], q.from_float_array(obs_dict['rotation']))
        obs.update(self.habitat_env.target_obs)

        obs_dict['distance'] = self.curr_distance
        target_dist_score = np.maximum(1 - np.array(obs_dict['distance']) / 2., 0.0)
        obs_dict['target_dist_score'] = np.array(target_dist_score).astype(np.float32).reshape(1)
        progress = np.clip(1 - self.curr_distance / self.starting_distance, 0, 1)
        obs_dict['progress'] = np.array(progress).astype(np.float32).reshape(1)

        # Transform
        obs_dict['panoramic_rgb'] = obs['panoramic_rgb'].copy()
        obs_dict['panoramic_depth'] = obs['panoramic_depth'].copy()
        obs_dict.update({"start_position": self._env._current_episode.start_position,
                         "start_rotation": self._env._current_episode.start_rotation})
        obs = self.update_objects(obs)
        obs_dict.update(obs)
        # self.detected_object_category = [CATEGORIES[self.dn][i] for i in obs['object_category']]
        # self.detected_object_score = obs['object_score'].copy()
        # self.detected_object_distance = obs['object_depth'].copy()
        # self.detected_object_position = obs['object_pose'].copy()
        if ("train" not in self.args.mode or self.render_map or self.record):
            obs_dict['object_seg'] = obs['object_seg'].copy()
        obs_dict['target_object_pose'] = self._env._current_episode.goals[self.curr_goal_idx].position
        return obs_dict

    def get_sim_location(self):
        """Returns x, y, o pose of the agent in the Habitat simulator."""

        agent_state = super().habitat_env.sim.get_agent_state(0)
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        axis = q.as_euler_angles(agent_state.rotation)[0]
        if (axis % (2 * np.pi)) < 0.1 or (axis %
                                          (2 * np.pi)) > 2 * np.pi - 0.1:
            o = q.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2 * np.pi - q.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o

    def get_sim_location_with_poserot(self, position, rotation):
        """Returns x, y, o pose of the agent in the Habitat simulator."""
        x = -position[2]
        y = -position[0]
        axis = q.as_euler_angles(rotation)[0]
        if (axis % (2 * np.pi)) < 0.1 or (axis %
                                          (2 * np.pi)) > 2 * np.pi - 0.1:
            o = q.as_euler_angles(rotation)[1]
        else:
            o = 2 * np.pi - q.as_euler_angles(rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return np.array([x, y, o]).reshape(-1)

    def save_video(self,imgs):
        video_name = 'ep_%03d_scene_%s.mp4'%(self.habitat_env._total_episode_id, self.scene_name.split('/')[-1][:-4])
        w, h = imgs[0].shape[0:2]
        resize_h, resize_w = (h//16)*16, (w//16)*16
        imgs = [cv2.resize(img, dsize=(resize_h, resize_w)) for img in imgs]
        imageio.mimsave(os.path.join(self.record_dir, video_name), imgs, fps=5)

    def get_reward_range(self):
        return (
            self.SLACK_REWARD - 1.0,
            self.SUCCESS_REWARD + 1.0,
        )

    def get_progress_reward(self, observations):
        reward = self.SLACK_REWARD
        current_measure = self.get_dist(self.curr_goal.position)
        # absolute decrease on measure
        self.move = self._previous_measure - current_measure
        reward += max(self.move,0.0) * 0.2
        if abs(self.move) < 0.01:
            self.stuck += 1
        else:
            self.stuck = 0

        self._previous_measure = current_measure

        if self._episode_success():
            reward += self.SUCCESS_REWARD * self.habitat_env.get_metrics()['spl']
        return reward

    def get_sparse_reward(self, observations):
        reward = self.config.SLACK_REWARD
        success = self._episode_success()
        if success:
            reward += self.config.SUCCESS_REWARD#* self.habitat_env.get_metrics()['spl']
        elif 0 in self._previous_action.values():
            reward += -2
        return reward

    def _episode_success(self):
        return self.habitat_env.get_metrics()['success']

    def get_success(self):
        return self._episode_success()

    def get_done(self, observations):
        done = False
        if self.habitat_env.episode_over or self._episode_success():
            done = True
        pose = self.current_position
        diff_floor = abs(pose[1] - self.initial_pose[1]) > 0.5
        if self.stuck > 50 or diff_floor:
            done = True
        if np.isinf(self.habitat_env.get_metrics()['distance_to_goal']):
            done = True
        return done

    def get_info(self, observations):
        info = self.habitat_env.get_metrics()
        return info


    def get_episode_over(self):
        return self.habitat_env.episode_over

    def get_agent_state(self):
        return self.habitat_env.sim.get_agent_state()

    def get_curr_goal_index(self):
        return self.curr_goal_idx

    def log_info(self, log_type='str', info=None):
        self.has_log_info = {'type': log_type,
                             'info': info}

    def render(self, mode='rgb', **kwargs):
        attns = kwargs['attns'] if 'attns' in kwargs else None
        info = self.get_info(None) if self.info is None else self.info
        img = observations_to_image(self.obs.copy(), info, mode='panoramic', clip = self.config.WRAPPER == "GraphWrapper", use_detector=self.config.USE_DETECTOR,
                                    task_name=self.config.TASK_CONFIG.TASK.TASK_NAME, dataset_name=self.config.TASK_CONFIG.DATASET.DATASET_NAME.split("_")[0],
                                     attns=attns, sim=self._env._sim)
        str_action = 'XX'
        if 'STOP' not in self.habitat_env.task.actions:
            action_list = ["MF", 'TL', 'TR']
        else:
            action_list = ["ST", "MF", 'TL', 'TR']
        if self._previous_action != -1:
            str_action = str(action_list[self._previous_action])

        reward = self.total_reward.sum() if isinstance(self.total_reward, np.ndarray) else self.total_reward
        txt = 't: %03d, r: %.2f ,dist: %.2f, stuck: %02d  a: %s '%(self.timestep,reward, self.get_dist(self.curr_goal.position)
                                                                   ,self.stuck, str_action)
        if hasattr(self, 'navigate_mode'):
            txt = self.navigate_mode + ' ' + txt
        if self.has_log_info is not None:
            if self.has_log_info['type'] == 'str':
                txt += ' ' + self.has_log_info['info']
        if hasattr(self.mapper, 'node_list'):
            if self.mapper.node_list is None:
                txt += ' node : NNNN'
                txt += ' curr : NNNN'
            else:
                num_node = len(self.mapper.node_list)
                txt += ' node : %03d' % (num_node)
                curr_info = self.mapper.curr_info
                if 'curr_node' in curr_info.keys():
                    txt += ' curr: {}'.format(curr_info['curr_node']+1)
                if 'goal_prob' in curr_info.keys():
                    txt += ' goal %.3f'%(curr_info['goal_prob'])

        img = append_text_to_image(img, txt)

        if mode == 'rgb' or mode == 'rgb_array':
            return cv2.resize(img, dsize=(950,450))
        elif mode == 'human':
            cv2.imshow('render', img[:,:,::-1])
            cv2.waitKey(1)
            return img
        return super().render(mode)

    def get_coverage_reward(self, observations):
        top_down_map = self.habitat_env.get_metrics()['top_down_map']
        fow = top_down_map["fog_of_war_mask"]
        self.map_size = (top_down_map['map'] != 0).sum()
        self.curr_coverage = np.sum(fow)
        new_pixel = self.curr_coverage - self.prev_coverage
        reward = np.clip(new_pixel, 0, 50) / 1000  # 0 ~ 0.1
        self.prev_coverage = self.curr_coverage

        reward += self.SLACK_REWARD
        current_measure = self.get_dist(self.curr_goal.position)
        # absolute decrease on measure
        self.move = self._previous_measure - current_measure
        if abs(self.move) < 0.01:
            self.stuck += 1
        else:
            self.stuck = 0

        self._previous_measure = current_measure
        if self._episode_success():
            reward += self.SUCCESS_REWARD

        return reward

    def get_dists(self, pose, other_poses):
        dists = np.linalg.norm(np.array(other_poses).reshape(len(other_poses),3) - np.array(pose).reshape(1,3), axis=1)
        return dists

    def update_objects(self, obs):
        obs['object'] = np.array([[0, 0, obs['panoramic_rgb'].shape[1] - 1, obs['panoramic_rgb'].shape[0] - 1]]).astype(np.float32)
        obs['object_score'] = np.array([0.])
        obs['object_category'] = np.array([0])
        obs['object_pose'] = np.array([[-100., -100., -100.]])
        obs['object_id'] = np.array([0])
        obs['object_depth'] = np.array([100.])
        obs['object_map_pose'] = np.array([[0., 0., 0.]])
        if self.config.USE_DETECTOR or self.args.mode == "eval":
            object_bbox, object_score, object_category, object_seg = self.detector.run_on_image(obs['panoramic_rgb'][:, :, :3])
            if len(object_bbox) > 0:
                object_world_pose, object_depth = self.get_box_world(object_bbox, obs['panoramic_depth'])
            else:
                object_world_pose = np.empty([0, 3])
                object_depth = np.empty([0, ])

            obs['object'] = object_bbox
            obs['object_seg'] = object_seg
            obs['object_category'] = object_category
            obs['object_pose'] = object_world_pose
            obs['object_score'] = object_score
            obs['object_depth'] = object_depth
            if len(object_bbox) > 0:
                obs['object_id'] = np.stack([-1] * len(object_bbox))
            else:
                obs['object_id'] = np.empty(0)

        else:
            if isinstance(obs['panoramic_semantic'], torch.Tensor):
                semantic_obs_instance = obs['panoramic_semantic'].cpu().detach().numpy()
            else:
                semantic_obs_instance = obs['panoramic_semantic']

            if isinstance(obs['panoramic_depth'], torch.Tensor):
                depth_image = obs['panoramic_depth'].cpu().detach().numpy()
            else:
                depth_image = obs['panoramic_depth']

            if np.max(depth_image) <= 1:
                depth_image = depth_image * 10.

            gt_object_bbox, gt_object_category, gt_object_id, gt_object_pose, gt_object_depth, gt_object_score, gt_object_seg = self.get_objects(semantic_obs_instance)
            obs['object'] = gt_object_bbox
            obs['object_seg'] = gt_object_seg
            obs['object_score'] = gt_object_score
            obs['object_category'] = gt_object_category
            obs['object_pose'] = gt_object_pose
            obs['object_id'] = gt_object_id
            obs['object_depth'] = gt_object_depth

        if len(obs['object']) > 0:
            obs['object'] = obs['object'].astype(np.float32)
            obs['object'][:, 0] = obs['object'][:, 0] / obs['panoramic_rgb'].shape[1]
            obs['object'][:, 1] = obs['object'][:, 1] / obs['panoramic_rgb'].shape[0]
            obs['object'][:, 2] = obs['object'][:, 2] / obs['panoramic_rgb'].shape[1]
            obs['object'][:, 3] = obs['object'][:, 3] / obs['panoramic_rgb'].shape[0]
            object_map_pose = []
            for i in range(len(obs['object_pose'])):
                object_map_pose.append(self.get_sim_location_with_poserot(obs['object_pose'][i], q.from_float_array(self._env._current_episode.start_rotation)))
            obs['object_map_pose'] = np.stack(object_map_pose)  # np.array(np.stack([-bb[:, 2], bb[:, 0]], 1), dtype=np.float32)
            obs['object_seg'] = np.ones_like(obs['panoramic_rgb'][...,0])*(-1)
        else:
            obs['object'] = np.array([[0, 0, 1., 1.]]).astype(np.float32)
            obs['object_score'] = np.array([0.])
            obs['object_category'] = np.array([0])
            obs['object_pose'] = np.array([[-100., -100., -100.]])
            obs['object_id'] = np.array([0])
            obs['object_depth'] = np.array([100.])
            obs['object_seg'] = np.ones_like(obs['panoramic_rgb'][...,0])*(-1)
            obs['object_map_pose'] = np.array([[0., 0., 0.]])
        return obs

    def get_objects(self, semantic):
        mapping = self.get_mapping()
        semantic = semantic.astype(np.int32)
        max_key = np.max(np.array(list(mapping.keys())))
        replace_values = []
        for i in np.arange(max_key + 1):
            try:
                replace_values.append(mapping[i])
            except:
                replace_values.append(-1)
        semantic_obs_class = np.take(replace_values, semantic)
        COI_MASK = [(semantic_obs_class == ci).astype(np.int32) for ci in COI_INDEX[self.dn]]  # class mask
        unique_instances = np.unique(semantic * np.sum(np.stack(COI_MASK), 0))[1:]
        semantic_obs_class = semantic_obs_class * np.sum(np.stack(COI_MASK), 0)
        semantic_obs_class[semantic_obs_class == 0] = -1
        bboxes= []
        if len(unique_instances) > 0:
            bbox_ids = unique_instances
            instance_segment = np.stack([(semantic == i).astype(np.int32) for i in unique_instances])
            box_categories = [np.unique(semantic_obs_class[semantic == i])[0] for i in unique_instances]
            if len(instance_segment) > 0:
                object_size = np.stack([np.sum(instance_segman) for instance_segman in instance_segment])
                mask = (object_size > self.img_height * self.img_width / 200)
                instance_segment = [instance_segman for i, instance_segman in enumerate(instance_segment) if mask[i] == 1]
                box_categories = np.stack(box_categories)[mask == 1]
                bbox_ids = np.array(bbox_ids)[mask == 1]
            else:
                bbox_ids = np.array(bbox_ids)
            #     box_categories = np.stack(box_categories)
            x1s = [np.min(np.where(instance_segment[i])[1]) for i in range(len(instance_segment))]
            y1s = [np.min(np.where(instance_segment[i])[0]) for i in range(len(instance_segment))]
            x2s = [np.max(np.where(instance_segment[i])[1]) for i in range(len(instance_segment))]
            y2s = [np.max(np.where(instance_segment[i])[0]) for i in range(len(instance_segment))]
            bboxes = np.stack((x1s, y1s, x2s, y2s), 1)
            if len(bboxes) > 0:
                edge_box_idx = np.where(bboxes[:, 2] - bboxes[:, 0] > self.img_width * 0.8)[0]
                not_edge_box_idx = np.where(bboxes[:, 2] - bboxes[:, 0] <= self.img_width * 0.8)[0]
                if len(edge_box_idx) > 0:
                    x1s1 = [np.min(np.where(instance_segment[i][:, :int(instance_segment[i].shape[1] / 2)])[1]) for i in edge_box_idx]
                    y1s1 = [np.min(np.where(instance_segment[i][:, :int(instance_segment[i].shape[1] / 2)])[0]) for i in edge_box_idx]
                    x2s1 = [np.max(np.where(instance_segment[i][:, :int(instance_segment[i].shape[1] / 2)])[1]) for i in edge_box_idx]
                    y2s1 = [np.max(np.where(instance_segment[i][:, :int(instance_segment[i].shape[1] / 2)])[0]) for i in edge_box_idx]
                    bboxes_1 = np.stack((x1s1, y1s1, x2s1, y2s1), 1)
                    bboxes_1_categories = box_categories[edge_box_idx]
                    bboxes_1_ids = bbox_ids[edge_box_idx]
                    x1s2 = [int(instance_segment[i].shape[1] / 2) + np.min(np.where(instance_segment[i][:, int(instance_segment[i].shape[1] / 2):])[1]) for i in edge_box_idx]
                    y1s2 = [np.min(np.where(instance_segment[i][:, int(instance_segment[i].shape[1] / 2):])[0]) for i in edge_box_idx]
                    x2s2 = [int(instance_segment[i].shape[1] / 2) + np.max(np.where(instance_segment[i][:, int(instance_segment[i].shape[1] / 2):])[1]) for i in edge_box_idx]
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

        if len(bboxes) > 0:
            agent_pose = self.habitat_env.sim.get_agent_state(0).position
            box_world = np.stack([self.habitat_env.object_loc[bbox_id] for bbox_id in bbox_ids])
            box_dist = np.sum((agent_pose - box_world)[:, [0, 2]] ** 2, 1) ** 0.5
            box_depth = np.ones([len(bboxes)]) * 100. #np.array([100.])
            box_score = np.ones([len(bboxes)]) #np.array([100.])
        else:
            bboxes = np.array([[0, 0, self.img_width-1, self.img_height-1]]).astype(np.float32)
            box_categories = np.array([0])
            bbox_ids = np.array([0])
            box_dist = np.array([0.])
            box_depth = np.array([0.])
            box_world = np.array([[-100., -100., -100.]])
            box_score = np.array([0.])
        return bboxes, box_categories, bbox_ids, box_world, box_depth, box_score, semantic_obs_class

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

    def get_xyz(self, depth):
        if np.max(depth) <= 1:
            panoramic_depth = depth.squeeze(-1) * 10.
        else:
            panoramic_depth = depth.squeeze(-1)
        xyz = get_point_cloud_from_z_panoramic(panoramic_depth, self.img_width, self.num_of_camera, self.angles, self.camera_matrix)
        xyz = xyz + np.array([0, 0.88, 0])
        T_world_camera = cam_to_world(q.as_rotation_matrix(self.current_rotation), self.current_position)
        xyz = np.concatenate([xyz.reshape(-1, 3).transpose(), np.ones([1, len(xyz.reshape(-1, 3))])], 0)
        xyz = np.matmul(T_world_camera, xyz)
        self.xyz = xyz[:3].reshape(3, panoramic_depth.shape[0], panoramic_depth.shape[1]).transpose(1,2,0)


class MultiImageGoalEnv(ImageGoalEnv):
    def __init__(self, config=None):
        super(MultiImageGoalEnv, self).__init__(config)

    def step(self, action):
        if isinstance(action, dict):
            action = action['action']
        self._previous_action = action
        if 'STOP' in self.action_space.spaces and action == 0:
            dist = self.get_dist(self.curr_goal.position)
            if dist <= self.success_distance:
                all_done = self.habitat_env.task.measurements.measures['goal_index'].increase_goal_index()
                state = self.habitat_env.sim.get_agent_state()
                obs = self.habitat_env._sim.get_observations_at(state.position, state.rotation)
                if np.max(obs['panoramic_depth']) > 1:
                    obs['panoramic_depth'] = np.clip(obs['panoramic_depth'], self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH, self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH)
                    obs['panoramic_depth'] = (obs['panoramic_depth'] - self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH) / (self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH - self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH)
                obs.update(self.habitat_env.task.sensor_suite.get_observations(
                    observations=obs,
                    episode=self.habitat_env.current_episode,
                    action=action,
                    task=self.habitat_env.task,
                ))
                obs = self.update_objects(obs)
                if all_done:
                    done = True
                    reward = self.SUCCESS_REWARD
                else:
                    done = False
                    reward = 0
            else:
                obs, reward, done, self.info = super(ImageGoalEnv, self).step(self.action_dict[action])
        else:
            obs, reward, done, self.info = super(ImageGoalEnv, self).step(self.action_dict[action])
        obs['target_idx'] = self.curr_goal_idx
        self.timestep += 1

        try:
            self.info['length'] = self.timestep * done
        except:
            self.info = {}
            self.info['length'] = self.timestep * done

        self.info['episode'] = int(self.current_episode.episode_id)
        self.info['distance_to_goal'] = self._previous_measure
        self.info['step'] = self.timestep

        self.info['start_to_goal'] = self.habitat_env._sim.geodesic_distance(self.initial_pose, self.curr_goal.position)
        self.info['scene'] = self.current_episode.scene_id.split("/")[-2]
        self.positions.append(self.current_position)
        if self.args.mode == "collect":
            self.obs = self.process_obs_collect(obs)
        else:
            self.obs = self.process_obs(obs)
        self.total_reward += reward
        self.prev_position = self.current_position.copy()
        self.prev_rotation = self.current_rotation.copy()
        if self.recording_now:
            self.imgs.append(self.render('rgb'))
            if done: self.save_video(self.imgs)
        return self.obs, reward, done, self.info

    @property
    def curr_goal_idx(self):
        if 'GOAL_INDEX' in self.config.TASK_CONFIG.TASK.MEASUREMENTS:
            # return self.habitat_env.get_metrics()['goal_index']['curr_goal_index']
            return self.habitat_env.curr_goal_idx
        else:
            return 0

    def _episode_success(self):
        return self.habitat_env.task.measurements.measures['goal_index'].all_done

    def increase_goal_idx(self):
        return self.habitat_env.task.measurements.measures['goal_index'].increase_goal_index()
