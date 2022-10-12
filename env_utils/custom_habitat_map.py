from typing import Any, Dict, List, Optional, Type, Union, Tuple
import numpy as np

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import Measure
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    Simulator,
)
from habitat.core.utils import not_none_validator, try_cv2_import
from habitat.tasks.utils import (
    cartesian_to_polar
)
import math
import skimage.morphology
import quaternion as q
from habitat.utils.visualizations import fog_of_war, maps
from habitat.tasks.nav.nav import TopDownMap
from habitat.utils.visualizations import utils
import os
import imageio
import scipy
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from matplotlib import colors
import csv
from utils.statics import STANDARD_COLORS, CATEGORIES, DETECTION_CATEGORIES, ALL_CATEGORIES, \
    AGENT_SPRITE, OBJECT_YELLOW, OBJECT_YELLOW_DIM, OBJECT_BLUE, OBJECT_GRAY, OBJECT_GREEN, OBJECT_PINK, OBJECT_RED, OBJECT_START_FLAG, OBJECT_GOAL_FLAG
cv2 = try_cv2_import()

agent_colors = ['red','blue', 'yellow', 'green']
AGENT_IMGS = []
for color in agent_colors:
    img = np.ascontiguousarray(np.flipud(imageio.imread(os.path.join(os.path.dirname(__file__), '../data/assets/maps_topdown_agent_sprite/agent_{}.png'.format(color)))))
    AGENT_IMGS.append(img)
import matplotlib.pyplot as plt

# consider up to 5 agents
AGENTS = {}
LAST_INDEX = 10
IMAGE_NODE = LAST_INDEX
IMAGE_EDGE = LAST_INDEX + 1
CURR_NODE = LAST_INDEX + 2
OBJECT_NODE = LAST_INDEX + 3
OBJECT_EDGE = LAST_INDEX + 4
SUBGOAL_NODE = LAST_INDEX + 5
# OBJECT_VIS_EDGE = LAST_INDEX + 6
LAST_INDEX += 6
OBJECT_CATEGORY_NODES = {}
for i in range(len(ALL_CATEGORIES)):
    OBJECT_CATEGORY_NODES[ALL_CATEGORIES[i]] = LAST_INDEX
    LAST_INDEX += 1
MAP_THICKNESS_SCALAR: int = 1250

COORDINATE_MIN = -62.3241 - 1e-6
COORDINATE_MAX = 90.0399 + 1e-6
import torch
import copy
size = 0.5
pt1 = np.array([150-150, 100-150])*size
pt2 = np.array([100-150, 200-150])*size
pt3 = np.array([200-150, 200-150])*size
pt1 = pt1.astype(np.int32)
pt2 = pt2.astype(np.int32)
pt3 = pt3.astype(np.int32)

def draw_agent(
    image: np.ndarray,
    agent_id,
    agent_center_coord: Tuple[int, int],
    agent_rotation: float,
    agent_radius_px: int = 5,
) -> np.ndarray:
    r"""Return an image with the agent image composited onto it.
    Args:
        image: the image onto which to put the agent.
        agent_center_coord: the image coordinates where to paste the agent.
        agent_rotation: the agent's current rotation in radians.
        agent_radius_px: 1/2 number of pixels the agent will be resized to.
    Returns:
        The modified background image. This operation is in place.
    """

    # Rotate before resize to keep good resolution.
    rotated_agent = scipy.ndimage.interpolation.rotate(
        AGENT_IMGS[agent_id], agent_rotation * 180 / np.pi
    )
    # Rescale because rotation may result in larger image than original, but
    # the agent sprite size should stay the same.
    initial_agent_size = AGENT_IMGS[agent_id].shape[0]
    new_size = rotated_agent.shape[0]
    agent_size_px = max(
        2, int(agent_radius_px * 2 * new_size / initial_agent_size)
    )
    resized_agent = cv2.resize(
        rotated_agent,
        (agent_size_px, agent_size_px),
        interpolation=cv2.INTER_LINEAR,
    )
    utils.paste_overlapping_image(image, resized_agent, agent_center_coord)
    return image


def draw_object(
    image: np.ndarray,
    object_center_coord: Tuple[int, int],
    object_radius_px: int = 5,
    type: str = "yellow"
) -> np.ndarray:
    r"""Return an image with the object image composited onto it.
    Args:
        image: the image onto which to put the object.
        object_center_coord: the image coordinates where to paste the object.
        object_radius_px: 1/2 number of pixels the object will be resized to.
    Returns:
        The modified background image. This operation is in place.
    """

    # Rotate before resize to keep good resolution.
    # Rescale because rotation may result in larger image than original, but
    # the object sprite size should stay the same.
    if type == "yellow":
        object = OBJECT_YELLOW
    elif type == "blue":
        object = OBJECT_BLUE
    elif type == "gray":
        object = OBJECT_GRAY
    elif type == "yellow_dim":
        object = OBJECT_YELLOW_DIM
    elif type == "green":
        object = OBJECT_GREEN
    elif type == "pink":
        object = OBJECT_PINK
    elif type == "start":
        object = OBJECT_START_FLAG
    elif type == "goal":
        object = OBJECT_GOAL_FLAG
    initial_object_size = object.shape[0]
    new_size = object.shape[0]
    object_size_px = max(
        1, int(object_radius_px * 2 * new_size / initial_object_size)
    )
    resized_object = cv2.resize(
        object,
        (object_size_px, object_size_px),
        interpolation=cv2.INTER_LINEAR,
    )
    if type == "goal" or type == "start":
        object_center_coord = (object_center_coord[0] - int(object_size_px/2), object_center_coord[1] + int(object_size_px/2))
    utils.paste_overlapping_image(image, resized_object, object_center_coord)
    return image

def get_topdown_map(
    sim: Simulator,
    map_resolution: 1250,
    num_samples: int = 20000,
    draw_border: bool = True,
    save_img: bool=True,
    draw_new_map: bool=False,
    loose_check: bool=False,
    height_th: float=0.1,
    project_dir: str='.',
) -> np.ndarray:
    r"""Return a top-down occupancy map for a sim. Note, this only returns valid
    values for whatever floor the agent is currently ogn.

    Args:
        sim: The simulator.
        map_resolution: The resolution of map which will be computed and
            returned.
        num_samples: The number of random navigable points which will be
            initially
            sampled. For large environments it may need to be increased.
        draw_border: Whether to outline the border of the occupied spaces.

    Returns:
        Image containing 0 if occupied, 1 if unoccupied, and 2 if border (if
        the flag is set).
    """
    # top_down_map = np.zeros(map_resolution, dtype=np.uint8)
    border_padding = 0

    pathfinder = sim.pathfinder
    start_position = sim.get_agent_state().position
    start_height = start_position[1]
    scene_name = sim.habitat_config.SCENE.split('/')[-1][:-4]
    map_name = os.path.join(project_dir, 'data/explore_map/%s_%.2f.png' % (scene_name, start_height))
    lower_bound, upper_bound = pathfinder.get_bounds()
    meters_per_pixel = min(
        abs(upper_bound[coord] - lower_bound[coord]) / map_resolution
        for coord in [0, 2]
    )

    if os.path.exists(map_name) and not draw_new_map and map_resolution == 1250:
        top_down_map = cv2.imread(map_name, cv2.IMREAD_GRAYSCALE)
    else:
        top_down_map = pathfinder.get_topdown_view(
            meters_per_pixel=meters_per_pixel, height=start_height
        ).astype(np.uint8)

        # if save_img:
        cv2.imwrite(map_name, top_down_map)
        top_down_map = cv2.imread(map_name, cv2.IMREAD_GRAYSCALE)

    if draw_border:
        # Recompute range in case padding added any more values.
        range_x = np.where(np.any(top_down_map, axis=1))[0]
        range_y = np.where(np.any(top_down_map, axis=0))[0]
        if len(range_x) > 0 and len(range_y) > 0:
            range_x = (
                max(range_x[0] - border_padding, 0),
                min(range_x[-1] + border_padding + 1, top_down_map.shape[0]),
            )
            range_y = (
                max(range_y[0] - border_padding, 0),
                min(range_y[-1] + border_padding + 1, top_down_map.shape[1]),
            )

            maps._outline_border(
                top_down_map[range_x[0] : range_x[1], range_y[0] : range_y[1]]
            )
    return top_down_map

@registry.register_measure(name='TopDownGraphMap')
class TopDownGraphMap(Measure):
    cls_uuid: str = "top_down_map"
    r"""Top Down Map measure
    """
    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        maps.TOP_DOWN_MAP_COLORS[IMAGE_NODE] = [119,91,138]
        maps.TOP_DOWN_MAP_COLORS[IMAGE_EDGE] = [189,164,204]
        maps.TOP_DOWN_MAP_COLORS[CURR_NODE] = [94, 66, 118]
        maps.TOP_DOWN_MAP_COLORS[OBJECT_NODE] = [139, 204, 133]
        maps.TOP_DOWN_MAP_COLORS[OBJECT_EDGE] = [164, 204, 167]
        maps.TOP_DOWN_MAP_COLORS[SUBGOAL_NODE] = [21, 171, 0]
        for i in range(len(ALL_CATEGORIES)):
            node_color = colors.to_rgb(STANDARD_COLORS[i])
            node_color = (node_color[0] * 255., node_color[1] * 255., node_color[2] * 255.)
            maps.TOP_DOWN_MAP_COLORS[OBJECT_CATEGORY_NODES[ALL_CATEGORIES[i]]] = node_color

        maps.TOP_DOWN_MAP_COLORS[LAST_INDEX:] = cv2.applyColorMap(
            np.arange(256-LAST_INDEX, dtype=np.uint8), cv2.COLORMAP_JET
        ).squeeze(1)[:, ::-1]

        self._sim = sim
        self._config = config
        self._grid_delta = config.MAP_PADDING
        self._step_count = None
        self._map_resolution = config.MAP_RESOLUTION
        self._num_samples = config.NUM_TOPDOWN_MAP_SAMPLE_POINTS
        self._ind_x_min = None
        self._ind_x_max = None
        self._ind_y_min = None
        self._ind_y_max = None
        self._previous_xy_location = None
        self._top_down_map = None
        self._shortest_path_points = None
        self.line_thickness = int(
            np.round(self._map_resolution * 5 / MAP_THICKNESS_SCALAR)
        )
        # max(tdv.shape[0:2]) // 100
        self.point_padding = 10 * int(
            np.ceil(self._map_resolution / MAP_THICKNESS_SCALAR)
        )
        self._previous_scene = None
        self._previous_position = None#self._sim.config.SCENE.split('/')[-2]
        self.delta = 12
        self.milli_delta = 60
        self.delta_angs = [(2*np.pi*i/self.milli_delta) for i in range(self.milli_delta)]
        self.delta_angs = self.delta_angs[30:] + self.delta_angs[:30]
        self.save = []
        self.graph_share = getattr(self._config, 'GRAPH_SHARE', None)
        self.draw_curr_location = getattr(self._config, 'DRAW_CURR_LOCATION', 'point')
        self.record = True
        self.loose_check = False
        self.height_th = 0.1
        self.use_detector = self._config.USE_DETECTOR
        self.dn = self._config.DATASET_NAME
        self.project_dir = self._config.PROJECT_DIR
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "top_down_map"

    def _check_valid_nav_point(self, point: List[float]):
        self._sim.is_navigable(point)

    def get_original_map(self):
        top_down_map = get_topdown_map(
            self._sim,
            self._map_resolution,
            self._num_samples,
            self._config.DRAW_BORDER,
            loose_check=self.loose_check,
            height_th = self.height_th,
            project_dir = self.project_dir
        )

        range_x = np.where(np.any(top_down_map, axis=1))[0]
        range_y = np.where(np.any(top_down_map, axis=0))[0]

        if len(range_x) > 0 and len(range_y) > 0:
            self._ind_x_min = range_x[0]
            self._ind_x_max = range_x[-1]
            self._ind_y_min = range_y[0]
            self._ind_y_max = range_y[-1]
        else:
            top_down_map = get_topdown_map(
                self._sim,
                self._map_resolution,
                self._num_samples,
                self._config.DRAW_BORDER,
                loose_check=self.loose_check,
                height_th=self.height_th,
                draw_new_map=True,
                project_dir = self.project_dir
            )

            range_x = np.where(np.any(top_down_map, axis=1))[0]
            range_y = np.where(np.any(top_down_map, axis=0))[0]
            if len(range_x) > 0 and len(range_y) > 0:
                self._ind_x_min = range_x[0]
                self._ind_x_max = range_x[-1]
                self._ind_y_min = range_y[0]
                self._ind_y_max = range_y[-1]

        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = np.zeros_like(top_down_map)

        self.line_thickness = max(top_down_map.shape[0:2]) // 200
        self.point_padding = max(top_down_map.shape[0:2]) // 100

        return top_down_map

    def _draw_point(self, position, point_type, ch=None, point_padding= None):
        point_padding = self.point_padding if point_padding == None else point_padding
        t_x, t_y = maps.to_grid(
            position[2],
            position[0],
            self._top_down_map.shape[0:2],
            sim=self._sim,
        )
        if ch is None:
            self._top_down_map[
                t_x - point_padding: t_x + point_padding+ 1,
                t_y - point_padding: t_y + point_padding+ 1,
            ] = point_type
        else:
            self._top_down_map[
                t_x - point_padding: t_x + point_padding+ 1,
                t_y - point_padding: t_y + point_padding+ 1,
                ch
            ] = point_type

    def _draw_boundary(self, position, point_type):
        t_x, t_y = maps.to_grid(
            position[2],
            position[0],
            self._top_down_map.shape[0:2],
            sim=self._sim,
        )
        padd = int(self.point_padding/2)
        original = copy.deepcopy(        self._top_down_map[
            t_x - padd : t_x + padd  + 1,
            t_y - padd : t_y + padd  + 1,

        ])
        self._top_down_map[
            t_x - self.point_padding  - 1: t_x + self.point_padding + 2,
            t_y - self.point_padding  - 1 : t_y + self.point_padding + 2
        ] = point_type

        self._top_down_map[
        t_x - padd : t_x + padd  + 1,
        t_y - padd : t_y + padd  + 1
        ] = original

    def _draw_goals_view_points(self, episode):
        if self._config.DRAW_VIEW_POINTS:
            for goal in episode.goals:
                # goal = episode.goals[episode.curr_goal_idx]
                # if True:
                # for goal in episode.goals:
                try:
                    if goal.view_points is not None:
                        for view_point in goal.view_points:
                            if "agent_state" in dir(view_point):
                                self._draw_point(
                                    view_point.agent_state.position,
                                    maps.MAP_VIEW_POINT_INDICATOR,
                                )
                            else:
                                self._draw_point(
                                    view_point,
                                    maps.MAP_VIEW_POINT_INDICATOR,
                                )
                except AttributeError:
                    pass

    def _draw_subgoals_positions(self, subgoal_positions):
        for subgoal_position in subgoal_positions:
            self._draw_point(
                subgoal_position,
                SUBGOAL_NODE,
            )

    def _draw_goals_positions(self, episode):
        if self._config.DRAW_GOAL_POSITIONS:

            # goal = episode.goals[episode.curr_goal_idx]
            # if True:
            for goal in episode.goals:
                try:
                    self._draw_point(
                        goal.position, maps.MAP_TARGET_POINT_INDICATOR
                    )
                except AttributeError:
                    pass

    def _draw_curr_goal_positions(self, goals, goal_mask=None):
        for goal in goals:
            self._draw_point(goal.position, maps.MAP_TARGET_POINT_INDICATOR)

    def _draw_goals_aabb(self, episode):
        if self._config.DRAW_GOAL_AABBS:
            # goal = episode.goals[episode.curr_goal_idx]
            # if True:
            for goal in episode.goals:
                try:
                    sem_scene = self._sim.semantic_annotations()
                    object_id = goal.object_id
                    assert int(
                        sem_scene.objects[object_id].id.split("_")[-1]
                    ) == int(
                        goal.object_id
                    ), f"Object_id doesn't correspond to id in semantic scene objects dictionary for episode: {episode}"

                    center = sem_scene.objects[object_id].aabb.center
                    x_len, _, z_len = (
                        sem_scene.objects[object_id].aabb.sizes / 2.0
                    )
                    # Nodes to draw rectangle
                    corners = [
                        center + np.array([x, 0, z])
                        for x, z in [
                            (-x_len, -z_len),
                            (-x_len, z_len),
                            (x_len, z_len),
                            (x_len, -z_len),
                            (-x_len, -z_len),
                        ]
                    ]

                    map_corners = [maps.to_grid(
                        p[2],
                        p[0],
                        self._top_down_map.shape[0:2],
                        sim=self._sim,
                    )
                        for p in corners
                    ]

                    maps.draw_path(
                        self._top_down_map,
                        map_corners,
                        maps.MAP_TARGET_BOUNDING_BOX,
                        self.line_thickness,
                    )
                except AttributeError:
                    pass

    def _draw_shortest_path(
        self, episode: Episode, agent_position: AgentState
    ):
        if self._config.DRAW_SHORTEST_PATH:
            self._shortest_path_points = self._sim.get_straight_shortest_path_points(
                agent_position, episode.goals[episode.curr_goal_idx].position
            )
            self._shortest_path_points = [maps.to_grid(
                        p[2],
                        p[0],
                        self._top_down_map.shape[0:2],
                        sim=self._sim,
                    )
                for p in self._shortest_path_points
            ]
            maps.draw_path(
                self._top_down_map,
                self._shortest_path_points,
                maps.MAP_SHORTEST_PATH_COLOR,
                self.line_thickness,
            )

    def _draw_path(self, p1, p2, color, ch=None):
        points = [ maps.to_grid(p1[2], p1[0], self._top_down_map.shape[0:2], sim=self._sim),
                   maps.to_grid(p2[2], p2[0], self._top_down_map.shape[0:2], sim=self._sim)]
        maps.draw_path(
            self._top_down_map,
            points,
            color,
            self.line_thickness,
        )

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self.node_list = None
        self.object_node_list = None
        self.object_node_category = None
        self._step_count = 0
        self._metric = None
        self.done_goals = []
        self.curr_goal = None
        if not self.record: return
        self._top_down_map = np.array(self.get_original_map())
        self._fog_of_war_mask = (self._top_down_map==0).astype(np.uint8) #np.zeros_like(self._top_down_map)

        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            self._top_down_map.shape[0:2],
            sim=self._sim,
        )
        self._previous_xy_location = (a_y, a_x)
        self.update_fog_of_war_mask(np.array([a_x, a_y]), agent_state.rotation)
        self.curr_goal_idx = episode.curr_goal_idx
        self._stored_map = copy.deepcopy(self._top_down_map)

        if self._config.DRAW_SOURCE:
            self._draw_point(
                episode.start_position, maps.MAP_SOURCE_POINT_INDICATOR
            )
        self.update_metric(episode, None)

        house_map, map_agent_x, map_agent_y = self.update_map(
            agent_state.position, agent_state.rotation
        )
        # polar_angle = self.get_polar_angle( agent_state.rotation)

        color_top_down_map = maps.colorize_topdown_map(
            self._top_down_map, self._fog_of_war_mask
        )
        top_down_map = draw_agent(
            image=color_top_down_map,
            agent_id=1,
            agent_center_coord=(map_agent_x, map_agent_y),
            agent_rotation= self.get_polar_angle(),
            agent_radius_px=int(self.point_padding * 4)
        )
        fog_of_war_mask = self._fog_of_war_mask
        # range_x = np.where(np.any(top_down_map.mean(-1) - 255, axis=1))[0]
        # range_y = np.where(np.any(top_down_map.mean(-1) - 255, axis=0))[0]
        # pad = 10
        # if len(range_x) > 0 and len(range_y) > 0:
        #     self._ind_x_min = range_x[0]
        #     self._ind_x_max = range_x[-1]
        #     self._ind_y_min = range_y[0]
        #     self._ind_y_max = range_y[-1]
        #
        #     min_x = max(self._ind_x_min - pad, 0)
        #     min_y = max(self._ind_y_min - pad, 0)
        #     top_down_map = top_down_map[min_x: self._ind_x_max + pad, min_y: self._ind_y_max + pad]
        #     fog_of_war_mask = fog_of_war_mask[min_x: self._ind_x_max + pad, min_y: self._ind_y_max + pad]

        self._metric = {
            "map": top_down_map,
            "fog_of_war_mask": fog_of_war_mask,
            "node_list": self.node_list,
            "object_node_list": self.object_node_list,
            "object_node_category": self.object_node_category,
        }

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        if not self.record: return
        self._step_count += 1

        if episode.curr_goal_idx != self.curr_goal_idx:
            self.curr_goal_idx = episode.curr_goal_idx
        self._draw_goals_view_points(episode)
        self._draw_goals_aabb(episode)
        self._draw_goals_positions(episode)
        if 'subgoals' in kwargs:
            self._draw_subgoals_positions(kwargs['subgoals'])

        agent_state = self._sim.get_agent_state()
        house_map, map_agent_x, map_agent_y = self.update_map(
            agent_state.position, agent_state.rotation
        )
        color_top_down_map = maps.colorize_topdown_map(
            house_map, self._fog_of_war_mask
        )
        top_down_map = draw_agent(
            image=color_top_down_map,
            agent_id=1,
            agent_center_coord=(map_agent_x, map_agent_y),
            agent_rotation= self.get_polar_angle(),
            agent_radius_px=int(self.point_padding * 4)
        )
        fog_of_war_mask = self._fog_of_war_mask

        # range_x = np.where(np.any(top_down_map.mean(-1) - 255, axis=1))[0]
        # range_y = np.where(np.any(top_down_map.mean(-1) - 255, axis=0))[0]
        # pad = 10
        # if len(range_x) > 0 and len(range_y) > 0:
        #     self._ind_x_min = range_x[0]
        #     self._ind_x_max = range_x[-1]
        #     self._ind_y_min = range_y[0]
        #     self._ind_y_max = range_y[-1]
        #
        #     min_x = max(self._ind_x_min - pad, 0)
        #     min_y = max(self._ind_y_min - pad, 0)
        #     top_down_map = top_down_map[min_x: self._ind_x_max + pad, min_y: self._ind_y_max + pad]
        #     fog_of_war_mask = fog_of_war_mask[min_x: self._ind_x_max + pad, min_y: self._ind_y_max + pad]

        self._metric = {
            "map": top_down_map,
            "fog_of_war_mask": fog_of_war_mask,
            "node_list": self.node_list,
            "object_node_list": self.object_node_list,
            "object_node_category": self.object_node_category,
        }
        self._top_down_map = copy.deepcopy(self._stored_map)

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

    def update_local_goal(self, goal_position):
        self._draw_point(goal_position, maps.MAP_TARGET_POINT_INDICATOR)

    def update_map(self, agent_position, agent_rotation=None):
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            self._top_down_map.shape[0:2],
            sim=self._sim,
        )
        self.update_fog_of_war_mask(np.array([a_x, a_y]), agent_rotation)

        self._previous_xy_location = (a_y, a_x)
        return self._top_down_map, a_x, a_y

    def draw_semantic_map(self, xyz, category):
        try:
            mask = category > -1
            mask = (
                skimage.morphology.binary_dilation(
                    ~mask, skimage.morphology.disk(4)
                )
                != True
            )
            mask = mask[::2,::2]
            xyz = xyz[::2,::2]
            category = category[::2,::2]
            t_xs, t_ys = self.batch_to_grid(
                xyz[...,2].reshape(-1),
                xyz[...,0].reshape(-1),
                self._top_down_map.shape[0:2],
                sim=self._sim,
            )
            point_padding = self.point_padding + 10
            for c_i, mask_i, t_x, t_y in zip(category.reshape(-1), mask.reshape(-1), t_xs, t_ys):
                if mask_i == True:
                    if self.use_detector:
                        node_color = OBJECT_CATEGORY_NODES[DETECTION_CATEGORIES[int(c_i)]]
                    else:
                        node_color = OBJECT_CATEGORY_NODES[CATEGORIES[self.dn][int(c_i)]]

                    self._stored_map[
                        t_x - point_padding: t_x + point_padding + 1,
                        t_y - point_padding: t_y + point_padding + 1,
                    ] = node_color
        except Exception as e:
            print(e)
            pass

    def batch_to_grid(
            self,
            realworld_xs: float,
            realworld_ys: float,
            grid_resolution: Tuple[int, int],
            sim: Optional["HabitatSim"] = None,
            pathfinder=None,
    ) -> Tuple[int, int]:
        r"""Return gridworld index of realworld coordinates assuming top-left corner
        is the origin. The real world coordinates of lower left corner are
        (coordinate_min, coordinate_min) and of top right corner are
        (coordinate_max, coordinate_max)
        """
        if pathfinder is None:
            pathfinder = sim.pathfinder

        lower_bound, upper_bound = pathfinder.get_bounds()

        grid_size = (
            abs(upper_bound[2] - lower_bound[2]) / grid_resolution[0],
            abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
        )
        grid_x = ((realworld_xs - lower_bound[2]) / grid_size[0]).astype(np.int32)
        grid_y = ((realworld_ys - lower_bound[0]) / grid_size[1]).astype(np.int32)
        return grid_x, grid_y

    def draw_image_graph_on_map(self, node_list, affinity, graph_mask, curr_info={}, flags=None, goal_id=None):
        self.node_list = node_list
        draw_point_list = []

        for idx, node_position in enumerate(node_list):
            neighbors = np.where(affinity[idx])[0]
            node_color_index = IMAGE_NODE
            edge_color_index = IMAGE_EDGE
            for neighbor_idx in neighbors:
                neighbor_position = node_list[neighbor_idx]
                self._draw_path(node_position, neighbor_position, edge_color_index)
            draw_point_list.append([node_position, node_color_index])

        for node_position, node_color_index in draw_point_list:
            self._draw_point(node_position, node_color_index)
        self._draw_boundary(self.node_list[curr_info['curr_node']], CURR_NODE)
        self.curr_info = curr_info

    # def draw_object_graph_on_map(self, node_list, node_category, affinity, graph_mask, curr_info={}, flags=None, goal_id=None):
    #     self.object_node_list = node_list
    #     self.object_node_category = []
    #     draw_point_list = [] #point_padding
    #     node_category = node_category[graph_mask==1]
    #     for idx, node_position in enumerate(node_list):
    #         if self.use_detector:
    #             draw_point_list.append([node_position, OBJECT_CATEGORY_NODES[DETECTION_CATEGORIES[int(node_category[idx])]]])
    #             self.object_node_category.append(DETECTION_CATEGORIES[int(node_category[idx])])
    #         else:
    #             draw_point_list.append([node_position, OBJECT_CATEGORY_NODES[CATEGORIES[self.dn][int(node_category[idx])]]])
    #             self.object_node_category.append(CATEGORIES[self.dn][int(node_category[idx])])
    #
    #     for node_position, node_color in draw_point_list:
    #         t_x, t_y = maps.to_grid(
    #             node_position[2],
    #             node_position[0],
    #             self._top_down_map.shape[0:2],
    #             sim=self._sim,
    #         )
    #         triangle_cnt = np.array(np.array([int(t_y), int(t_x)]) + np.array([pt1, pt2, pt3]).astype(np.int32))
    #         cv2.drawContours(self._top_down_map, [triangle_cnt], 0, [248, 106, 176], -1)
    #         self._draw_point(node_position, node_color, point_padding=20)
    #     self.curr_object_info = curr_info

    def draw_object_graph_on_map(self, node_list, node_category, node_score, vis_node_list, affinity, graph_mask, curr_info={}, flags=None, goal_id=None):
        self.object_node_list = node_list
        draw_point_list = [] #
        self.object_node_category = []
        affinity = affinity[graph_mask==1]
        node_category = node_category[graph_mask==1]
        for idx, node_position in enumerate(node_list):
            if node_score[idx] >= 0.3:# and node_depth[idx] > 0.01:
                neighbors = np.where(affinity[idx])[0]
                if self.use_detector:
                    object_color = OBJECT_CATEGORY_NODES[DETECTION_CATEGORIES[int(node_category[idx])]]
                    draw_point_list.append([node_position, object_color])
                    self.object_node_category.append(DETECTION_CATEGORIES[int(node_category[idx])])
                else:
                    object_color = OBJECT_CATEGORY_NODES[CATEGORIES[self.dn][int(node_category[idx])]]
                    draw_point_list.append([node_position, object_color])
                    self.object_node_category.append(CATEGORIES[self.dn][int(node_category[idx])])
                # for neighbor_idx in neighbors:
                #     neighbor_position = vis_node_list[neighbor_idx]
                #     self._draw_path(node_position, neighbor_position, object_color)

        for node_position, node_color in draw_point_list:
            t_x, t_y = maps.to_grid(
                node_position[2],
                node_position[0],
                self._top_down_map.shape[0:2],
                sim=self._sim,
            )
            # triangle_cnt = np.array(np.array([int(t_y), int(t_x)]) + np.array([pt1, pt2, pt3]).astype(np.int32))
            # self._top_down_map = cv2.drawContours(self._top_down_map.copy(), [triangle_cnt], 0, node_color, -1)
            # self._draw_point(node_position, node_color, point_padding=40)
            self._top_down_map = cv2.circle(self._top_down_map.copy(), (int(t_y), int(t_x)), 40, node_color, -1)
        self.curr_object_info = curr_info

    def update_fog_of_war_mask(self, agent_position, agent_rotation=None):
        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
                self._top_down_map,
                self._fog_of_war_mask,
                agent_position,
                self.get_polar_angle(),
                fov=self._config.FOG_OF_WAR.FOV,
                max_line_len=self._config.FOG_OF_WAR.VISIBILITY_DIST
                / maps.calculate_meters_per_pixel(
                    self._map_resolution, sim=self._sim
                ),
            )