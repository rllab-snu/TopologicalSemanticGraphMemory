# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from enum import Enum
import numpy as np
import habitat_sim
import habitat_sim.agent
from habitat.utils.visualizations import utils as vis_utils
import quaternion as q
import cv2
import imutils
from habitat import logger
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
import copy
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.visualizations import fog_of_war, maps
from habitat.utils.visualizations.maps import TOP_DOWN_MAP_COLORS
from numpy import bool_, float32, float64, ndarray
from habitat_sim.utils.common import quat_from_two_vectors
from quaternion import quaternion
import imageio
import torch
from typing import Any, Dict, List, Optional, Sequence, Tuple
from env_utils.custom_habitat_map import OBJECT_YELLOW, OBJECT_GREEN, OBJECT_GRAY, OBJECT_BLUE, OBJECT_PINK, OBJECT_RED, OBJECT_YELLOW_DIM, OBJECT_GOAL_FLAG, OBJECT_START_FLAG, AGENT_SPRITE
from env_utils.custom_habitat_map import IMAGE_EDGE, IMAGE_NODE, CURR_NODE, COORDINATE_MAX, COORDINATE_MIN, OBJECT_NODE, \
    OBJECT_EDGE, OBJECT_CATEGORY_NODES, DETECTION_CATEGORIES, CATEGORIES
import scipy
from habitat_sim.nav import GreedyGeodesicFollower
TOP_DOWN_MAP_COLORS[IMAGE_NODE] = [119,91,138]
TOP_DOWN_MAP_COLORS[IMAGE_EDGE] = [189,164,204]
TOP_DOWN_MAP_COLORS[CURR_NODE] = [94, 66, 118]
TOP_DOWN_MAP_COLORS[OBJECT_NODE] = [139, 204, 133]
TOP_DOWN_MAP_COLORS[OBJECT_EDGE] = [164, 204, 167]
try:
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
except ImportError:
    pass
_barrier = None
SURFNORM_KERNEL = None


class DemoRunnerType(Enum):
    BENCHMARK = 1
    EXAMPLE = 2


stitcher = cv2.createStitcher(1) if imutils.is_cv3() else cv2.Stitcher_create(1)
vis = False


class TopdownView:
    def __init__(self, sim=None, meters_per_pixel=0.03, dataset="gibson", project_dir=".."):
        self.sim = sim
        self.meters_per_pixel = meters_per_pixel
        self.dataset = dataset
        self.project_dir = project_dir

    def draw_top_down_map(self, height, house):
        if self.dataset == "mp3d":
            height = "%.1f" % height
        else:
            height = "%.2f" % height
        if os.path.isfile(os.path.join(self.project_dir, f"data/explore_map/{house}_{height}.png")):
            top_down_map = cv2.imread(os.path.join(self.project_dir, f"data/explore_map/{house}_{height}.png"), cv2.IMREAD_GRAYSCALE)
            self.clip_tdv = True
        else:
            top_down_map = np.zeros([2000, 2000]).astype(np.int32)
            self.clip_tdv = False
        self.top_down_map = maps.colorize_topdown_map(top_down_map)
        self._map_resolution = self.top_down_map.shape[:2]
        return self.top_down_map

    @staticmethod
    def draw_point(
            image: np.ndarray,
            map_pose: Tuple[int, int],
            radius: float = 0.01,
            color: Tuple[int, int, int] = (50, 50, 50),
            alpha: float = 0.
    ) -> np.ndarray:
        r"""Return an image with the agent image composited onto it.
        Args:
            image: the image onto which to put the agent.
            lidar_coord: the image coordinates where to paste the lidar points.
            lidar_radius: lidar_radius
        Returns:
            The modified background image. This operation is in place.
        """

        point_size = max(3, int(2 * radius))
        overlay = image.copy()
        cv2.circle(
            overlay,
            map_pose,
            radius=point_size,
            color=color,
            thickness=-1,
        )
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        return overlay

    def from_grid(self, mapXY):
        world_x, world_y = maps.from_grid(
            int(mapXY[1]),
            int(mapXY[0]),
            self.top_down_map.shape[0:2],
            sim=self.sim,
        )
        return np.array([world_x, world_y])

    def update_map(self, agent_position):
        a_x, a_y = self.to_grid(
            agent_position[2],
            agent_position[0],
            self.top_down_map.shape[0:2],
            sim=self.sim,
        )

        return a_x, a_y

    def get_polar_angle(self, ref_rotation):
        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        z_neg_z_flip = np.pi
        return np.array(phi) + z_neg_z_flip

    def polar_to_cartesian(self, rho, phi):
        x = np.sqrt(5 / 4 - np.tan(phi)) - 0.5
        y = 1 - x ** 2
        return x, y

    def _compute_quat(self, cam_normal: ndarray) -> quaternion:
        """Rotations start from -z axis"""
        return quat_from_two_vectors(habitat_sim.geo.FRONT, cam_normal)

    def draw_points(self, top_down_map, points, radius=0.01, color=10):
        # plt.plot(lidar[:, 0], lidar[:, 1], "ok")
        # plt.show()
        for point in points:
            top_down_map = self.draw_point(
                image=top_down_map,
                map_pose=(point[0], point[1]),
                radius=radius,
                color=(int(TOP_DOWN_MAP_COLORS[color][0]), int(TOP_DOWN_MAP_COLORS[color][1]), int(TOP_DOWN_MAP_COLORS[color][2]))
            )
        return top_down_map

    @staticmethod
    def draw_object(
            image: np.ndarray,
            object_center_coord: Tuple[int, int],
            object_radius_px: int = 5,
            type: str = "yellow",
            flag_reversed: bool = False
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
        elif type == "red":
            object = OBJECT_RED
        elif type == "start":
            object = OBJECT_START_FLAG
        elif type == "goal":
            object = OBJECT_GOAL_FLAG
        if flag_reversed:
            object = np.ascontiguousarray(np.rot90(np.rot90(np.rot90(object))))
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
            object_center_coord = (
                object_center_coord[0] + int(object_size_px / 2), object_center_coord[1] - int(object_size_px / 2))
        vis_utils.paste_overlapping_image(image, resized_object, object_center_coord)
        return image

    def draw_objects_attention(self, points, top_down_map, objects_attn, radius=1, type="yellow"):
        for idx, point in enumerate(points):
            if idx in objects_attn:
                top_down_map = self.draw_object(
                    image=top_down_map,
                    object_center_coord=(point[1], point[0]),
                    object_radius_px=radius,
                    type=type
                )
        return top_down_map

    def draw_objects(self, points, top_down_map, objects_attn, objects_edge=[], radius=1, type="yellow"):
        if len(objects_edge) > 0:
            for i, pt_i in enumerate(points):
                for j, pt_j in enumerate(points):
                    if objects_edge[i, j] > 0:
                        thickness = int(objects_edge[i, j] * 10)
                        if thickness > 0:
                            cv2.line(
                                top_down_map,
                                tuple(pt_i),
                                tuple(pt_j),
                                color=(100, 100, 100),  # edge: light gray
                                thickness=thickness
                            )

        for idx, point in enumerate(points):
            if idx not in objects_attn:
                top_down_map = self.draw_object(
                    image=top_down_map,
                    object_center_coord=(point[1], point[0]),
                    object_radius_px=radius,
                    type=type
                )

        return top_down_map

    def draw_patch(self, patch_point, top_down_map, radius=25, type="goal", flag_reversed=False):
        top_down_map = self.draw_object(
            image=top_down_map,
            object_center_coord=(patch_point[1], patch_point[0]),
            object_radius_px=int(radius),
            type=type,
            flag_reversed=flag_reversed
        )
        return top_down_map

    def draw_paths(self, top_down_map, path_points, navigable_points=None, thickness=2, radius=1, color=2):
        # Path
        top_down_map = self.draw_path(
            top_down_map=top_down_map,
            path_points=path_points,
            thickness=thickness,
            color=color
        )
        if navigable_points != None:
            for idx, point in enumerate(path_points):
                if navigable_points[idx]:  # navigable
                    color = (255, 0, 0)
                else:  # not navigable
                    color = (0, 0, 255)
                top_down_map = self.draw_point(
                    image=top_down_map,
                    map_pose=(point[0], point[1]),
                    color=color,
                    radius=radius
                )
        return top_down_map

    @staticmethod
    def draw_path(
            top_down_map: np.ndarray,
            path_points: List[Tuple],
            color: int = 10,
            thickness: int = 4,
    ):
        r"""Draw path on top_down_map (in place) with specified color.
        Args:
            top_down_map: A colored version of the map.
            color: color code of the path, from TOP_DOWN_MAP_COLORS.
            path_points: list of points that specify the path to be drawn
            thickness: thickness of the path.
        """
        for prev_pt, next_pt in zip(path_points[:-1], path_points[1:]):
            # Swapping x y
            cv2.line(
                top_down_map,
                tuple(prev_pt),
                tuple(next_pt),
                color=(int(TOP_DOWN_MAP_COLORS[color][0]), int(TOP_DOWN_MAP_COLORS[color][1]), int(TOP_DOWN_MAP_COLORS[color][2])),
                thickness=thickness,
            )
        return top_down_map

    def _draw_path(self,
            top_down_map: np.ndarray,
            p1,
            p2,
            color: int = 10,
            thickness: int = 4,
    ):
        top_down_map = self.draw_path(
            top_down_map,
            [p1, p2],
            color,
            thickness,
        )
        return top_down_map

    @staticmethod
    def to_grid(
            realworld_x: float,
            realworld_y: float,
            grid_resolution: Tuple[int, int],
            sim: Optional["HabitatSim"] = None,
            pathfinder=None,
            bounds=None
    ) -> Tuple[int, int]:
        r"""Return gridworld index of realworld coordinates assuming top-left corner
        is the origin. The real world coordinates of lower left corner are
        (coordinate_min, coordinate_min) and of top right corner are
        (coordinate_max, coordinate_max)
        """
        if sim is None and pathfinder is None and bounds is None:
            raise RuntimeError(
                "Must provide either a simulator or pathfinder instance"
            )

        if sim is not None:
            if pathfinder is None:
                pathfinder = sim.pathfinder

            lower_bound, upper_bound = pathfinder.get_bounds()
        else:
            lower_bound, upper_bound = bounds

        grid_size = (
            abs(upper_bound[2] - lower_bound[2]) / grid_resolution[0],
            abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
        )
        grid_x = int((realworld_x - lower_bound[2]) / grid_size[0])
        grid_y = int((realworld_y - lower_bound[0]) / grid_size[1])
        return grid_x, grid_y

    @staticmethod
    def draw_agent(
            image: np.ndarray,
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
            AGENT_SPRITE, agent_rotation * 180 / np.pi
        )
        # Rescale because rotation may result in larger image than original, but
        # the agent sprite size should stay the same.
        initial_agent_size = AGENT_SPRITE.shape[0]
        new_size = rotated_agent.shape[0]
        agent_size_px = max(
            1, int(agent_radius_px * 2 * new_size / initial_agent_size)
        )
        resized_agent = cv2.resize(
            rotated_agent,
            (agent_size_px, agent_size_px),
            interpolation=cv2.INTER_LINEAR,
        )
        vis_utils.paste_overlapping_image(image, resized_agent, agent_center_coord)
        return image

    def _draw_point(self, top_down_map, position, point_type, ch=None, thickness= 10):
        t_y, t_x = position
        if ch is None:
            top_down_map[
                t_x - thickness : t_x + thickness + 1,
                t_y - thickness : t_y + thickness + 1,
            ] = TOP_DOWN_MAP_COLORS[point_type]
        else:
            top_down_map[
                t_x - thickness : t_x + thickness + 1,
                t_y - thickness : t_y + thickness + 1,
                ch
            ] = TOP_DOWN_MAP_COLORS[point_type]
        return top_down_map

    def _draw_boundary(
            self,
            top_down_map: np.ndarray,
            position: np.ndarray,
            color: int = 10,
            thickness: int = 10,
    ):
        r"""Draw path on top_down_map (in place) with specified color.
        Args:
            top_down_map: A colored version of the map.
            color: color code of the boundary, from TOP_DOWN_MAP_COLORS.
            path_points: list of points that specify the path to be drawn
            thickness: thickness of the boundary.
        """
        t_y, t_x = position
        padd = int(thickness / 2)
        original = copy.deepcopy(top_down_map[
                                 t_x - padd: t_x + padd + 1,
                                 t_y - padd: t_y + padd + 1,
                                 ])
        top_down_map[
        t_x - thickness - 1: t_x + thickness + 2,
        t_y - thickness - 1: t_y + thickness + 2
        ] = TOP_DOWN_MAP_COLORS[color]

        top_down_map[
        t_x - padd: t_x + padd + 1,
        t_y - padd: t_y + padd + 1
        ] = original
        return top_down_map

