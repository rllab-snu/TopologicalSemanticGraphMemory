
import os
import textwrap
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from habitat.core.logging import logger
from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations import maps
import cv2
import copy
import pickle

import scipy
from habitat.utils.visualizations import utils
from matplotlib import colors
from utils.statics import STANDARD_COLORS, CATEGORIES, DETECTION_CATEGORIES, ALL_CATEGORIES
from env_utils.custom_habitat_map import AGENT_IMGS, OBJECT_CATEGORY_NODES
for i in range(len(ALL_CATEGORIES)):
    node_color = colors.to_rgb(STANDARD_COLORS[i])
    node_color = (node_color[0] * 255., node_color[1] * 255., node_color[2] * 255.)
    maps.TOP_DOWN_MAP_COLORS[OBJECT_CATEGORY_NODES[ALL_CATEGORIES[i]]] = node_color

PRED_CURR_NODE = 18
PRED_GOAL_NODE = 6
PRED_OBJECT_CURR_NODE = 29
PRED_OBJECT_GOAL_NODE = 8

colors_rgb: np.ndarray = np.array(
        [(100, 100, 5), (0, 0, 255), (0, 255, 0), (0, 255, 239), (246, 0, 0), (255, 0, 235), (255, 255, 0), (255, 254, 255), (22, 198, 1), (4, 112, 255), (150, 0, 126), (0, 255, 155), (255, 147, 0), (255, 148, 255),
         (0, 0, 123), (0, 110, 125), (0, 244, 67), (175, 0, 0), (81, 0, 255), (120, 51, 0), (10, 16, 189), (135, 255, 0), (153, 204, 255), (255, 2, 128), (203, 0, 73), (247, 255, 171), (178, 3, 255), (69, 239, 31),
         (0, 100, 0), (34, 54, 255), (66, 253, 198), (77, 8, 144), (255, 88, 13), (255, 58, 255), (225, 201, 252), (80, 0, 0), (44, 0, 64), (0, 184, 66), (55, 144, 0), (0, 77, 198), (127, 12, 56), (128, 47, 255),
         (186, 157, 0), (57, 215, 255), (115, 255, 255), (62, 255, 110), (202, 0, 181), (255, 129, 159), (196, 254, 43), (187, 255, 238), (255, 249, 87), (190, 255, 122), (129, 6, 198), (255, 201, 36), (40, 50, 19),
         (28, 55, 138), (0, 69, 75), (0, 174, 134), (0, 172, 255), (103, 191, 0), (0, 208, 197), (77, 52, 82), (116, 152, 255), (151, 105, 0), (77, 50, 195), (85, 97, 34), (0, 136, 187), (196, 57, 0), (255, 39, 62),
         (202, 44, 127), (255, 186, 181), (130, 255, 76), (186, 154, 232), (78, 197, 82), (161, 64, 68), (140, 255, 179), (243, 149, 82), (194, 215, 180), (20, 126, 60), (65, 104, 167), (65, 207, 152), (162, 202, 44),
         (61, 170, 209), (132, 56, 144), (181, 55, 202), (115, 103, 101), (82, 100, 238), (144, 207, 118), (240, 91, 98), (220, 103, 220), (156, 102, 255), (255, 66, 171), (215, 201, 104), (132, 184, 187),
         (115, 150, 53), (96, 155, 132), (180, 123, 68), (192, 158, 153), (181, 100, 141), (133, 125, 189)],
    dtype=np.uint8,
)


def draw_localized(topdownmap, position, sim, NODE_TYPE):
    point_padding = max(topdownmap.shape[0:2]) // 100
    t_x, t_y = maps.to_grid(
        position[2],
        position[0],
        topdownmap.shape[0:2],
        sim=sim,
    )
    padd = int(point_padding/2)
    original = copy.deepcopy(topdownmap[
        t_x - padd : t_x + padd  + 1,
        t_y - padd : t_y + padd  + 1,

    ])
    topdownmap[
        t_x - point_padding  - 1: t_x + point_padding + 2,
        t_y - point_padding  - 1 : t_y + point_padding + 2
    ] = maps.TOP_DOWN_MAP_COLORS[NODE_TYPE]

    topdownmap[
        t_x - padd : t_x + padd  + 1,
        t_y - padd : t_y + padd  + 1
    ] = original
    return topdownmap

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
        1, int(agent_radius_px * 2 * new_size / initial_agent_size)
    )
    resized_agent = cv2.resize(
        rotated_agent,
        (agent_size_px, agent_size_px),
        interpolation=cv2.INTER_LINEAR,
    )
    utils.paste_overlapping_image(image, resized_agent, agent_center_coord)
    return image

def clip_map_birdseye_view(image, clip_size, pixel_pose):
    half_clip_size = clip_size//2

    delta_x = pixel_pose[0] - half_clip_size
    delta_y = pixel_pose[1] - half_clip_size
    min_x = max(delta_x, 0)
    max_x = min(pixel_pose[0] + half_clip_size, image.shape[0])
    min_y = max(delta_y, 0)
    max_y = min(pixel_pose[1] + half_clip_size, image.shape[1])

    return_image = np.zeros([clip_size, clip_size, 3],dtype=np.uint8)
    cliped_image = image[min_x:max_x, min_y:max_y]
    start_x = max(-delta_x,0)
    start_y = max(-delta_y,0)
    try:
        return_image[start_x:start_x+cliped_image.shape[0],start_y:start_y+cliped_image.shape[1]] = cliped_image
    except:
        print('image shape ', image.shape, 'min_x', min_x,'max_x', max_x,'min_y',min_y,'max_y',max_y, 'return_image.shape',return_image.shape, 'cliped', cliped_image.shape, 'start_x,y', start_x, start_y)
    return return_image


def append_text_to_image(image: np.ndarray, text: str, font_size=0.5, font_line=cv2.LINE_AA):
    r""" Appends text underneath an image of size (height, width, channels).
    The returned image has white text on a black background. Uses textwrap to
    split long text into multiple lines.
    Args:
        image: the image to put text underneath
        text: a string to display
    Returns:
        A new image with text inserted underneath the input image
    """
    h, w, c = image.shape
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_image = np.zeros(image.shape, dtype=np.uint8)
    linetype = font_line if font_line is not None else cv2.LINE_8

    char_size = cv2.getTextSize(" ", font, font_size, font_thickness)[0]
    wrapped_text = textwrap.wrap(text, width=int(w / char_size[0]))

    y = 0
    for line in wrapped_text:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        if y % 2 == 1 :
            y += 1
        x = 10
        cv2.putText(
            blank_image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255),
            font_thickness,
            lineType=linetype,
        )
    text_image = blank_image[0 : y + 10, 0:w]
    final = np.concatenate((image, text_image), axis=0)
    return final


def draw_collision(view: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    r"""Draw translucent red strips on the border of input view to indicate
    a collision has taken place.
    Args:
        view: input view of size HxWx3 in RGB order.
        alpha: Opacity of red collision strip. 1 is completely non-transparent.
    Returns:
        A view with collision effect drawn.
    """
    strip_width = view.shape[0] // 20
    mask = np.ones(view.shape)
    mask[strip_width:-strip_width, strip_width:-strip_width] = 0
    mask = mask == 1
    view[mask] = (alpha * np.array([255, 0, 0]) + (1.0 - alpha) * view)[mask]
    return view


def observations_to_rgbmap(observation: Dict, info: Dict, mode='panoramic', local_imgs=None, clip=False, use_detector = False, task_name="imggoal",
                          dataset_name="gibson", draw_object=True, attns=None, sim=None, rgbmap=None) -> \
        np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    size = 2.0
    egocentric_view = []
    if "rgb" in observation and mode != 'panoramic':
        rgb = observation["rgb"]
    elif "panoramic_rgb" in observation and mode == 'panoramic':
        rgb = observation['panoramic_rgb']

    if not isinstance(rgb, np.ndarray):
        rgb = rgb.cpu().numpy()

    if "object" in observation and draw_object:
        object_mask = observation['object_score'] >= 0.3
        num_objects = object_mask.sum()
        if num_objects > 0:
            bboxes = observation["object"][object_mask == 1]
            if bboxes.shape[1] == 5:
                bboxes = bboxes[:, 1:]
            bbox_category = observation["object_category"][object_mask == 1]
            rgb = draw_bbox(rgb, bboxes, bbox_category, use_detector=use_detector, dataset_name=dataset_name)

    rgb = cv2.putText(np.ascontiguousarray(rgb), 'current obs', (5,10), cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255))
    egocentric_view.append(rgb)

    if "target_goal" in observation and len(observation['target_goal']) > 0:
        goal_rgb = (observation['target_goal']*255) #TODO: ?
        if not isinstance(goal_rgb, np.ndarray):
            goal_rgb = goal_rgb.cpu().numpy()
        if len(goal_rgb.shape) == 4:
            if info is not None:
                goal_rgb = goal_rgb * (1 - info['total_success']).reshape(-1, *[1] * len(goal_rgb.shape[1:]))
            goal_rgb = np.concatenate(np.split(goal_rgb[:,:,:,:3],goal_rgb.shape[0],axis=0),1).squeeze(axis=0)
        else:
            goal_rgb = goal_rgb[:,:,:3]
        goal_rgb = goal_rgb.astype(np.uint8)

        if "target_object" in observation:
            bboxes = observation["target_object"][:, 1:]
            bbox_category = observation["target_object_category"]
            goal_rgb = draw_bbox(goal_rgb.astype(np.uint8), bboxes, bbox_category, use_detector=use_detector)
            if task_name == "ObjTarget":
                if use_detector:
                    goal_rgb = cv2.putText(np.ascontiguousarray(goal_rgb), f'target_obs: {DETECTION_CATEGORIES[int(bbox_category[0])]}',(5,10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255))
                else:
                    goal_rgb = cv2.putText(np.ascontiguousarray(goal_rgb), f'target_obs: {CATEGORIES[dataset_name][int(bbox_category[0])]}',(5,10),
                                           cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255))
        else:
            goal_rgb = cv2.putText(np.ascontiguousarray(goal_rgb), 'target_obs',(5,10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255))
        egocentric_view.append(goal_rgb)

    if len(egocentric_view) > 0:
        if mode == 'panoramic':
            egocentric_view = np.concatenate(egocentric_view, axis=0)
        else:
            egocentric_view = np.concatenate(egocentric_view, axis=1)
        if "collisions" in info and info['collisions'] is not None:
            if info["collisions"]["is_collision"]:
                egocentric_view = draw_collision(egocentric_view)
        frame = cv2.resize(egocentric_view, dsize=None, fx=size*0.75, fy=size)
    else:
        frame = None

    if info is not None and "top_down_map" in info:
        if info['top_down_map'] is not None:
            top_down_height = frame.shape[0] if frame is not None else info["top_down_map"]["map"].shape[0]
            top_down_map = info["top_down_map"]["map"]

            if attns != None:
                pred_goal = int(attns['goal_attn'][0].argmax(-1))
                pred_goal = info["top_down_map"]["node_list"][pred_goal]
                top_down_map = draw_localized(top_down_map, pred_goal, sim, PRED_GOAL_NODE)
                pred_curr = int(attns['curr_attn'][0].argmax(-1))
                pred_curr = info["top_down_map"]["node_list"][pred_curr]
                top_down_map = draw_localized(top_down_map, pred_curr, sim, PRED_CURR_NODE)
                # pred_curr_obj = int(attns['curr_obj_attn'][0].sum(0).argmax())
                try:
                    pred_curr_obj = int(attns['curr_obj_attn'][0].max(0).values.argmax(-1))
                    pred_curr_obj = info["top_down_map"]["object_node_list"][pred_curr_obj]
                    top_down_map = draw_localized(top_down_map, pred_curr_obj, sim, PRED_OBJECT_CURR_NODE)
                except:
                    pass
                # pred_goal_obj = int(attns['goal_obj_attn'][0].sum(0).argmax())
                try:
                    pred_goal_obj = int(attns['goal_obj_attn'][0].max(0).values.argmax(-1))
                    pred_goal_obj = info["top_down_map"]["object_node_list"][pred_goal_obj]
                    top_down_map = draw_localized(top_down_map, pred_goal_obj, sim, PRED_OBJECT_GOAL_NODE)
                except:
                    pass
                # plt.imshow(top_down_map)
                # plt.show()
            # scale top down map to align with rgb view
            old_h, old_w, _ = top_down_map.shape
            top_down_width = int(float(top_down_height) / old_h * old_w)
            # cv2 resize (dsize is width first)
            if frame is not None:
                top_down_map = cv2.resize(
                    top_down_map,
                    (top_down_width, top_down_height),
                    interpolation=cv2.INTER_CUBIC,
                )
        else:
            height = frame.shape[0] if frame is not None else 512
            top_down_map = np.zeros([height, height, 3],dtype=np.uint8)

        frame = np.concatenate((frame, top_down_map), axis=1) if frame is not None else top_down_map

    return frame

def observations_to_image(observation: Dict, info: Dict, mode='panoramic', local_imgs=None, clip=False, use_detector = False, task_name="imggoal",
                          dataset_name="gibson", draw_object=True, attns=None, sim=None, rgbmap=None) -> \
        np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    size = 2.0
    egocentric_view = []
    if "rgb" in observation and mode != 'panoramic':
        rgb = observation["rgb"]
    elif "panoramic_rgb" in observation and mode == 'panoramic':
        rgb = observation['panoramic_rgb']

    if not isinstance(rgb, np.ndarray):
        rgb = rgb.cpu().numpy()

    if "object" in observation and draw_object:
        object_mask = observation['object_score'] >= 0.3
        num_objects = object_mask.sum()
        if num_objects > 0:
            bboxes = observation["object"][object_mask == 1]
            if bboxes.shape[1] == 5:
                bboxes = bboxes[:, 1:]
            bbox_category = observation["object_category"][object_mask == 1]
            rgb = draw_bbox(rgb.copy(), bboxes, bbox_category, use_detector=use_detector, dataset_name=dataset_name)

    rgb = cv2.putText(np.ascontiguousarray(rgb), 'current obs', (5,10), cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255))
    egocentric_view.append(rgb)

    if "target_goal" in observation and len(observation['target_goal']) > 0:
        goal_rgb = (observation['target_goal']*255) #TODO: ?
        if not isinstance(goal_rgb, np.ndarray):
            goal_rgb = goal_rgb.cpu().numpy()
        if len(goal_rgb.shape) == 4:
            if info is not None:
                goal_rgb = goal_rgb * (1 - info['total_success']).reshape(-1, *[1] * len(goal_rgb.shape[1:]))
            goal_rgb = np.concatenate(np.split(goal_rgb[:,:,:,:3],goal_rgb.shape[0],axis=0),1).squeeze(axis=0)
        else:
            goal_rgb = goal_rgb[:,:,:3]
        goal_rgb = goal_rgb.astype(np.uint8)

        if "target_object" in observation:
            bboxes = observation["target_object"][:, 1:]
            bbox_category = observation["target_object_category"]
            goal_rgb = draw_bbox(goal_rgb, bboxes, bbox_category, use_detector=use_detector)
            if task_name == "ObjTarget":
                if use_detector:
                    goal_rgb = cv2.putText(np.ascontiguousarray(goal_rgb), f'target_obs: {DETECTION_CATEGORIES[int(bbox_category[0])]}',(5,10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255))
                else:
                    goal_rgb = cv2.putText(np.ascontiguousarray(goal_rgb), f'target_obs: {CATEGORIES[dataset_name][int(bbox_category[0])]}',(5,10),
                                           cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255))
        else:
            goal_rgb = cv2.putText(np.ascontiguousarray(goal_rgb), 'target_obs',(5,10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255))
        egocentric_view.append(goal_rgb)

    if len(egocentric_view) > 0:
        if mode == 'panoramic':
            egocentric_view = np.concatenate(egocentric_view, axis=0)
        else:
            egocentric_view = np.concatenate(egocentric_view, axis=1)
        if "collisions" in info and info['collisions'] is not None:
            if info["collisions"]["is_collision"]:
                egocentric_view = draw_collision(egocentric_view)
        frame = cv2.resize(egocentric_view, dsize=None, fx=size*0.75, fy=size)
    else:
        frame = None

    if info is not None and "top_down_map" in info:
        if info['top_down_map'] is not None:
            top_down_height = frame.shape[0] if frame is not None else info["top_down_map"]["map"].shape[0]
            top_down_map = info["top_down_map"]["map"]

            try:
                if attns != None:
                    pred_goal = int(attns['goal_attn'][0].argmax(-1))
                    pred_goal = info["top_down_map"]["node_list"][pred_goal]
                    top_down_map = draw_localized(top_down_map, pred_goal, sim, PRED_GOAL_NODE)
                    pred_curr = int(attns['curr_attn'][0].argmax(-1))
                    pred_curr = info["top_down_map"]["node_list"][pred_curr]
                    top_down_map = draw_localized(top_down_map, pred_curr, sim, PRED_CURR_NODE)
                    # pred_curr_obj = int(attns['curr_obj_attn'][0].sum(0).argmax())
                    try:
                        pred_curr_obj = int(attns['curr_obj_attn'][0].max(0).values.argmax(-1))
                        pred_curr_obj = info["top_down_map"]["object_node_list"][pred_curr_obj]
                        top_down_map = draw_localized(top_down_map, pred_curr_obj, sim, PRED_OBJECT_CURR_NODE)
                        print(info["top_down_map"]["object_node_category"][attns['curr_obj_attn'][0].max(-1)[1][0].item()])
                    except:
                        pass
                    # pred_goal_obj = int(attns['goal_obj_attn'][0].sum(0).argmax())
                    try:
                        pred_goal_obj = int(attns['goal_obj_attn'][0].max(0).values.argmax(-1))
                        pred_goal_obj = info["top_down_map"]["object_node_list"][pred_goal_obj]
                        top_down_map = draw_localized(top_down_map, pred_goal_obj, sim, PRED_OBJECT_GOAL_NODE)
                        print(info["top_down_map"]["object_node_category"][attns['goal_obj_attn'][0].max(-1)[1][0].item()])
                    except:
                        pass
            except:
                pass

                # plt.imshow(top_down_map)
                # plt.show()
            # scale top down map to align with rgb view
            old_h, old_w, _ = top_down_map.shape
            top_down_width = int(float(top_down_height) / old_h * old_w)
            # cv2 resize (dsize is width first)
            if frame is not None:
                top_down_map = cv2.resize(
                    top_down_map,
                    (top_down_width, top_down_height),
                    interpolation=cv2.INTER_CUBIC,
                )
        else:
            height = frame.shape[0] if frame is not None else 512
            top_down_map = np.zeros([height, height, 3],dtype=np.uint8)

        frame = np.concatenate((frame, top_down_map), axis=1) if frame is not None else top_down_map

    return frame

def draw_bbox(rgb: np.ndarray, bboxes: np.ndarray, bbox_category = [], use_detector=False, dataset_name="gibson") -> np.ndarray:
    H, W = rgb.shape[:2]
    if bboxes.max() <= 1:
        bboxes[:, 0] = bboxes[:, 0] * W
        bboxes[:, 1] = bboxes[:, 1] * H
        bboxes[:, 2] = bboxes[:, 2] * W
        bboxes[:, 3] = bboxes[:, 3] * H
    for i, bbox in enumerate(bboxes):
        if use_detector:
            color = maps.TOP_DOWN_MAP_COLORS[OBJECT_CATEGORY_NODES[DETECTION_CATEGORIES[int(bbox_category[i])]]]
        else:
            color = maps.TOP_DOWN_MAP_COLORS[OBJECT_CATEGORY_NODES[CATEGORIES[dataset_name][int(bbox_category[i])]]]

        cv2.rectangle(rgb, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (float(color[0]), float(color[1]), float(color[2])), 3)
        if len(bbox_category) > 0:
            if use_detector:
                label = DETECTION_CATEGORIES[int(bbox_category[i])]
            else:
                label = CATEGORIES[dataset_name][int(bbox_category[i])]
            imgHeight, imgWidth, _ = rgb.shape
            cv2.putText(rgb, label, (int(bbox[0]), int(bbox[1]) + 10), 0, 5e-3 * imgHeight, (255,255,0), 1)
    return rgb


def concat_horizontal(img1, img2):
    img1_shape = img1.shape
    img2_shape = img2.shape
    if img1_shape[0] > img2_shape[0]:
        img2 = cv2.resize(img2, ( int(img2_shape[1] / img2_shape[0] * img1_shape[0]), img1_shape[0]))
        # img2 = cv2.resize(img2, (img1_shape[0], img2_shape[1]))
    elif img1_shape[0] < img2_shape[0]:
        img1 = cv2.resize(img1, ( int(img1_shape[1] / img1_shape[0] * img2_shape[0]), img2_shape[0]))
    return np.concatenate((img1, img2), axis=1)


def concat_vertical(img1, img2):
    img1_shape = img1.shape
    img2_shape = img2.shape
    if img1_shape[1] > img2_shape[1]:
        img2 = cv2.resize(img2, (img1_shape[1], int(img2_shape[0] / img2_shape[1] * img1_shape[1])))
    elif img1_shape[1] < img2_shape[1]:
        img1 = cv2.resize(img1, (img2_shape[1], int(img1_shape[0] / img1_shape[1] * img2_shape[1])))
    return np.concatenate((img1, img2), axis=0)

