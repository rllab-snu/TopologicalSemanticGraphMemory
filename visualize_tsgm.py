import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse, pickle
import joblib
from env_utils.custom_habitat_map import AGENT_IMGS, OBJECT_CATEGORY_NODES
from habitat.utils.visualizations import utils, maps
from habitat.tasks.utils import cartesian_to_polar
import imageio
from scipy.ndimage.interpolation import rotate
import cv2
import csv
from utils.statics import STANDARD_COLORS, CATEGORIES, DETECTION_CATEGORIES, ALL_CATEGORIES
import matplotlib.patches as mpatches
import glob
from utils.vis_utils import colors_rgb

parser = argparse.ArgumentParser()
parser.add_argument(
    "--tag",
    default="tsgm_gibsontiny",
    type=str,
)
parser.add_argument(
    "--dataset",
    default="gibson",
    type=str,
)
parser.add_argument(
    "--draw-obj-graph",
    action='store_true',
    default=False,
)
parser.add_argument(
    "--draw-im-graph",
    action='store_true',
    default=False,
)
parser.add_argument(
    "--attn-method",
    default="curr",
    type=str,
)
parser.add_argument(
    "--data-dir",
    default="/data4/nuri/tsgm_visualize",
    type=str,
)
parser.add_argument(
    "--file-idx",
    default=-1,
    type=int,
)
args = parser.parse_args()
# index = [12]
fps = 20
args.data_dir = os.path.join(args.data_dir, args.tag)
os.makedirs(args.data_dir, exist_ok=True)
os.makedirs(os.path.join(args.data_dir, "output"), exist_ok=True)
os.makedirs(os.path.join(args.data_dir, "temp"), exist_ok=True)

# AGENT_SPRITE = imageio.imread("./NuriUtils/assets/maps_topdown_agent_sprite/100x100.png")
# AGENT_SPRITE = np.ascontiguousarray(np.flipud(AGENT_SPRITE))
GOAL_FLAG = imageio.imread("./NuriUtils/assets/maps_topdown_flag/flag2_red.png")
IMG_NODE_FLAG = imageio.imread("./NuriUtils/assets/maps_topdown_node/circle.png")
OBJ_NODE_FLAG = imageio.imread("./NuriUtils/assets/maps_topdown_node/yellow_circle.png")
JACKAL_SPRITE = imageio.imread("./NuriUtils/assets/maps_topdown_agent_sprite/jackal.png")
JACKAL_SPRITE = np.ascontiguousarray(np.flipud(JACKAL_SPRITE))
initial_jackal_size = JACKAL_SPRITE.shape[0]
obj_thresh = 0.7
with open(f"./data/scene_info/{args.dataset}/{args.dataset}_bounds.txt", "rb") as fp:  # Unpickling
    bounds = pickle.load(fp)
files = list(np.sort(os.listdir(os.path.join(args.data_dir, "video"))))
# collected_files = list(np.sort(os.listdir(os.path.join(args.data_dir, "output", "gif"))))
if len(files) > 0:
    files = np.stack(files)

if args.file_idx > -1:
    file_indices = [args.file_idx]
else:
    file_indices = list(np.arange(len(files)))
    start_index = 0
    collected_data = os.listdir(os.path.join(args.data_dir, "output"))
    collected_ids = np.unique([int(aa.split(".")[0].split("_")[0]) for aa in collected_data])
    for collected_id in collected_ids:
        try:
            file_indices.remove(collected_id)
        except:
            pass
render_configs = {}
with open(os.path.join("./data/floorplans", f"{args.dataset}_floorplans/render_config.csv")) as csvfile:
    reader = csv.DictReader(csvfile)
    for item in reader:
        if item["level"] == "0":
            render_configs[item["scanId"]] = {}
        render_configs[item["scanId"]][item["level"]] = [
            item["x_low"],
            item["y_low"],
            item["z_low"],
            item["x_high"],
            item["y_high"],
            item["z_high"],
            item["width"],
            item["height"],
            item["Projection"]
        ]


def draw_graph(node_image, i, vis_data, vis_features, attn_method="curr", draw_im_graph=True, draw_obj_graph=True, use_detector=False, font_size=2, font_thickness=2,
               im_node_size=10, obj_node_size=10, im_edge_size=3, obj_edge_size=1, rotated=False):
    node_list = vis_data['graph'][i]['global_memory_pose']
    affinity = vis_data['graph'][i]['global_A']
    # last_localized_imnode = vis_data['graph'][i]['global_idx']
    obj_node_list = vis_data['graph'][i]['object_memory_pose']
    obj_node_category_list = vis_data['graph'][i]['object_memory_category']
    obj_node_score = vis_data['graph'][i]['object_memory_score']
    ov_affinity = vis_data['graph'][i]['object_memory_A_OV']
    global_step = vis_data['global_step'][i]['global_step']
    curr_im_attn = vis_features['features'][global_step]['curr_attn'].reshape(-1).cpu().detach().numpy()
    curr_obj_attn = np.mean(vis_features['features'][global_step]['curr_obj_attn'].squeeze(0).cpu().detach().numpy(), 0)
    goal_obj_attn = vis_features['features'][global_step]['goal_obj_attn'].squeeze(0).cpu().detach().numpy()
    draw_obj_point_list = []
    draw_im_point_list = []
    if draw_obj_graph:
        h, w, _ = node_image.shape
        for idx, node_position in enumerate(obj_node_list):
            if obj_node_score[idx] > obj_thresh:
                if (node_position[0] < upper_bound[0]) & (node_position[2] < upper_bound[2]) & (node_position[0] > lower_bound[0]) & (node_position[2] > lower_bound[2]):
                    try:
                        draw_obj_point_list.append([node_position, obj_node_category_list[idx], curr_obj_attn[idx], goal_obj_attn[idx]])
                    except:
                        draw_obj_point_list.append([node_position, obj_node_category_list[idx], 0, 0])
                    if draw_im_graph:
                        neighbors = np.where(ov_affinity[idx])[0]
                        for neighbor_idx in neighbors:
                            if use_detector:
                                node_color = maps.TOP_DOWN_MAP_COLORS[OBJECT_CATEGORY_NODES[DETECTION_CATEGORIES[int(obj_node_category_list[idx])]]]
                            else:
                                node_color = tuple(colors_rgb[int(obj_node_category_list[idx])])
                            node_color = (int(node_color[0]), int(node_color[1]), int(node_color[2]))
                            neighbor_position = node_list[neighbor_idx]
                            node_position_ = get_map_coord(node_position)[::-1]
                            neighbor_position_ = get_map_coord(neighbor_position)[::-1]
                            line = cv2.line(
                                node_image,
                                tuple(node_position_),
                                tuple(neighbor_position_),
                                node_color,
                                thickness=obj_edge_size
                            )
                            alpha = 0.8
                            node_image = cv2.addWeighted(line, alpha, node_image, 1 - alpha, 0)

    if draw_im_graph:
        for idx, node_position in enumerate(node_list):
            try:
                draw_im_point_list.append([node_position, [15, 119, 143], curr_im_attn[idx]])#, goal_im_attn[idx]])
            except:
                draw_im_point_list.append([node_position, [15, 119, 143], 0.])#, goal_im_attn[idx]])
            neighbors = np.where(affinity[idx])[0]
            for neighbor_idx in neighbors:
                neighbor_position = node_list[neighbor_idx]
                node_position_ = get_map_coord(node_position)
                neighbor_position_ = get_map_coord(neighbor_position)
                cv2.line(
                    node_image,
                    tuple(node_position_[::-1]),
                    tuple(neighbor_position_[::-1]),
                    [177, 232, 246],
                    thickness=im_edge_size,
                )

        for node_position, node_color, curr_att in draw_im_point_list:
            graph_node_center = get_map_coord(node_position)[::-1]
            frame_cpy = node_image.copy()
            if attn_method == "curr":
                att = curr_att
            else:
                att = 0
            if att > 0.1:
                cv2.circle(node_image, tuple(graph_node_center), im_node_size, node_color, -1)
                att = float(att.reshape(-1))
                # node_image = cv2.addWeighted(node_image, att, frame_cpy, 1 - att, gamma=0)
                node_image = cv2.addWeighted(node_image, np.maximum(att, 0.5), frame_cpy, 1 - np.maximum(att, 0.5), gamma=0)
            cv2.circle(node_image, tuple(graph_node_center), im_node_size // 3, node_color, -1)
            if att > 0.1:
                if rotated:
                    H, W = node_image.shape[:2]
                    font_image = np.zeros_like(cv2.rotate(node_image, cv2.ROTATE_90_COUNTERCLOCKWISE)).astype(np.uint8)
                    font_image = cv2.putText(font_image, "%.1f" % (att * 100.), (graph_node_center[1], W - graph_node_center[0]), cv2.FONT_HERSHEY_SIMPLEX, font_size, [255, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)
                    font_image = cv2.rotate(font_image, cv2.ROTATE_90_CLOCKWISE)
                    node_image[font_image.sum(-1) > 0] = 0
                else:
                    node_image = cv2.putText(node_image, "%.1f" % (att * 100.), tuple(graph_node_center), cv2.FONT_HERSHEY_SIMPLEX, font_size, [0, 0, 0], thickness=font_thickness, lineType=cv2.LINE_AA)

    if draw_obj_graph:
        cnt = 0
        arg_idx = list(np.argsort(curr_obj_attn)[::-1][:3])
        for node_position, node_category, curr_obj_attn_, curr_goal_attn in reversed(draw_obj_point_list):
            if use_detector:
                node_color = maps.TOP_DOWN_MAP_COLORS[OBJECT_CATEGORY_NODES[DETECTION_CATEGORIES[int(node_category)]]]
            else:
                # node_color = maps.TOP_DOWN_MAP_COLORS[OBJECT_CATEGORY_NODES[CATEGORIES[dn][int(node_category)]]]
                node_color = tuple(colors_rgb[int(node_category)])
            node_color = (int(node_color[0]), int(node_color[1]), int(node_color[2]))
            if attn_method == "curr":
                obj_att = curr_obj_attn_
            elif attn_method == "goal":
                obj_att = curr_goal_attn
            else:
                obj_att = 0.0
            graph_node_center = get_map_coord(node_position)[::-1]
            frame_cpy = node_image.copy()
            if attn_method != "none":
                obj_att = float(obj_att)
                if cnt in arg_idx:
                    cv2.circle(node_image, tuple(graph_node_center), obj_node_size, node_color, -1)
                node_image = cv2.addWeighted(node_image, np.maximum(obj_att, 0.5), frame_cpy, 1 - np.maximum(obj_att, 0.5), gamma=0)
            cv2.circle(node_image, tuple(graph_node_center), obj_node_size // 2, node_color, -1)
            cnt += 1
        cnt = 0
        for node_position, node_category, curr_obj_attn, curr_goal_attn in reversed(draw_obj_point_list):
            if attn_method == "curr":
                obj_att = curr_obj_attn
            elif attn_method == "goal":
                obj_att = curr_goal_attn
            else:
                obj_att = 0.0
            if cnt in arg_idx:
                graph_node_center = get_map_coord(node_position)[::-1]
                if rotated:
                    H, W = node_image.shape[:2]
                    font_image = np.zeros_like(cv2.rotate(node_image, cv2.ROTATE_90_COUNTERCLOCKWISE)).astype(np.uint8)
                    font_image = cv2.putText(font_image, "%.1f" % (obj_att * 100.), (graph_node_center[1], W - graph_node_center[0]), cv2.FONT_HERSHEY_SIMPLEX, font_size, [255, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)
                    font_image = cv2.rotate(font_image, cv2.ROTATE_90_CLOCKWISE)
                    node_image[font_image.sum(-1) > 0] = 0
                else:
                    node_image = cv2.putText(node_image, "%.1f" % (obj_att * 100.), tuple(graph_node_center), cv2.FONT_HERSHEY_SIMPLEX, font_size, [0, 0, 0], thickness=font_thickness, lineType=cv2.LINE_AA)
            cnt += 1
    return node_image


def get_floor(position, scan_name):
    end_point = np.asarray(position)
    z = end_point[1]
    floor = str(np.argmin([abs(float(render_configs[scan_name][i][2]) - z) for i in render_configs[scan_name].keys()]))
    return floor


def get_map_coord(position):
    A = [position[0] - (upper_bound[0] + lower_bound[0]) / 2, position[2] - (upper_bound[2] + lower_bound[2]) / 2, 1, 1]
    grid_x, grid_y = np.array([imgWidth / 2, imgHeight / 2]) * np.matmul(P, A)[:2] + np.array([imgWidth / 2, imgHeight / 2])
    return tuple(np.array([int(grid_y), int(grid_x)]))


def draw_bbox(object_info, rgb, mode="obs"):
    if mode == "obs":
        bboxes = object_info['object']
        object_mask = bboxes.sum(1) > 0
        bboxes = bboxes[object_mask]
        object_score = object_info['object_score'][object_mask]
        object_category = object_info['object_category'][object_mask].astype(np.int32)
        # object_pose = object_info['object_pose'][object_mask]
    elif mode == "target":
        bboxes = object_info['target_object']
        object_mask = bboxes.sum(1) > 0
        bboxes = bboxes[object_mask]
        object_score = object_info['target_object_score'][object_mask]
        object_category = object_info['target_object_category'][object_mask].astype(np.int32)
        # object_pose = object_info['target_object_pose'][object_mask]
    else:
        raise NotImplementedError
    if len(bboxes) > 0:
        H, W = rgb.shape[:2]
        if bboxes.max() <= 1:
            bboxes[:, 0] = bboxes[:, 0] * W
            bboxes[:, 1] = bboxes[:, 1] * H
            bboxes[:, 2] = bboxes[:, 2] * W
            bboxes[:, 3] = bboxes[:, 3] * H
        bboxes_mask = ((bboxes[:, 2] - bboxes[:, 0]) < W - 2) & ((bboxes[:, 3] - bboxes[:, 1]) < H - 2)
        for bbox_i, bbox in enumerate(bboxes):
            if object_score[bbox_i] > obj_thresh and bboxes_mask[bbox_i]:
                color = tuple(colors_rgb[int(object_category[bbox_i])])
                cv2.rectangle(rgb, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                              (float(color[0]), float(color[1]), float(color[2])), thickness=3)
        for bbox_i, bbox in enumerate(bboxes):
            if object_score[bbox_i] > obj_thresh and bboxes_mask[bbox_i]:
                if len(object_category) > 0:
                    label = CATEGORIES[args.dataset.split("_")[0]][object_category[bbox_i]]
                    imgHeight, imgWidth, _ = rgb.shape
                    cv2.putText(rgb, label, (int(bbox[0]), int(bbox[1]) + 10), 0, 5e-3 * imgHeight, (255, 255, 0), 1)
    return rgb


def create_gif(path_to_images, name_gif):
    filenames = glob.glob(path_to_images)
    filenames = sorted(filenames)
    images = []
    for filename in tqdm(filenames):
        images.append(imageio.imread(filename))
    kargs = {"duration": 0.25}
    # kargs = {'fps': 5.0, 'quantizer': 'nq'}
    imageio.mimsave(name_gif, images, **kargs)#, "GIF-FI", **kargs)


def load_file(file_idx):
    video_name = os.path.join(args.data_dir, "video", files[file_idx])
    data_name = os.path.join(args.data_dir, "others", files[file_idx].replace("_success", "_data_success").replace(".mp4", ".dat.gz"))
    feat_name = os.path.join(args.data_dir, "others", files[file_idx].replace("_success", "_global_success").replace(".mp4", ".dat.gz"))
    vis_data = joblib.load(data_name)
    vis_features = joblib.load(feat_name)
    scan_name = video_name.split("_")[-4]
    target_loc = vis_data['map'][0]['target_loc']
    try:
        floor = get_floor(vis_data['position'][-2][0], scan_name)
    except:
        floor = get_floor(vis_data['position'][0][0], scan_name)
    imgWidth = round(float(render_configs[scan_name][floor][6]))
    imgHeight = round(float(render_configs[scan_name][floor][7]))
    P = np.reshape([float(a) for a in "".join(render_configs[scan_name][floor][8].split("(")[1:]).split(")")[0].split(",")], [4, 4])
    lower_bound = bounds[scan_name][0] #vis_data['map'][0]['lower_bound']
    upper_bound = bounds[scan_name][1]
    return vis_data, vis_features, video_name, scan_name, floor, imgWidth, imgHeight, P, lower_bound, upper_bound


def get_polar_angle(ref_rotation=None):
    vq = np.quaternion(0, 0, 0, 0)
    vq.imag = np.array([0, 0, -1])
    heading_vector = (ref_rotation.inverse() * vq * ref_rotation).imag
    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    x_y_flip = -np.pi / 2
    return np.array(phi) + x_y_flip


# legend = cv2.imread('/disk2/nuri/visualize/legend.png')[...,::-1]

if args.dataset.split("_")[0] == "gibson":
    existing_categories = {}
    cats = ["skateboard", "bottle", "bowl", "bench", "suitcase", "handbag", "couch", "sports ball", "chair", "bed", "tv", "microwave", "sink", "clock", "dining table", "laptop", "keyboard", "oven", "refrigerator", "vase", "potted plant",
            "toilet", "cell phone", "book"]
    for i, name in enumerate(CATEGORIES["gibson"]):
        if name in cats:
            existing_categories[name] = tuple(colors_rgb[CATEGORIES[args.dataset.split("_")[0]].index(name)])  # maps.TOP_DOWN_MAP_COLORS[OBJECT_CATEGORY_NODES[name]]  # (int(color[0]), int(color[1]),int(color[2]))
elif args.dataset == "mp3d":
    existing_categories = {}
    # cats = ["skateboard", "bottle", "bowl", "bench", "suitcase", "handbag", "couch", "sports ball", "chair", "bed", "tv", "microwave", "sink", "clock", "dining table", "laptop", "keyboard", "oven", "refrigerator", "vase", "potted plant",
    #         "toilet", "cell phone", "book"]
    for i, name in enumerate(CATEGORIES["mp3d"]):
        # if name in cats:
        existing_categories[name] = tuple(colors_rgb[CATEGORIES[args.dataset.split("_")[0]].index(name)])  # maps.TOP_DOWN_MAP_COLORS[OBJECT_CATEGORY_NODES[name]]  # (int(color[0]), int(color[1]),int(color[2]))

# file_indices = [33]
for file_idx in tqdm(file_indices):
    vis_data, vis_features, video_name, scan_name, floor, imgWidth, imgHeight, P, lower_bound, upper_bound = load_file(file_idx)
    map_name = os.path.join(args.data_dir, "..", "{}_floorplans/out_dir_rgb_png/output_{}_level_{}.0.png".format(args.dataset, scan_name, floor))
    ortho_map = cv2.imread(map_name)[..., ::-1][..., :3]
    map_name = os.path.join(args.data_dir, "..", "{}_floorplans/out_dir_depth_png/output_{}_level_{}.0.png".format(args.dataset, scan_name, floor))
    ortho_depth = cv2.imread(map_name, 0)
    ortho_depth = (ortho_depth == ortho_depth[0][0])
    aa = np.stack(np.where(ortho_depth == 0), 1)
    ortho_map[ortho_depth == 1] = 255
    x1, y1 = aa[:, 0].min(), aa[:, 1].min()
    x2, y2 = aa[:, 0].max(), aa[:, 1].max()
    map_width, map_height = ortho_depth.shape
    pixel_per_meter = np.maximum((x2 - x1 + 1) / (upper_bound - lower_bound)[0], (y2 - y1 + 1) / (upper_bound - lower_bound)[2])
    jackal_radius_px = int(pixel_per_meter * 0.2)
    goal_size_px = int(pixel_per_meter * 1.0)

    im_node_size = int(pixel_per_meter * 0.3)
    obj_node_size = int(pixel_per_meter * 0.4)
    im_edge_size = int(pixel_per_meter * 0.16)
    obj_edge_size = int(pixel_per_meter * 0.07)
    font_size = np.max([int((x2 - x1 + 1) / 600.), int((y2 - y1 + 1) / 600.), 1])
    font_thickness = np.max([int((x2 - x1 + 1) / 200.), int((y2 - y1 + 1) / 200.), 3])
    # font_size = np.max([int(pixel_per_meter*0.01), 1])
    # font_thickness = np.max([int(pixel_per_meter*0.024),3])

    """
    Save video
    """
    cap = cv2.VideoCapture(video_name)
    input_rgb = []
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            input_rgb.append(frame[:, :, ::-1])
        else:
            break
    cap.release()
    input_rgb = np.array(np.stack(input_rgb).astype(np.uint8))

    # GOAL_FLAG
    rotated = False
    video = []
    fog_of_war_mask = (ortho_depth.astype(np.int32)).copy()[..., None]
    for i in tqdm(np.arange(len(vis_data['map']))):
        map_image = ortho_map.copy()
        map_image[ortho_depth] = 255
        gray_image = cv2.cvtColor(map_image, cv2.COLOR_BGR2GRAY)
        gray_image = gray_image[..., None]
        agent_rotation = vis_data['map'][i]['agent_angle']
        agent_loc = np.array(vis_data['map'][i]['agent_loc'])
        agent_center_coord = get_map_coord(agent_loc)
        map_image = draw_graph(map_image, i, vis_data, vis_features, draw_obj_graph=args.draw_obj_graph, attn_method=args.attn_method,
                               font_size=font_size, font_thickness=font_thickness, im_node_size=im_node_size, obj_node_size=obj_node_size, im_edge_size=im_edge_size, obj_edge_size=obj_edge_size, rotated=rotated)
        # path = np.stack([vis_data['map'][j]['agent_loc'] for j in range(1, len(vis_data['map']))])[:i]
        # map_image = utils.paste_overlapping_image(map_image, goal_flag, target_coord_map)
        rotated_jackal = rotate(JACKAL_SPRITE, agent_rotation * 180 / np.pi)
        new_size = rotated_jackal.shape[0]
        jackal_size_px = max(1, int(jackal_radius_px * 2 * new_size / initial_jackal_size))
        resized_jackal = cv2.resize(rotated_jackal, (jackal_size_px, jackal_size_px), interpolation=cv2.INTER_LINEAR)
        map_image = utils.paste_overlapping_image(map_image, resized_jackal, agent_center_coord)
        map_image = map_image[x1:x2, y1:y2]
        if args.draw_obj_graph:
            input_rgb_i = draw_bbox(vis_data['objects'][i], input_rgb[i])
        else:
            input_rgb_i = input_rgb[i]
        fig, ax = plt.subplots(3, 2, figsize=(9, 5), gridspec_kw={'width_ratios': [2, 1]})
        gs = ax[0, 1].get_gridspec()
        # remove the underlying axes
        ax[0, 1].remove()
        ax[1, 1].remove()
        ax[2, 1].remove()
        axbig = fig.add_subplot(gs[:, 1])
        ax[0][0].imshow(target_rgb)
        # ax[0][0].set_title(f"Target: {vis_data['episode']['goal_name']}")
        ax[0][0].axis("off")
        ax[1][0].imshow(input_rgb_i)
        ax[1][0].set_title(f'Observation at time step {i}')
        ax[1][0].axis("off")
        ax[2][0].legend([mpatches.Patch(color=(v[0] / 255., v[1] / 255., v[2] / 255.)) for k, v in existing_categories.items()],
                        ['{}'.format(k) for k, v in existing_categories.items()], fontsize=1.2, loc='upper left',  # , bbox_to_anchor=(0., 1.0)
                        fancybox=True, ncol=4, prop={'family': 'monospace', 'size': 8})  # , 'size': 2
        ax[2][0].set_title(f'Object Category Legend')
        ax[2][0].margins(0)
        ax[2][0].axis("off")
        if map_image.shape[1] > map_image.shape[0]:
            map_image = cv2.rotate(map_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        axbig.imshow(map_image)
        axbig.set_title(f'Graph Attention')
        axbig.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(args.data_dir, f"temp/{file_idx:03d}_{i:03d}.png"))
        # fig.savefig(os.path.join(args.data_dir, f"temp/{file_idx:03d}_{i:03d}.svg"), format="svg")
        plt.close()
    create_gif(os.path.join(args.data_dir, f"temp/{file_idx:03d}_*.png"), os.path.join(args.data_dir, f"output/{file_idx:03d}.gif"))