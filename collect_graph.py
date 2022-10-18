import argparse, glob, joblib, torch, os, parmap, numpy as np
from configs.default import get_config
from torchvision.ops import nms as torch_nms
import quaternion as q
from env_utils import *

torch.set_num_threads(5)
torch.backends.cudnn.enabled = True
os.environ['GLOG_minloglevel'] = "3"
os.environ['MAGNUM_LOG'] = "quiet"
os.environ['HABITAT_SIM_LOG'] = "quiet"

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--config", type=str, default="configs/TSGM.yaml", help="path to config yaml containing info about experiment")
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--split", choices=['val', 'train', 'min_val'], default='train')
parser.add_argument('--record', choices=['0','1','2','3'], default='0') # 0: no record 1: env.render 2: pose + action numerical traj 3: features
parser.add_argument('--img-node-th', type=str, default='0.75')
parser.add_argument('--obj-node-th', type=str, default='0.8')
parser.add_argument('--obj-score-th', default=0.3, type=float)
parser.add_argument('--dataset', default='gibson', type=str)
parser.add_argument('--task', default='imggoalnav', type=str)
parser.add_argument('--project-dir', default='.', type=str)
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--policy', default='TSGMPolicy', type=str)
parser.add_argument('--mode', default='collect_graph', type=str)
parser.add_argument('--data-dir', default='IL_data/gibson_fd', type=str)
parser.add_argument('--record-dir', type=str, default='data')
parser.add_argument('--num-procs', default=16, type=int)

args = parser.parse_args()
args.record = int(args.record)
args.img_node_th = float(args.img_node_th)
args.obj_node_th = float(args.obj_node_th)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.gpu != 'cpu':
    torch.cuda.manual_seed(args.seed)
device = 'cpu' if args.gpu == '-1' else 'cuda:{}'.format(args.gpu)


def collect_graph(data_list):
    data_list = [data_list] if type(data_list) is not list else data_list
    config = collect_config(args)
    env = eval(config.ENV_NAME)(config)
    graph_dir = os.path.join(args.record_dir, 'graph', args.split)
    with torch.no_grad():
        for data_path in data_list:
            batch = pull_image(data_path[0], config)
            record_graphs = []
            for t in range(batch['panoramic_rgb'].shape[0]):
                obs_t = {
                    'panoramic_rgb': batch['panoramic_rgb'][t],
                    'panoramic_depth': batch['panoramic_depth'][t],
                    'position': batch['position'][t],
                    'rotation': batch['rotation'][t],
                    'object': batch['object'][t],
                    'object_mask': batch['object_mask'][t],
                    'object_score': batch['object_score'][t],
                    'object_category': batch['object_category'][t],
                    'object_pose': batch['object_pose'][t],
                    'object_depth': np.sqrt(np.sum((batch['object_pose'][t] - batch['position'][t])[:, [0, 2]] ** 2, -1)),
                    'step': t,
                }
                if t == 0:
                    env.build_graph(obs_t, reset=True)
                else:
                    env.build_graph((obs_t,None,None,None))
                max_num_img_node = env.imggraph.num_node()
                max_num_obj_node = env.objgraph.num_node()
                img_memory_dict = {
                    'img_memory_feat': env.imggraph.graph_memory[:max_num_img_node].copy(),
                    'img_memory_pose': np.stack(env.imggraph.node_position_list).copy(),
                    'img_memory_mask': env.imggraph.graph_mask[:max_num_img_node].copy(),
                    'img_memory_A': env.imggraph.A[:max_num_img_node, :max_num_img_node].copy(),
                    'img_memory_idx': env.imggraph.last_localized_node_idx,
                    'img_memory_time': env.imggraph.graph_time[:max_num_img_node].copy()
                }
                obj_memory_dict = {
                    'obj_memory_feat': env.objgraph.graph_memory[:max_num_obj_node].copy(),
                    'obj_memory_pose': np.stack(env.objgraph.node_position_list).copy(),
                    'obj_memory_score': env.objgraph.graph_score[:max_num_obj_node].copy(),
                    'obj_memory_category': env.objgraph.graph_category[:max_num_obj_node].copy(),
                    'obj_memory_mask': env.objgraph.graph_mask[:max_num_obj_node].copy(),
                    'obj_memory_A_OV': env.objgraph.A_OV[:max_num_obj_node, :max_num_img_node].copy(),
                    'obj_memory_time': env.objgraph.graph_time[:max_num_obj_node].copy()
                }
                img_memory_dict.update(obj_memory_dict)
                record_graphs.append(img_memory_dict)
            file_name = os.path.join(graph_dir, data_path.split('/')[-1])
            data = {'graph': record_graphs}
            joblib.dump(data, file_name)
            print(f"Processing... {len(glob.glob(os.path.join(graph_dir, '*.dat.gz')))}/{len(glob.glob(os.path.join(args.data_dir, args.split, '*.dat.gz')))}.")
            del data


def pull_image(data_path, config):
    input_data = joblib.load(data_path)
    scene = data_path.split('/')[-1].split('_')[0]
    input_rgb = np.array(input_data['rgb'], dtype=np.float32)
    input_dep = np.array(input_data['depth'], dtype=np.float32)
    max_num_object = config.memory.num_objects
    input_object = input_data['object']
    input_object_category = input_data['object_category']
    input_object_pose = input_data['object_pose']
    input_object_score = input_data['object_score']
    max_input_length = input_rgb.shape[0]
    input_object_out = np.zeros((max_input_length, max_num_object, 5))
    input_object_score_out = np.zeros((max_input_length, max_num_object))
    input_object_category_out = np.ones((max_input_length, max_num_object)) * (-1)
    input_object_mask_out = np.zeros((max_input_length, max_num_object))
    input_object_pose_out = -100. * np.ones((max_input_length, max_num_object, 3))
    for i in range(len(input_object)):
        if len(input_object[i]) > 0:
            input_object_t = np.array(input_object[i]).reshape(-1, 4)
            input_object_score_t = np.array(input_object_score[i]).reshape(-1)
            input_object_score_t = input_object_score_t[:len(input_object_t)]
            keep = torch_nms(torch.from_numpy(input_object_t).float(), torch.from_numpy(input_object_score_t).float(), 0.5)
            input_object_t = np.array(input_object_t[keep]).reshape(-1, 4)
            input_object_score_t = np.array(input_object_score_t[keep]).reshape(-1)
            input_object_category_t = np.array(input_object_category[i]).reshape(-1)[keep].reshape(-1)
            input_object_pose_t = np.array(input_object_pose[i][keep]).reshape(-1, 3)

            score_mask = input_object_score_t > args.obj_score_th
            input_object_t = input_object_t[score_mask]
            input_object_score_t = input_object_score_t[score_mask]
            input_object_category_t = input_object_category_t[score_mask]
            input_object_pose_t = input_object_pose_t[score_mask]

            num_object_t = len(input_object_t)

            input_object_out[i, :min(max_num_object, num_object_t), 1:] = input_object_t[:min(max_num_object, num_object_t), :4]
            input_object_score_out[i, :min(max_num_object, num_object_t)] = input_object_score_t[:min(max_num_object, num_object_t)]
            input_object_category_out[i, :min(max_num_object, num_object_t)] = input_object_category_t[:min(max_num_object, num_object_t)]
            input_object_pose_out[i, :min(max_num_object, num_object_t)] = input_object_pose_t[:min(max_num_object, num_object_t)]
            input_object_mask_out[i, :min(max_num_object, num_object_t)] = 1

    train_info = {}
    train_info["panoramic_rgb"] = input_rgb
    train_info["panoramic_depth"] = input_dep
    train_info["object"] = input_object_out
    train_info["object_mask"] = input_object_mask_out
    train_info["object_score"] = input_object_score_out
    train_info["object_category"] = input_object_category_out
    train_info["object_pose"] = input_object_pose_out
    train_info["position"] = np.stack(input_data['position'])
    train_info["rotation"] = q.as_euler_angles(np.array(q.from_float_array(input_data['rotation'])))[:, 1]
    train_info["scene"] = scene
    return train_info


def collect_config(args):
    if os.path.exists("./configs/{}_{}.yaml".format(args.task, args.dataset)):
        config_path = "./configs/{}_{}.yaml".format(args.task, args.dataset)
    else:
        raise ValueError("No config file for {}_{}".format(args.task, args.dataset))
    config = get_config(args.config, base_task_config_path=config_path, arguments=vars(args))
    config.defrost()
    config.use_depth = config.TASK_CONFIG.use_depth = True
    config.scene_data = args.dataset
    config.DATASET_NAME = args.dataset
    config.TASK_CONFIG.DATASET.DATASET_NAME = args.dataset
    config.ACTION_DIM = 4
    config.ENV_NAME = "ImageGoalGraphEnv"
    config.TASK_CONFIG['ARGS'] = vars(args)
    config.features.object_category_num = 80
    config.TASK_CONFIG.img_node_th = config.img_node_th = args.img_node_th
    config.TASK_CONFIG.obj_node_th = config.obj_node_th = args.obj_node_th
    config.record = False
    config.render = False
    config.render_map = False
    config.noisy_actuation = False
    config.freeze()
    return config


if __name__=='__main__':
    config = collect_config(args)

    print('====================================')
    print('Dataset Name: ', args.dataset)
    print('Split: ', args.split)
    print('Image Graph Threshold: ', config.TASK_CONFIG.img_node_th)
    print('Object Graph Threshold: ', config.TASK_CONFIG.obj_node_th)
    print('====================================')

    os.makedirs(os.path.join(args.record_dir, 'graph'), exist_ok=True)
    graph_dir = os.path.join(args.record_dir, 'graph', args.split)
    os.makedirs(graph_dir, exist_ok=True)
    existing_files = glob.glob(graph_dir + "/*")

    data_list = [os.path.join(args.data_dir, args.split, x) for x in sorted(os.listdir(os.path.join(args.data_dir, args.split))) if "dat.gz" in x]
    existing_files = [os.path.join(args.data_dir, args.split, data_path.split('/')[-1]) for data_path in existing_files]
    if len(existing_files) > 0:
        for ef in existing_files:
            if ef in data_list:
                data_list.remove(ef)
    num_data = len(data_list)
    data_list = np.stack(sorted(data_list))
    collect_graph(data_list)
    # parmap.map(collect_graph, data_list, pm_processes = args.num_procs)