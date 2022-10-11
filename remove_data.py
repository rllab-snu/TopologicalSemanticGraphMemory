import os, argparse, torch, joblib, numpy as np
from tqdm import tqdm
project_dir = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/TSGM.yaml", help="path to config yaml containing info about experiment")
parser.add_argument("--prebuild-path", type=str, default="data/graph", help="path to prebuild graph")
parser.add_argument("--gpu", type=str, default="0", help="gpus",)
parser.add_argument("--num-gpu", type=int, default=1, help="gpus",)
parser.add_argument("--version", type=str, default="test", help="name to save")
parser.add_argument('--data-dir', default='IL_data/gibson_noisy', type=str)
parser.add_argument('--project-dir', default='.', type=str)
parser.add_argument('--dataset', default='gibson', type=str)
parser.add_argument('--resume', default='none', type=str)
parser.add_argument('--task', default='imggoalnav', type=str)
parser.add_argument('--num-object', default=10, type=int)
parser.add_argument('--memory-size', default=0, type=int)
parser.add_argument('--multi-target', action='store_true', default=False)
parser.add_argument('--mode', default='train_il', type=str)
parser.add_argument('--record', default=0, type=int)
parser.add_argument('--detector-th', default=0.01, type=float)
parser.add_argument('--img-node-th', type=str, default='0.75')
parser.add_argument('--obj-node-th', type=str, default='0.8')
parser.add_argument("--wandb", action='store_true')
parser.add_argument('--debug', action='store_true', default=False)

args = parser.parse_args()
device = 'cpu' if args.gpu == '-1' else torch.device('cuda', 0)
device_ids = list(np.arange(args.num_gpu))
device_ids = [int(device_id) for device_id in device_ids]
args.img_node_th = float(args.img_node_th)
args.obj_node_th = float(args.obj_node_th)


def train():
    DATA_DIR = args.data_dir = os.path.join(project_dir, args.data_dir)
    GRAPH_DIR = args.prebuild_path = os.path.join(project_dir, args.prebuild_path)

    train_data_list = [os.path.join(DATA_DIR, 'train', x) for x in sorted(os.listdir(os.path.join(DATA_DIR, 'train'))) if "dat.gz" in x if os.path.exists(os.path.join(GRAPH_DIR, 'train', x))]
    valid_data_list = [os.path.join(DATA_DIR, 'val', x) for x in sorted(os.listdir(os.path.join(DATA_DIR, 'val'))) if "dat.gz" in x if os.path.exists(os.path.join(GRAPH_DIR, 'val', x))]

    data_list = train_data_list + valid_data_list
    for data_path in tqdm(data_list):
        try:
            joblib.load(data_path)
        except:
            print("Error in loading {}".format(data_path))
            os.remove(data_path)
            continue

if __name__ == '__main__':
    train()
