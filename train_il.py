from configs.default import get_config
from model.policy import *
from trainer.il.il_trainer import *
from gym.spaces.dict import Dict as SpaceDict
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
import os, argparse, torch, time, wandb, numpy as np
import torchvision.transforms as transforms
from dataset.habitatdataset import ILDataset
from habitat.core.logging import logger
from torch.utils.data import DataLoader
import datetime
from utils.augmentations import GaussianBlur
project_dir = os.path.dirname(os.path.abspath(__file__))
torch.backends.cudnn.enabled = True

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/TSGM.yaml", help="path to config yaml containing info about experiment")
parser.add_argument("--prebuild-path", type=str, default="data/graph", help="path to prebuild graph")
parser.add_argument("--gpu", type=str, default="0", help="gpus",)
parser.add_argument("--num-gpu", type=int, default=1, help="gpus",)
parser.add_argument("--version", type=str, default="test", help="name to save")
parser.add_argument('--data-dir', default='IL_data', type=str)
parser.add_argument('--project-dir', default='.', type=str)
parser.add_argument('--dataset', default='gibson', type=str)
parser.add_argument('--resume', default='none', type=str)
parser.add_argument('--task', default='imggoalnav', type=str)
parser.add_argument('--num-object', default=10, type=int)
parser.add_argument('--memory-size', default=0, type=int)
parser.add_argument('--max-input-length', default=100, type=int)
parser.add_argument('--multi-target', action='store_true', default=False)
parser.add_argument('--mode', default='train_il', type=str)
parser.add_argument('--policy', default='TSGMPolicy', required=True, type=str)
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
    observation_space = SpaceDict({
        'panoramic_rgb': Box(low=0, high=256, shape=(64, 256, 3), dtype=np.float32),
        'panoramic_depth': Box(low=0, high=256, shape=(64, 256, 1), dtype=np.float32),
        'target_goal': Box(low=0, high=256, shape=(64, 256, 3), dtype=np.float32),
        'step': Box(low=0, high=500, shape=(1,), dtype=np.float32),
        'prev_act': Box(low=0, high=3, shape=(1,), dtype=np.int32),
        'gt_action': Box(low=0, high=3, shape=(1,), dtype=np.int32)
    })

    DATA_DIR = args.data_dir = os.path.join(project_dir, args.data_dir)
    GRAPH_DIR = args.prebuild_path = os.path.join(project_dir, args.prebuild_path)
    config = get_config(args.config, base_task_config_path="./configs/{}_{}.yaml".format(args.task, args.dataset), arguments=vars(args))
    action_space = Discrete(4)
    config.defrost()
    config.POLICY = args.policy
    config.IL.batch_size = config.IL.batch_size * int(args.num_gpu)
    config.NUM_PROCESSES = config.IL.batch_size
    config.TORCH_GPU_ID = args.gpu
    config.scene_data = args.dataset
    config.IL.WRAPPER = "ILWrapper"
    if args.memory_size > 0:
        config.memory.memory_size = args.memory_size
    config.features.object_category_num = 80
    config.memory.num_objects = args.num_object
    config.ENV_NAME = "ImageGoalEnv"
    config.TASK_CONFIG.TRAIN_IL = True
    config.TASK_CONFIG.DATASET.DATASET_NAME = args.dataset
    config.IMG_SHAPE = (64, 252) #config.TASK_CONFIG.IMG_SHAPE
    config.detector_th = config.TASK_CONFIG.detector_th = args.detector_th
    config.TASK_CONFIG.img_node_th = args.img_node_th
    config.TASK_CONFIG.obj_node_th = args.obj_node_th
    config.max_input_length = args.max_input_length
    config.OBJECTGRAPH.SPARSE = False
    config.freeze()

    """
    Print configuration
    """

    print('====================================')
    print('Dataset Name: ', args.dataset)
    print('POLICY : {}'.format(config.POLICY))
    print('Image Graph Threshold: ', config.TASK_CONFIG.img_node_th)
    print('Object Graph Threshold: ', config.TASK_CONFIG.obj_node_th)
    print('====================================')

    policy = eval(config.POLICY)(
        observation_space=observation_space,
        action_space=action_space,
        hidden_size=config.features.hidden_size,
        rnn_type=config.features.rnn_type,
        num_recurrent_layers=config.features.num_recurrent_layers,
        backbone=config.features.backbone,
        goal_sensor_uuid=config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
        normalize_visual_inputs=True,
        cfg=config
    )
    if len(device_ids) > 1:
        policy = nn.DataParallel(policy, device_ids=device_ids).cuda()
    trainer = eval(config.TASK_CONFIG.IL_TRAINER)(config, policy)
    train_data_list = [os.path.join(DATA_DIR, 'train', x) for x in sorted(os.listdir(os.path.join(DATA_DIR, 'train'))) if "dat.gz" in x if os.path.exists(os.path.join(GRAPH_DIR, 'train', x))]
    valid_data_list = [os.path.join(DATA_DIR, 'val', x) for x in sorted(os.listdir(os.path.join(DATA_DIR, 'val'))) if "dat.gz" in x if os.path.exists(os.path.join(GRAPH_DIR, 'val', x))]

    params = {'batch_size': config.IL.batch_size,
              'shuffle': True,
              'num_workers': config.IL.num_workers,
              'pin_memory': True}

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    eval_augmentation = [
        transforms.Resize(config.IMG_SHAPE),
        transforms.ToTensor(),
        normalize
    ]
    augmentation = [
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.Resize(config.IMG_SHAPE),
        transforms.ToTensor(),
        normalize
    ]
    train_dataset = ILDataset(config, train_data_list, transforms.Compose(augmentation))
    valid_dataset = ILDataset(config, valid_data_list,transforms.Compose(eval_augmentation))
    valid_params = params

    valid_dataloader = DataLoader(valid_dataset, **valid_params)
    valid_iter = iter(valid_dataloader)

    version_name = config.saving.name if args.version == 'none' else args.version
    version_name += '_{}'.format(args.dataset)
    version_name += '_{}'.format(args.task)
    curr_hostname = os.uname()[1]
    if args.wandb:
        wandb_run = wandb.init(project="TSGM_{}".format(args.task), config=config, name=version_name + '_{}'.format(curr_hostname), tags=[curr_hostname])

    IMAGE_DIR = os.path.join(project_dir, 'data', 'images', version_name)
    SAVE_DIR = os.path.join(project_dir, 'data', 'checkpoints', version_name)
    LOG_DIR = os.path.join(project_dir, 'data', 'logs', version_name)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    if not args.debug:
        os.makedirs(SAVE_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)

    start_step = 0
    start_epoch = 0
    step_index = 0
    step_values = [20000, 50000, 100000]
    if args.resume != 'none':
        sd = torch.load(args.resume)
        start_epoch, start_step = sd['trained']
        trainer.agent.load_state_dict(sd['state_dict'])
        for step_value in step_values:
            if start_step >= step_value:
                step_index += 1
            else:
                break
        print('load {}, start_ep {}, strat_step {}'.format(args.resume, start_epoch, start_step))
    print_every = config.IL.LOG_INTERVAL
    save_every = config.IL.CHECKPOINT_INTERVAL
    eval_every = config.saving.eval_interval

    start = time.time()
    temp = start
    step = start_step
    lr = config.IL.lr

    trainer.to(device)
    trainer.train()
    for epoch in range(start_epoch, config.IL.max_epoch):
        train_dataloader = DataLoader(train_dataset, **params)
        train_iter = iter(train_dataloader)
        loss_summary_dict = {}
        for iteration, batch in enumerate(train_iter):
            results, loss_dict = trainer(batch)
            for k,v in loss_dict.items():
                if k not in loss_summary_dict.keys():
                    loss_summary_dict[k] = []
                loss_summary_dict[k].append(v)

            if step in step_values:
                step_index += 1
                lr = adjust_learning_rate(trainer.optim, step_index, config.IL.lr_decay, config.IL.lr)

            if step % print_every == 0:
                loss_str = ''
                writer_dict = {}
                for k,v in loss_summary_dict.items():
                    value = np.array(v).mean()
                    loss_str += '%s: %.3f '%(k,value)
                    writer_dict[k] = value

                logger.info("time = %.0fh %.0fm, epo %d, step %d, lr: %.5f, %ds per %d iters || " % ((time.time() - start) // 3600, ((time.time() - start) / 60) % 60, epoch + 1,
                                                                                                     step + 1, lr, time.time() - temp, print_every) + loss_str)

                temp = time.time()
                if args.wandb:
                    wandb_run.log(
                        {
                            'act_loss': loss_summary_dict['act_loss'][0],
                            "havebeen_loss":  loss_summary_dict['have_been'][0],
                            "progress_loss":  loss_summary_dict['progress'][0],
                            "haveseen_loss":  loss_summary_dict['have_seen'][0],
                            "target_loss":  loss_summary_dict['is_target'][0],
                            "lr": lr,
                        },
                        step=step
                    )
                loss_summary_dict = {}

            if step % save_every == 0 and not args.debug:
                trainer.save(file_name=os.path.join(SAVE_DIR, 'epoch%04diter%05d.pt' % (epoch, step)),epoch=epoch, step=step)
                logger.info("Saved checkpoint to '{}'".format(os.path.join(SAVE_DIR, 'epoch%04diter%05d.pt' % (epoch, step))))

            del results, batch, loss_dict
            if step % eval_every == 0:# and step > 0:
                trainer.eval()
                eval_start = time.time()
                with torch.no_grad():
                    val_loss_summary_dict = {}
                    for j in range(100):
                        try:
                            batch = next(valid_iter)
                        except:
                            valid_dataloader = DataLoader(valid_dataset, **valid_params)
                            valid_iter = iter(valid_dataloader)
                            batch = next(valid_iter)
                        results, loss_dict = trainer(batch, train=False)
                        # if j % 100 == 0:
                        #     trainer.visualize(results, os.path.join(IMAGE_DIR, 'validate_{}_{}_{}'.format(results['scene'], step, j)))
                        for k, v in loss_dict.items():
                            if k not in val_loss_summary_dict.keys():
                                val_loss_summary_dict[k] = []
                            val_loss_summary_dict[k].append(v)

                    loss_str = ''
                    writer_dict = {}
                    for k, v in val_loss_summary_dict.items():
                        value = np.array(v).mean()
                        loss_str += '%s: %.3f ' %(k, value)
                        writer_dict[k] = value
                    logger.info("validation time = %.0fh %.0fm, epo %d, step %d, lr: %.5f, %ds per %d iters || loss : " % (
                        (time.time() - start) // 3600, ((time.time() - start) / 60) % 60, epoch + 1, step + 1, lr, time.time() - eval_start, print_every) + loss_str)
                    temp = time.time()
                    loss_summary_dict = {}
                del batch, results, loss_dict
                trainer.train()
            step += 1
    print('===> end training')


def adjust_learning_rate(optimizer, step_index, lr_decay, lr):
    lr = lr * (lr_decay ** step_index)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    train()
