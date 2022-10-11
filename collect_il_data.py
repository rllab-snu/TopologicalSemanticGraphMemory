import argparse
import json
import glob
import os
import numpy as np
import habitat
import habitat.sims
import habitat.sims.habitat_simulator
import joblib
import torch
from env_utils import *
from configs.default import get_config
from tqdm import tqdm
from habitat import make_dataset
from env_utils.make_env_utils import add_panoramic_camera
from NuriUtils.statics import GIBSON_TINY_TRAIN_SCENE, GIBSON_TINY_TEST_SCENE


os.environ['GLOG_minloglevel'] = "2"
os.environ['MAGNUM_LOG'] = "quiet"
import warnings
warnings.simplefilter("ignore", UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="./configs/TSGM.yaml", help="path to config yaml containing info about experiment")
parser.add_argument('--ep-per-env', type=int, default=200, help='number of episodes per environments')
parser.add_argument('--num-procs', type=int, default=1, help='number of processes to run simultaneously')
parser.add_argument('--num-goals', type=int, default=5, help='number of goals per episodes')
parser.add_argument('--split', type=str, default="train", choices=['train', 'val'], help='data split to use')
parser.add_argument('--data-dir', type=str, default="IL_data/gibson", help='directory to save the collected data')
parser.add_argument('--dataset', default='gibson', type=str)
parser.add_argument("--version", type=str, default="collect", help="name to save")
parser.add_argument('--task', default='imggoalnav', type=str)
parser.add_argument('--use-detector', action='store_true', default=False)
parser.add_argument('--fd', action='store_true', default=False)
parser.add_argument('--num-splits', type=int, default=1, help='number of processes to run simultaneously')
parser.add_argument('--split-idx', default=0, type=int)
parser.add_argument('--project-dir', default='.', type=str)
parser.add_argument('--mode', default='collect', type=str)
args = parser.parse_args()


def make_env_fn(config_env, rank):
    config_env.defrost()
    config_env.SEED = rank * 1121
    config_env.freeze()
    env = eval(config_env.ENV_NAME)(config=config_env)
    env.seed(rank * 1121)
    return env


def data_collect(config, DATA_DIR, space_id, tot_space_num, start_idx, num_episodes):
    num_of_envs = args.num_procs
    configs = []
    gpu_ids = np.zeros(num_of_envs)
    if torch.cuda.device_count() > 1:
        gpu_ids[1:] = 1
    for i in range(num_of_envs):
        proc_config = config.clone()
        proc_config.defrost()
        task_config = proc_config.TASK_CONFIG
        task_config.PROC_ID = i
        task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = (int(gpu_ids[i]))
        proc_config.freeze()
        configs.append(proc_config)

    envs = habitat.VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(
                zip(configs, range(num_of_envs))
            )
        ),
        auto_reset_done=False
    )
    num_episodes = int(num_episodes)

    episode = start_idx
    episode_names = []
    for idx in range(num_episodes):
        space_name = config.TASK_CONFIG.DATASET.CONTENT_SCENES[0]
        episode_name = '%s_%03d' % (space_name, idx)
        episode_names.append(episode_name)

    with tqdm(total=num_episodes) as pbar:
        pbar.update(episode)
        while True:
            observations = envs.reset()
            episodes = envs.current_episodes()

            datas = [{'rgb': [], 'depth': [], 'position': [], 'rotation': [], 'action': [],
                      'target_idx': [], 'target_img': None, 'target_pose': None,  'distance': [],
                      'object': [], 'object_score': [], 'object_category': [], 'object_pose': [],
                      'target_object': None, 'target_object_score': None, 'target_object_pose': None, 'target_object_category': None,
                      } for _ in range(num_of_envs)]
            step = 0
            dones = envs.call(['get_episode_over'] * num_of_envs)
            paused = [False] * num_of_envs
            env_ind_states = np.arange(num_of_envs)
            for i in range(num_of_envs):
                datas[i]['target_object'] = []
                datas[i]['target_object_score'] = []
                datas[i]['target_object_pose'] = []
                datas[i]['target_object_category'] = []
                datas[i]['target_img'] = []
                datas[i]['target_pose'] = []

                for e in range(len(episodes[i].goals)):
                    datas[i]['target_object'].append(observations[i]['target_loc_object'][e])
                    datas[i]['target_object_score'].append(observations[i]['target_loc_object_score'][e])
                    datas[i]['target_object_pose'].append(observations[i]['target_loc_object_pose'][e])
                    datas[i]['target_object_category'].append(observations[i]['target_loc_object_category'][e])
                    datas[i]['target_img'].append(observations[i]['target_goal'][e])
                    datas[i]['target_pose'].append(episodes[i].goals[e].position)

            past_alive_indices = np.where(np.array(paused) == False)
            while (np.array(dones) == 0).any():
                best_actions = np.array(envs.call(['get_best_action'] * num_of_envs))
                curr_goal_indices = envs.call(['get_curr_goal_index'] * num_of_envs)
                alive_indices = np.where(np.array(paused) == False)
                past_obs = observations

                best_actions[np.where(best_actions == None)] = 0
                best_actions[np.where(envs.call(['get_episode_over'] * num_of_envs)) == 1] = 0
                outputs = envs.step(best_actions)
                observations, rewards, dones, infos = [
                    list(x) for x in zip(*outputs)
                ]
                for i, j in enumerate(past_alive_indices[0]):
                    datas[j]['rgb'].append(past_obs[i]['panoramic_rgb'])
                    datas[j]['depth'].append(past_obs[i]['panoramic_depth'])
                    datas[j]['object'].append(past_obs[i]['object'])
                    datas[j]['object_pose'].append(past_obs[i]['object_pose'])
                    datas[j]['object_score'].append(past_obs[i]['object_score'])
                    datas[j]['object_category'].append(past_obs[i]['object_category'])
                    datas[j]['position'].append(past_obs[i]['position'])
                    datas[j]['rotation'].append(past_obs[i]['rotation'])
                    datas[j]['distance'].append(past_obs[i]['distance'])
                    if j in alive_indices[0]:
                        datas[j]['action'].append(best_actions[alive_indices[0].tolist().index(j)])
                        datas[j]['target_idx'].append(curr_goal_indices[alive_indices[0].tolist().index(j)])
                    try:
                        if j in alive_indices[0] and dones[alive_indices[0].tolist().index(j)] == 1:
                            ind = np.where(env_ind_states == j)
                            envs.pause_at(ind[0][0])
                            env_ind_states = np.delete(env_ind_states, ind)
                            paused[j] = True
                            continue
                    except:
                        pass

                step += 1
                past_alive_indices = alive_indices

            envs.resume_all()
            successes = envs.call(['get_success'] * num_of_envs)
            for i in range(num_of_envs):
                success = successes[i]
                if success:
                    joblib.dump(datas[i], os.path.join(DATA_DIR, episode_names[episode] + '.dat.gz'))

                    episode += 1
                    pbar.update(1)
                    pbar.set_description('Total %05d, %s SPACE[%03d/%03d] %03d/%03d data collected' % (len(os.listdir(DATA_DIR)),
                                                                                                       space_name,
                                                                                                       space_id + 1,
                                                                                                       tot_space_num,
                                                                                                       len(glob.glob(os.path.join(DATA_DIR, space_name) + '*')),
                                                                                                       num_episodes))
                    if episode >= num_episodes:
                        break
                if episode >= num_episodes:
                    break
            if episode >= num_episodes:
                break
    envs.close()


def main():
    split = args.split
    DATA_DIR = os.path.join(args.project_dir, args.data_dir)
    if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)
    DATA_DIR = os.path.join(DATA_DIR, split)
    if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)

    config = get_config(args.config, base_task_config_path="./configs/{}_{}.yaml".format(args.task, args.dataset), arguments=vars(args))
    config.defrost()
    config.noisy_actuation = True
    if args.num_procs > 0:
        config.NUM_PROCESSES = args.num_procs
    config.USE_DETECTOR = config.TASK_CONFIG.USE_DETECTOR = args.use_detector
    config.detector_th = config.TASK_CONFIG.detector_th = 0.01
    config.DATASET_NAME = args.dataset
    config.TASK_CONFIG.DATASET.DATASET_NAME = args.dataset
    config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    config.TASK_CONFIG.TASK.MEASUREMENTS = ["GOAL_INDEX"] + config.TASK_CONFIG.TASK.MEASUREMENTS
    config.TASK_CONFIG.TASK.GOAL_INDEX = config.TASK_CONFIG.TASK.SPL.clone()
    config.TASK_CONFIG.TASK.GOAL_INDEX.TYPE = 'GoalIndex'
    config.TASK_CONFIG.TASK.SUCCESS_DISTANCE = float(np.clip(float(config.TASK_CONFIG.TASK.SUCCESS_DISTANCE) - 0.5, 0.0, 1.0))
    config.RL.SUCCESS_DISTANCE = float(np.clip(float(config.RL.SUCCESS_DISTANCE) - 0.5, 0.0, 1.0))
    print(config.TASK_CONFIG.TASK.SUCCESS_DISTANCE)
    print(config.RL.SUCCESS_DISTANCE)
    config.TRAINER_NAME = config.RL_TRAINER_NAME
    config.features.object_category_num = 80
    config.img_node_th = 0.7
    config.TASK_CONFIG.img_node_th = 0.7
    config.TASK_CONFIG.obj_node_th = 0.8
    config.TASK_CONFIG.TRAIN_IL = False
    config.TASK_CONFIG.DATASET.DATASET_NAME = args.dataset
    config.TASK_CONFIG.PROC_ID = 0
    config.IMG_SHAPE = (64, 252) #config.TASK_CONFIG.IMG_SHAPE
    config.CHECKPOINT_FOLDER = os.path.join(args.project_dir, config.CHECKPOINT_FOLDER)
    habitat_api_path = os.path.join(os.path.dirname(habitat.__file__), '../')
    config.ENV_NAME = "MultiImageGoalEnv"
    config.TASK_CONFIG.DATASET.SCENES_DIR = os.path.join(habitat_api_path, config.TASK_CONFIG.DATASET.SCENES_DIR)
    config.TASK_CONFIG.DATASET.DATA_PATH = os.path.join(habitat_api_path, config.TASK_CONFIG.DATASET.DATA_PATH)
    config.TASK_CONFIG.DATASET.SPLIT = split
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = 2000
    config.TASK_CONFIG.ENVIRONMENT.NUM_GOALS = args.num_goals
    config.TASK_CONFIG = add_panoramic_camera(config.TASK_CONFIG, normalize_depth=True)
    config.DIFFICULTY = 'collect'
    config.DATASET_NAME = args.dataset
    config.TASK_CONFIG.DATASET.DATASET_NAME = args.dataset
    config.record = False
    config.render_map = False
    config.noisy_actuation = False

    config.USE_DETECTOR = config.TASK_CONFIG.USE_DETECTOR = True
    if config.USE_DETECTOR:
        print('Detector th: ', config.TASK_CONFIG.detector_th)
    config.freeze()

    if "tiny" in args.dataset:
        if args.split == "train":
            scenes = GIBSON_TINY_TRAIN_SCENE
        elif args.split == "val":
            scenes = GIBSON_TINY_TEST_SCENE
    else:
        dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
        scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)
    print(len(scenes))
    ep_per_env = {}
    # if "gibson" in args.dataset:
    #     data_info = json.load(open("./data/scene_info/gibson/gibson_dset_with_qual.json", "r"))
    #     areas = {}
    #     for scene in scenes:
    #         areas[scene] = data_info[scene]['stats']['area']
    #     sum_areas = np.sum(list(areas.values()))
    #     for scene in scenes:
    #         ep_per_env[scene] = int(np.ceil(20000 * areas[scene]/sum_areas))
    # else:
    for scene in scenes:
        ep_per_env[scene] = args.ep_per_env
    collected_scenes = list(np.unique([pp.split("_")[0] for pp in os.listdir(DATA_DIR)]))
    collected_scenes_orig = collected_scenes.copy()
    scene_dict = {}
    for cs in collected_scenes_orig:
        list_collected = sorted(glob.glob(os.path.join(DATA_DIR, cs) + '*'))
        print("collected", cs, len(list_collected))
        if len(list_collected) < ep_per_env[cs]:
            collected_scenes.remove(cs)
            scene_dict[cs] = len(list_collected)
            # for lc in list_collected:
            #     os.remove(lc)
    print("collected:", collected_scenes)
    scenes = np.sort(scenes)
    if args.split_idx == 0:
        with open(os.path.join(args.project_dir, args.data_dir, f'{args.split}_config.json'), 'w') as f:
            json.dump(config, f)
    scenes = np.array_split(scenes, args.num_splits)[args.split_idx]
    scenes = list(scenes)
    scenes = np.array(scenes)
    print(scenes)
    scene_dict = {}
    for cc in scenes:
        list_collected = sorted(glob.glob(os.path.join(DATA_DIR, cc) + '*'))
        scene_dict[cc] = len(list_collected)
    print(scene_dict)
    for space_id, (space, start_idx) in enumerate(scene_dict.items()):
        if start_idx < ep_per_env[space]:
            print('=' * 50)
            print('SPACE[%03d/%03d] STARTED %s' % (space_id + 1, len(scenes), space))
            config.defrost()
            config.TASK_CONFIG.DATASET.CONTENT_SCENES = [space]
            config.freeze()
            data_collect(config, DATA_DIR, space_id, len(scenes), start_idx, ep_per_env[space])


if __name__ == "__main__":
    main()
