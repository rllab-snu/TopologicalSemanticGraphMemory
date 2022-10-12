import sys
import argparse
import imageio
import gzip
from copy import deepcopy
import datetime
import torch
from env_utils.make_env_utils import add_panoramic_camera, add_orthographic_camera, add_equirect_camera
from NuriUtils.utils import get_remain_time
import habitat
from habitat import make_dataset
from env_utils import *
from configs.default import get_config, CN
import time
import cv2
import os
import json
from runner import *

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--ep-per-env", type=int, default=1000)
parser.add_argument("--config", type=str, default="./configs/TSGM.yaml", help="path to config yaml containing info about experiment")
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--version", type=str, required=True)
parser.add_argument("--diff", choices=['random', 'easy', 'medium', 'hard'], default='')
parser.add_argument("--split", choices=['val', 'train', 'val_mini','test'], default='val')
parser.add_argument('--eval-ckpt', type=str, required=True)
parser.add_argument('--record', choices=['0','1','2','3'], default='0') # 0: no record 1: env.render 2: pose + action numerical traj 3: features
parser.add_argument('--graph-th', type=float, default=0.75) # s_th
parser.add_argument('--project-dir', default='.', type=str)
parser.add_argument('--dataset', default='gibson' , type=str)
parser.add_argument('--task', default='imggoalnav', type=str)
parser.add_argument('--record-dir', type=str, default='data/video_dir')
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--use-detector', action='store_true', default=False)
parser.add_argument('--num-object', default=10, type=int)
parser.add_argument('--detector-th', default=0.01, type=float)
parser.add_argument('--multi-target', action='store_true', default=False)
parser.add_argument('--wandb', action='store_true', default=False)
parser.add_argument('--mode', default='eval', type=str)
parser.add_argument('--episode_name', default='VGM', type=str) #[VGM, NRNS_curved, NRNS_straight]
parser.add_argument('--policy', default='TSGMPolicy', required=True, type=str)
parser.add_argument('--coverage', action='store_true', default=False)
parser.add_argument('--fd', action='store_true', default=False, help="use finetuned detector")
parser.add_argument('--obj-score-th', default=0.1, type=float)
parser.add_argument('--img-node-th', type=str, default='0.75')
parser.add_argument('--obj-node-th', type=str, default='0.8')
parser.add_argument('--debug', action='store_true', default=False)

# Noise settings
parser.add_argument('--depth_noise', action='store_true', default=False)
parser.add_argument('--actua_noise', action='store_true', default=False)
parser.add_argument('--sensor_noise', action='store_true', default=False)
parser.add_argument('--depth-noise-level', default=4.0, type=float)
parser.add_argument('--actua-noise-level', default=4.0, type=float)
parser.add_argument('--sensor-noise-level', default=4.0, type=float)


args = parser.parse_args()
args.record = int(args.record)
args.graph_th = float(args.graph_th)
args.img_node_th = float(args.img_node_th)
args.obj_node_th = float(args.obj_node_th)
os.environ['GLOG_minloglevel'] = "3"
os.environ['MAGNUM_LOG'] = "quiet"
os.environ['HABITAT_SIM_LOG'] = "quiet"
import numpy as np
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.gpu != 'cpu':
    torch.cuda.manual_seed(args.seed)
torch.set_num_threads(5)
torch.backends.cudnn.enabled = True

device = 'cpu' if args.gpu == '-1' else 'cuda:{}'.format(args.gpu)

def eval_config(args):
    config = get_config(args.config, base_task_config_path="./configs/{}_{}.yaml".format(args.task, args.dataset), arguments=vars(args))
    config.defrost()
    config.POLICY = args.policy
    config.use_depth = True
    config.USE_DETECTOR = True
    config.DIFFICULTY = args.diff
    config.scene_data = args.dataset
    habitat_api_path = os.path.join(os.path.dirname(habitat.__file__), '../')
    print(args.config)
    config.TASK_CONFIG = add_panoramic_camera(config.TASK_CONFIG, normalize_depth=True)
    config.TASK_CONFIG = add_orthographic_camera(config.TASK_CONFIG)
    config.TASK_CONFIG = add_equirect_camera(config.TASK_CONFIG)
    config.TASK_CONFIG.DATASET.SPLIT = args.split #if 'gibson' in config.TASK_CONFIG.DATASET.DATA_PATH else 'val'
    config.TASK_CONFIG.DATASET.SCENES_DIR = os.path.join(habitat_api_path, config.TASK_CONFIG.DATASET.SCENES_DIR)
    config.TASK_CONFIG.DATASET.DATA_PATH = os.path.join(habitat_api_path, config.TASK_CONFIG.DATASET.DATA_PATH)
    config.TASK_CONFIG.DATASET.DATASET_NAME = args.dataset
    config.TASK_CONFIG.TASK.MEASUREMENTS = ["GOAL_INDEX"] + config.TASK_CONFIG.TASK.MEASUREMENTS
    config.TASK_CONFIG.TASK.GOAL_INDEX = config.TASK_CONFIG.TASK.SPL.clone()
    config.TASK_CONFIG.TASK.GOAL_INDEX.TYPE = 'GoalIndex'
    if 'COLLISIONS' not in config.TASK_CONFIG.TASK.MEASUREMENTS:
        config.TASK_CONFIG.TASK.MEASUREMENTS += ['COLLISIONS']
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    if config.TASK_CONFIG.DATASET.CONTENT_SCENES == ['*']:
        print("*"*100)
        print(config.TASK_CONFIG.DATASET)
        print("*"*100)
        scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)
    else:
        scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES

    config.TASK_CONFIG.DATASET.CONTENT_SCENES = scenes
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = args.ep_per_env

    config.ACTION_DIM = 4
    config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS= ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]

    config.TASK_CONFIG.PROC_ID = 0
    config.freeze()
    return config

def load(ckpt):
    if ckpt != 'none':
        sd = torch.load(ckpt,map_location=torch.device('cpu'))
        state_dict = sd['state_dict']
        new_state_dict = {}
        for key in state_dict.keys():
            if 'actor_critic' in key:
                new_state_dict[key[len('actor_critic.'):]] = state_dict[key]
            else:
                new_state_dict[key] = state_dict[key]
        if 'config' in sd.keys():
            return (new_state_dict, sd['config'])
        return (new_state_dict,None)
    else:
        return (None, None)

def evaluate(eval_config, state_dict, ckpt_config):
    if ckpt_config is not None:
        task_config = eval_config.TASK_CONFIG
        ckpt_config.defrost()
        task_config.defrost()
        ckpt_config.TASK_CONFIG = task_config
        ckpt_config.runner = eval_config.runner
        ckpt_config.iter_num = eval_config.iter_num
        ckpt_config.AGENT_TASK = 'search'
        ckpt_config.USE_DETECTOR = ckpt_config.TASK_CONFIG.USE_DETECTOR =  eval_config.USE_DETECTOR
        ckpt_config.detector_th = ckpt_config.TASK_CONFIG.detector_th = args.detector_th
        ckpt_config.use_depth = ckpt_config.TASK_CONFIG.use_depth = eval_config.use_depth
        try:
            ckpt_config['ARGS']['project_dir'] = args.project_dir
        except:
            ckpt_config['ARGS'] = vars(args)
        ckpt_config.POLICY = eval_config.POLICY
        ckpt_config.DIFFICULTY = eval_config.DIFFICULTY
        ckpt_config.ACTION_DIM = eval_config.ACTION_DIM
        ckpt_config.memory = eval_config.memory
        ckpt_config.scene_data = eval_config.scene_data
        ckpt_config.WRAPPER = eval_config.WRAPPER
        ckpt_config.REWARD_METHOD = eval_config.REWARD_METHOD
        ckpt_config.ENV_NAME = eval_config.ENV_NAME

        for k, v in eval_config.items():
            if k not in ckpt_config:
                ckpt_config.update({k:v})
            if isinstance(v, CN):
                for kk, vv in v.items():
                    if kk not in ckpt_config[k]:
                        ckpt_config[k].update({kk: vv})
        ckpt_config.freeze()
        eval_config = ckpt_config
    eval_config.defrost()
    eval_config.img_node_th = args.img_node_th
    eval_config.TASK_CONFIG.img_node_th = args.img_node_th
    eval_config.TASK_CONFIG.obj_node_th = args.obj_node_th
    eval_config.record = args.record > 0
    eval_config.render_map = args.record > 0 or args.render or 'hand' in args.config
    eval_config.noisy_actuation = True
    eval_config.memory.num_objects = args.num_object
    eval_config.OBJECTGRAPH.SPARSE = False
    eval_config.features.object_category_num = 80
    eval_config.gpu = args.gpu.split(',')
    if len(args.gpu) > 1:
        eval_config.TORCH_GPU_ID = int(args.gpu[0])
        eval_config.SIMULATOR_GPU_ID = int(args.gpu[1])
        eval_config.TASK_CONFIG.DETECTOR_GPU_ID = int(args.gpu[1])
    else:
        eval_config.TORCH_GPU_ID = int(args.gpu[0])
        eval_config.SIMULATOR_GPU_ID = int(args.gpu[0])
        eval_config.TASK_CONFIG.DETECTOR_GPU_ID = int(args.gpu[0])
    eval_config.TASK_CONFIG['ARGS'] = vars(args)
    eval_config['ARGS'] = vars(args)
    eval_config.freeze()
    runner = eval(eval_config.runner)(args, eval_config, return_features=True)

    print(eval_config.memory)
    print('====================================')
    print('Version Name: ', args.version)
    print('Dataset Name: ', args.dataset)
    print('Evaluating: ', eval_config.iter_num)
    print('Runner : ', eval_config.runner)
    print('Policy : ', eval_config.POLICY)
    print('Difficulty: ', eval_config.DIFFICULTY)
    print('Use Detector: ', eval_config.USE_DETECTOR)
    print('Detector threshold: ', eval_config.detector_th)
    print('Stop action: ', 'True' if eval_config.ACTION_DIM==4 else 'False')
    print('====================================')
    curr_hostname = os.uname()[1]
    version_name = eval_config.saving.name if args.version == 'none' else args.version
    version_name += '_{}'.format(args.dataset)
    version_name += '_{}'.format(args.task)
    if args.wandb:
        import wandb
        from wandb import AlertLevel
        wandb.init(project="(eval)TSGM_{}".format(args.task), config=eval_config, name=version_name + '_{}'.format(curr_hostname), tags=[curr_hostname])
    
    runner.eval()
    if torch.cuda.device_count() > 0:
        runner.to(device)

    try:
        runner.load(state_dict)
        print('Loaded model from checkpoint')
    except:
        agent_dict = runner.agent.state_dict()
        new_sd = {k: v for k, v in state_dict.items() if k in agent_dict.keys() and (v.shape == agent_dict[k].shape)}
        agent_dict.update(new_sd)
        runner.load(agent_dict)
        print('Loaded partial model')

    eval_config.defrost()

    tot_episodes = 0
    if args.episode_name == "VGM":
        for scene in eval_config.TASK_CONFIG.DATASET.CONTENT_SCENES:
            json_file = os.path.join(args.project_dir, 'data/episodes/{}/{}/{}_{}.json'.format(args.episode_name, args.dataset.split("_")[0], scene, args.diff))
            with open(json_file, 'r') as f:
                episodes = json.load(f)
            tot_episodes += len(episodes)
    elif args.episode_name.split("_")[0] == "NRNS":
        json_file = os.path.join(args.project_dir, 'data/episodes/{}/{}/{}/test_{}.json.gz'.format(args.episode_name.split("_")[0], args.dataset.split("_")[0], args.episode_name.split("_")[1], args.diff))
        with gzip.open(json_file, "r") as fin:
            episodes = json.loads(fin.read().decode("utf-8"))['episodes']
        tot_episodes = len(episodes)
    elif args.episode_name == "MARL":
        for scene in eval_config.TASK_CONFIG.DATASET.CONTENT_SCENES:
            json_file = os.path.join(args.project_dir, 'data/episodes/{}/{}/{}.json.gz'.format(args.episode_name, args.dataset.split("_")[0], scene))
            with gzip.open(json_file, "r") as fin:
                episodes = json.loads(fin.read().decode("utf-8"))
            episodes = [episode for episode in episodes if episode['info']['difficulty'] == args.diff]
            tot_episodes += len(episodes)
    # 573/214->505/205
    eval_config.freeze()
    env = eval(eval_config.ENV_NAME)(eval_config)
    env.habitat_env._sim.seed(args.seed)
    if runner.need_env_wrapper:
        env = runner.wrap_env(env, eval_config)

    result = {}
    result['config'] = eval_config
    result['args'] = args
    result['version'] = str(args.version)
    datetime_now = str(datetime.datetime.today()).split(".")[0].replace(" ","_")
    result['start_time'] = datetime_now
    result['noisy_action'] = bool(env.noise)
    scene_dict = {}
    render_check = False
    start_time = time.time()
    with torch.no_grad():
        ep_list = []
        total_success, total_spl, total_dtg, total_softspl, total_localize_success, total_node_dists, total_success_timesteps = [], [], [], [], [], [], []
        for episode_id in range(tot_episodes):
            obs = env.reset()
            if render_check == False:
                if obs['panoramic_rgb'].sum() == 0 :
                    print('NO RENDERING!!!!!!!!!!!!!!!!!! YOU SHOULD CHECK YOUT DISPLAY SETTING')
                else:
                    render_check=True
            runner.reset()
            scene_name = env.current_episode.scene_id.split('/')[-1][:-4]
            if scene_name not in scene_dict.keys():
                scene_dict[scene_name] = {'success': [], 'spl': [], 'dtg': [], 'softspl': []}
            done = True
            reward = None
            info = None
            if args.record > 0:
                img = env.render('rgb')
                imgs = [img]
            step = 0
            records = []
            record_graphs = []
            record_maps = []
            record_features = []
            record_objects = []
            localize_success = []
            while True:
                action = runner.step(obs, reward, done, info, env)
                if action == 100: # handcrafted navigation mode
                    paths = env.get_navigation_path(obs)
                    obs['planned_path'] = paths[0]
                    action = runner.step(obs, reward, done, info, env)
                if 'curr_attn' in runner.features.keys():
                    dist_nodes = []
                    for iii in range(len(env.imggraph.node_position_list)):
                        dist_nodes.append(env.env._env.sim.geodesic_distance(env.imggraph.node_position_list[iii], env.current_position))
                    if np.argmin(dist_nodes) == runner.features['curr_attn'].squeeze().argmax().item():
                        localize_success.append(1)
                    else:
                        localize_success.append(0)
                if args.record > 1:
                    records.append([env.get_agent_state().position, env.get_agent_state().rotation.components, action])
                    if hasattr(env.mapper, 'node_list'):
                        num_img_node = env.imggraph.num_node()
                        num_obj_node = env.objgraph.num_node()
                        img_memory_dict = {
                            'img_memory_feat': env.imggraph.graph_memory[:num_img_node].copy(),
                            'img_memory_pose': np.stack(env.imggraph.node_position_list).copy(),
                            'img_memory_mask': env.imggraph.graph_mask[:num_img_node].copy(),
                            'img_memory_A': env.imggraph.A[:num_img_node, :num_img_node].copy(),
                            'img_memory_idx': env.imggraph.last_localized_node_idx,
                            'img_memory_time': env.imggraph.graph_time[:num_img_node].copy()
                        }
                        obj_memory_dict = {
                            'obj_memory_feat': env.objgraph.graph_memory[:num_obj_node].copy(),
                            'obj_memory_pose': np.stack(env.objgraph.node_position_list[0]),
                            'obj_memory_score': env.objgraph.graph_score[:num_obj_node].copy(),
                            'obj_memory_category': env.objgraph.graph_category[:num_obj_node].copy(),
                            'obj_memory_mask': env.objgraph.graph_mask[:num_obj_node].copy(),
                            'obj_memory_A_OV': env.objgraph.A_OV[:num_obj_node, :num_img_node].copy(),
                            'obj_memory_time': env.objgraph.graph_time[:num_obj_node].copy()
                        }
                        img_memory_dict.update(obj_memory_dict)
                        record_graphs.append(img_memory_dict)
                        record_objects.append({
                            "object": deepcopy(obs['object'][0][:, 1:].cpu().detach().numpy()),
                            "object_score": deepcopy(obs['object_score'][0].cpu().detach().numpy()),
                            "object_category": deepcopy(obs['object_category'][0].cpu().detach().numpy()),
                            "object_pose": deepcopy(obs['object_pose'][0].cpu().detach().numpy()),
                        })
                    if info is not None:
                        record_maps.append({'agent_angle': deepcopy(info['ortho_map']['agent_rot']),
                                            'agent_loc': deepcopy(info['ortho_map']['agent_loc']),
                                            })
                    else:
                        lower_bound, upper_bound = env.habitat_env._sim.pathfinder.get_bounds()
                        record_maps.append({
                            'ortho_map': deepcopy(env.ortho_rgb),
                            'P': deepcopy(np.array(env.P)),
                            'target_loc': np.array(env.habitat_env._current_episode.goals[0].position),
                            'lower_bound': deepcopy(lower_bound),
                            'upper_bound': deepcopy(upper_bound),
                            'WIDTH': env.habitat_env._config.SIMULATOR.ORTHO_RGB_SENSOR.WIDTH,
                            'HEIGHT': env.habitat_env._config.SIMULATOR.ORTHO_RGB_SENSOR.HEIGHT
                        })
                if args.record > 2:
                    record_features.append(runner.features)

                obs, reward, done, info = env.step(action)
                step += 1
                if args.record > 2:
                    img = env.render(mode='rgb', attns=runner.features)
                    imgs.append(img)
                elif args.record > 0:
                    img = env.render('rgb')
                    imgs.append(img)
                if args.render:
                    env.render('human')
                if done: break
            spl = info['spl']
            if np.isnan(spl):
                spl = 0.0
                print('spl nan!', env.habitat_env._sim.geodesic_distance(env.current_episode.start_position, env.current_episode.goals[0].position))
            if np.isinf(spl):
                spl = 0.0
            scene_dict[scene_name]['success'].append(info['success'])
            scene_dict[scene_name]['spl'].append(spl)
            scene_dict[scene_name]['dtg'].append(info['distance_to_goal'])
            scene_dict[scene_name]['softspl'].append(info['softspl'])
            total_success.append(info['success'])
            total_spl.append(spl)
            total_dtg.append(info['distance_to_goal'])
            total_softspl.append(info['softspl'])
            total_localize_success.append(np.array(localize_success).mean())
            if info['success']:
                total_success_timesteps.append(step)
            #total_node_dists.append(np.array(node_dists).mean())
            ep_list.append({'house': scene_name,
                            'ep_id': env.current_episode.episode_id,
                            'start_pose': [list(env.current_episode.start_position), list(env.current_episode.start_rotation)],
                            'target_pose': [env.current_episode.goals[0].position , env.current_episode.goals[0].rotation],
                            'total_step': step,
                            'collision': info['collisions']['count'] if isinstance(info['collisions'], dict) else info['collisions'],
                            'success': info['success'],
                            'spl': spl,
                            'distance_to_goal': info['distance_to_goal'],
                            'target_distance': env.habitat_env._sim.geodesic_distance(env.habitat_env.current_episode.goals[0].position,env.current_episode.start_position),
                           'localize_success': localize_success})
            if args.record > 0:
                video_name = os.path.join(VIDEO_DIR,'%04d_%s_success=%.1f_spl=%.1f.mp4'%(episode_id, scene_name, info['success'], spl))
                with imageio.get_writer(video_name, fps=30) as writer:
                    im_shape = imgs[-1].shape
                    for im in imgs:
                        if (im.shape[0] != im_shape[0]) or (im.shape[1] != im_shape[1]):
                            im = cv2.resize(im, (im_shape[1], im_shape[0]))
                        writer.append_data(im.astype(np.uint8))
                    writer.close()
                if args.record > 1:
                    file_name = os.path.join(OTHER_DIR, '%04d_%s_data_success=%.1f_spl=%.1f.dat.gz' % (episode_id, scene_name, info['success'], spl))
                    data = {'position': records, 'graph': record_graphs, 'map': record_maps, 'episode': ep_list[-1]}
                    joblib.dump(data, file_name)
                    del data
                if args.record > 2:
                    file_name = os.path.join(OTHER_DIR, '%04d_%s_features_success=%.1f_spl=%.1f.dat.gz' % (episode_id, scene_name, info['success'], spl))
                    joblib.dump(record_features, file_name)
                    del record_features
            remain_time = get_remain_time((time.time() - start_time) / (episode_id+1), (tot_episodes - episode_id))
            print(remain_time + ' [%04d/%04d] %s success %.4f, spl %.4f, softspl %.4f, dtg %.4f, localize %.4f, total success %.4f, spl %.4f, softspl %.4f, dtg %.4f, localize %.4f, success time step %.4f'
                  %(episode_id,
                  tot_episodes,
                  scene_name,
                  np.array(scene_dict[scene_name]['success']).mean(),
                  np.array(scene_dict[scene_name]['spl']).mean(),
                  np.array(scene_dict[scene_name]['softspl']).mean(),
                  np.array(scene_dict[scene_name]['dtg']).mean(),
                  np.array(localize_success).mean(),
                  np.array(total_success).mean(),
                  np.array(total_spl).mean(),
                  np.array(total_softspl).mean(),
                  np.array(total_dtg).mean(),
                  np.array(total_localize_success).mean(),
                  np.array(total_success_timesteps).mean()))
    result['detailed_info'] = ep_list
    result['each_house_result'] = {}
    success = []
    spl = []
    dtg = []
    softspl = []
    for scene_name in scene_dict.keys():
        mean_success = np.array(scene_dict[scene_name]['success']).mean()
        mean_spl = np.array(scene_dict[scene_name]['spl']).mean()
        result['each_house_result'][scene_name] = {'success': mean_success, 'spl': mean_spl}
        print('SCENE %s: success %.4f, spl %.4f'%(scene_name, mean_success,mean_spl))
        success.extend(scene_dict[scene_name]['success'])
        spl.extend(scene_dict[scene_name]['spl'])
        softspl.extend(scene_dict[scene_name]['softspl'])
        dtg.extend(scene_dict[scene_name]['dtg'])
    result['total_success'] = np.array(success).mean()
    result['total_spl'] = np.array(spl).mean()
    result['total_softspl'] = np.array(softspl).mean()
    result['total_dtg'] = np.array(dtg).mean()
    result['total_timesteps'] = np.array(total_success_timesteps)
    result['iter'] = str(eval_config.iter_num)
    print('================================================')
    print('total success : %.4f'%(np.array(success).mean()))
    print('total spl : %.4f'%(np.array(spl).mean()))
    print('total softspl : %.4f'%(np.array(softspl).mean()))
    print('total dtg : %.4f'%(np.array(dtg).mean()))
    print('total timesteps : %.3f'%(np.array(total_success_timesteps).mean()))
    env.close()
    if args.wandb:
        wandb.alert(
            title="Performance",
            text="Success %3f SPL %3f Soft-SPL %3f DTG %2f on model %s iter %s"
                 %(np.array(success).mean(), np.array(spl).mean(), np.array(softspl).mean(), np.array(dtg).mean(), args.version, result['iter']),
            level=AlertLevel.INFO
        )
    return result


if __name__=='__main__':
    import joblib
    import glob
    cfg = eval_config(args)

    curr_hostname = os.uname()[1]
    # eval_data_name = os.path.join(args.project_dir, 'results', 'eval_result_{}.dat.gz'.format(curr_hostname))
    args.eval_ckpt = os.path.join(args.project_dir, args.eval_ckpt)
    # Load checked ckpts
    checked_ckpt = []
    # if os.path.isfile(eval_data_name):
    loaded_dict = False
    cnt = 0
    cnt_name = 0
    while not loaded_dict:
        cnt += 1
        try:
            eval_data_name = os.path.join(args.project_dir, 'results', 'eval_result_{}_v{}.dat.gz'.format(curr_hostname, cnt_name))
            if os.path.isfile(eval_data_name):
                try:
                    result_dict = joblib.load(eval_data_name)
                    loaded_dict = True
                except:
                    cnt_name += 1
        except:
            pass
        if cnt > 10:
            result_dict = {}
            break
    if args.version in result_dict:
        result_dict = result_dict[args.version]
        for i in result_dict.keys():
            if "/".join(i.split("/")[:-1]) == args.eval_ckpt:
                checked_ckpt.append(i.split(".pt")[0]+ ".pt")
            elif "/".join(i.split("/")[:-1]).replace(args.project_dir, ".") == args.eval_ckpt:
                checked_ckpt.append(i.split(".pt")[0]+ ".pt")
        print("The code has been evaluated at: ", checked_ckpt)

    while True:
        try:
            if os.path.isfile(args.eval_ckpt):
                ckpts = [args.eval_ckpt]
            elif os.path.isdir(args.eval_ckpt):
                # print('eval_ckpt ', args.eval_ckpt, ' is directory')
                ckpts = [os.path.join(args.eval_ckpt, x) for x in sorted(os.listdir(args.eval_ckpt))]
                ckpts.reverse()
            elif os.path.exists(args.eval_ckpt):
                ckpts = args.eval_ckpt.split(",")
            else:
                ckpts = [x for x in sorted(glob.glob(args.eval_ckpt + '*'))]
                ckpts.reverse()
            last_ckpt = ckpts[0]
        except:
            time.sleep(1000)
            continue

        iter_num = last_ckpt.split("/")[-1]
        if last_ckpt not in checked_ckpt:
            ckpt_dir = last_ckpt
            print('start evaluate {} '.format(ckpt_dir))
            while True:
                try:
                    state_dict, ckpt_config = load(ckpt_dir)
                    break
                except:
                    continue

            if args.record > 0:
                if not os.path.exists(os.path.join(args.project_dir, args.record_dir, args.version)):
                    os.mkdir(os.path.join(args.project_dir, args.record_dir, args.version))
                VIDEO_DIR = os.path.join(args.project_dir, args.record_dir, args.version + '_video_' + ckpt_dir.split('/')[-1] + '_' + str(time.ctime()))
                if not os.path.exists(VIDEO_DIR): os.mkdir(VIDEO_DIR)
                if args.record > 1:
                    OTHER_DIR = os.path.join(args.project_dir, args.record_dir, args.version + '_other_' + ckpt_dir.split('/')[-1] + '_' + str(time.ctime()))
                    if not os.path.exists(OTHER_DIR): os.mkdir(OTHER_DIR)

            print('='*30, iter_num, '='*30)
            cfg.defrost()
            cfg.iter_num = iter_num
            cfg.freeze()
            result = evaluate(cfg, state_dict, ckpt_config)
            datetime_now = str(datetime.datetime.today()).split(".")[0].replace(" ","_")
            if os.path.exists(eval_data_name):
                data = joblib.load(eval_data_name)
                if args.version in data.keys():
                    data[args.version].update({ckpt_dir + '_{}'.format(datetime_now): result})
                else:
                    data.update({args.version: {ckpt_dir + '_{}'.format(datetime_now): result}})
            else:
                data = {args.version: {ckpt_dir + '_{}'.format(datetime_now): result}}
            # joblib.dump(data, eval_data_name)
            checked_ckpt.append(ckpt_dir)
