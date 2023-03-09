import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import math, os

import math
import sys
from typing import Dict, List
import os
import json

import numpy as np
import gym

import torch
import random

import argparse
import copy

from torch.utils.tensorboard import SummaryWriter

import platform
import sys
sys_platform = platform.platform().lower()
if "windows" in sys_platform:
    sys.path.append('D://project-ant/TPCAP/IEEE/Autonomous-Parking-Narrow-Space/')
elif "linux-4.15.0" in sys_platform:
    sys.path.append('/mnt/disk_ant/yuanzheng/Autonomous-Parking-Narrow-Space/')
elif "linux-5.4.0" in sys_platform:
    sys.path.append('//root/autodl-tmp/ant/Autonomous-Parking-Narrow-Space/')

from rl.sparse_rewards.prioritized.agent import Agent
from rl.sparse_rewards.prioritized.test import test
from rl.utils.savePath import show, saveCsv
from rl.utils.dealData import *
from Vehicle import OBCAPath, Vehicle, Path
from pyobca.search import VehicleConfig
from quadraticOBCA import quadraticPath

from case import Case


# fix random seed
def same_seeds(seed):
    print("===========Set the random seed: {}================".format(seed))
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = True  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构


same_seeds(42)


def get_args():
    args = argparse.ArgumentParser("The args of TPCAP RL")
    args.add_argument('--alg', type=str, default="DDPG", help="The algorithm of model")
    args.add_argument('--random_choice', action="store_true", default=False, help="If random choice the path num?")
    args.add_argument('--exp_name', default="test_fed", type=str, help="The name of exp")
    args.add_argument('--maxT', type=int, default=10, help="The total time of simulation.")
    args.add_argument('--dis_factor', type=int, default=5, help="The factor of dis loss")
    args.add_argument('--delta_factor', type=int, default=10, help="The factor of delta loss")
    args.add_argument('--obs_factor', type=float, default=10, help="The factor of obstacle avoidance")
    args.add_argument('--her', action="store_true", default=False, help="If use the her to augment data")
    args.add_argument('--evaluate_loop', type=int, default=20, help="The frequency of evaluate")
    args.add_argument('--case', type=int, default=7, help="The special case for training")
    args.add_argument('--warm_start', action="store_true", default=False, help="Generate the warm data to train")
    args.add_argument('--action_c', action="store_true", default=False, help="If use the continues action?")
    args.add_argument('--large_interval', action="store_true", default=False, help="Use the large control interval")
    args.add_argument('--reward_back', action="store_true", default=False, help="If use the reward back when using warm start")
    args.add_argument('--cuda', action="store_true", help='run on CUDA (default: False)')
    args.add_argument('--relative', action="store_true", help="Use the relatively goal position")
    args.add_argument('--obca', action="store_true", default=False, help="Use the obca to deal data after test")

    # ===================== Federated Learning ======================
    args.add_argument('--agent_num', type=int, default=4, help='The number of federated agent')
    args.add_argument('--avg_loop', type=int, default=10, help='The gap of federated')
    args.add_argument('--checkpoint_interval', type=int, default=0, help='The interval of save agent model(default:Never)')
    args.add_argument('--shared', action="store_true", default=False, help="Use the shared model to train.")


    # ===================== DDPG Bottleneck===========================
    args.add_argument('--ddpg_hidden', type=int, default=256, help="The hidden dim of DDPG IB")
    args.add_argument('--ddpg_std_init', type=float, default=0.3, metavar='G',
                      help='action std init (default: 0.6)')

    # =====================Soft Actor Critic =========================
    args.add_argument('--sac_policy', default="Gaussian",
                      help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    args.add_argument('--sac_gamma', type=float, default=0.99, metavar='G',
                      help='discount factor for reward (default: 0.99)')
    args.add_argument('--sac_tau', type=float, default=0.005, metavar='G',
                      help='target smoothing coefficient(τ) (default: 0.005)')
    args.add_argument('--sac_lr', type=float, default=0.0003, metavar='G',
                      help='learning rate (default: 0.0003)')
    args.add_argument('--sac_alpha', type=float, default=0.2, metavar='G',
                      help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    args.add_argument('--sac_batch_size', type=int, default=256, metavar='N',
                      help='batch size (default: 256)')
    args.add_argument('--sac_automatic_entropy_tuning', type=bool, default=False, metavar='G',
                      help='Automaically adjust α (default: False)')
    args.add_argument('--sac_target_update_interval', type=int, default=1, metavar='N',
                      help='Value target update per no. of updates per step (default: 1)')
    args.add_argument('--sac_hidden_size', type=int, default=256, metavar='N', help='hidden size (default: 256)')

    # ======================     PPO    =================================
    args.add_argument('--ppo_std_decay_rate', type=float, default=0.05, metavar='G',
                      help='linearly decay action_std (action_std = action_std - action_std_decay_rate)')
    args.add_argument('--ppo_min_action_std', type=float, default=0.00, metavar='G',
                      help='minimum action_std (stop decay after action_std <= min_action_std)')
    args.add_argument('--ppo_std_decay_freq', type=int, default=10000,
                      help='action_std decay frequency (in num timesteps)')

    return args.parse_args()


def warm_start(args, agent, tb=None, path_num=7, env=None, saveFigPath=None):
    print("================== Warm Start ====================")
    inputfiledir = "../../../Result/case-{}".format(args.case)
    for j in range(4):
        inputfile = os.path.join(inputfiledir, "data_{}".format(j))
        path_t = np.load(os.path.join(inputfile, "array_t.npy"))
        path_x = np.load(os.path.join(inputfile, "array_x.npy"))
        path_y = np.load(os.path.join(inputfile, "array_y.npy"))
        path_v = np.load(os.path.join(inputfile, "array_v.npy"))
        path_a = np.load(os.path.join(inputfile, "array_a.npy"))
        path_yaw = np.load(os.path.join(inputfile, "array_yaw.npy"))
        path_steer = np.load(os.path.join(inputfile, "array_steer.npy"))
        path_steer_rate = np.load(os.path.join(inputfile, "array_steer_rate.npy"))
        new_path_t, new_path_x, new_path_y, new_path_v, new_path_a, new_path_yaw, new_path_steer, new_path_steer_rate = \
            interpData(path_t, path_x, path_y, path_v, path_a, path_yaw, path_steer, path_steer_rate)
        # 反向计算，后面的奖励可以影响前面的奖励
        rewards = [0] * (len(new_path_t) - 1)
        for i in range(len(new_path_t)-1, 0, -1):
            state = generateObs(new_path_x[i], new_path_y[i], new_path_yaw[i], new_path_v[i], new_path_steer[i], agent.env, args)
            pre_state = generateObs(new_path_x[i-1], new_path_y[i-1], new_path_yaw[i-1], new_path_v[i-1],
                                    new_path_steer[i-1], agent.env, args)
            real_state = generateRealObs(new_path_x[i], new_path_y[i], new_path_yaw[i], new_path_v[i], new_path_steer[i],
                                agent.env)
            real_pre_state = generateRealObs(new_path_x[i - 1], new_path_y[i - 1], new_path_yaw[i - 1], new_path_v[i - 1],
                                    new_path_steer[i - 1], agent.env)
            action = np.array([new_path_a[i-1]/agent.env.a_max, new_path_steer_rate[i-1]/agent.env.omega_max])
            reward, done = agent.env.cal_reward_from_state(real_state, real_pre_state)

            if i < len(new_path_t) - 1:
                if args.reward_back:
                    reward = 0.8 * reward + 0.2 * rewards[i]  # 后续奖励叠加
                else:
                    reward = reward

            rewards[i-1] = reward

            agent.memory.add(pre_state, action, reward, state, done)

    for warm_train in range(1000):
        agent.optimize(tb)
    test_score, path = test(agent=agent, path_num=path_num, env=env, args=args)
    show(path, env.case, path_num, os.path.join(saveFigPath, "fig/Warm-Start-Case-{}.svg").format(path_num))
    path_t = [env.deltaT * k for k in range(len(path.x))]
    saveCsv(path_t=path_t, path_x=path.x, path_y=path.y, path_v=path.v, path_yaw=path.yaw, path_a=path.a,
            path_steer=path.steer, path_steer_rate=path.steer_rate, init_x=env.case.x0, init_y=env.case.y0,
            sampleT=env.deltaT, save_path=saveFigPath, i=-1, j=0, case_num=path_num)
    refineOBCA(path.x, path.y, path.yaw, env.x_goal, env.y_goal, env.yaw_goal, env, path_num=path_num)


def refineOBCA(path_x, path_y, path_yaw, goal_x, goal_y, goal_yaw, env, path_num=7, index=None):
    print("================= Operate By OBCA ===================")
    initialPath = []
    for i in range(len(path_x)):
        initialPath.append(OBCAPath(path_x[i], path_y[i], path_yaw[i]))
    initialPath.append(OBCAPath(goal_x, goal_y, goal_yaw))
    obstacles = []
    for obs_i in range(len(env.case.obs)):
        obs = list(env.case.obs[obs_i])
        obstacles.append(obs)

    # OBCA二次优化
    cfg = VehicleConfig()
    cfg.T = 0.1
    gap = 1
    sampleT = 0.1
    vehicle = Vehicle()
    path_x, path_y, path_v, path_yaw, path_steer, path_a, path_steer_rate = quadraticPath(
            initialQuadraticPath=initialPath, obstacles=obstacles,
            vehicle=vehicle, max_x=env.case.xmax, max_y=env.case.ymax,
            min_x=env.case.xmin, min_y=env.case.ymin,
            gap=gap, cfg=cfg, sampleT=sampleT)
    obcaPath = Path(path_x, path_y, path_yaw)
    show(obcaPath, env.case, path_num, os.path.join(saveFigPath, "fig/OBCA-Warm-Start-Case-{}-agent-{}.svg").format(path_num, index))


# 客户端融合权重
def average_weights(list_of_weight):
    """aggregate all weights"""
    averga_actor_w, averga_critic_w = copy.deepcopy(list_of_weight[0])
    for key in averga_actor_w.keys():
        for ind in range(1, len(list_of_weight)):
            averga_actor_w[key] += list_of_weight[ind][0][key]
        averga_actor_w[key] = torch.div(averga_actor_w[key], len(list_of_weight))
    for key in averga_critic_w.keys():
        for ind in range(1, len(list_of_weight)):
            averga_critic_w[key] += list_of_weight[ind][1][key]
        averga_critic_w[key] = torch.div(averga_critic_w[key], len(list_of_weight))
    return averga_actor_w, averga_critic_w


if __name__ == '__main__':
    TOTAL_PATH_NUM = [1, 2, 3, 7, 8]
    PATH_NUM = [1, 2, 3, 7]
    args = get_args()
    # Init. tensorboard summary writer
    tb_global = SummaryWriter(log_dir=os.path.abspath('data/{}/tensorboard/global'.format(args.exp_name)))
    tbs = []
    for i in range(args.agent_num):
        tbs.append(SummaryWriter(log_dir=os.path.abspath('data/{}/tensorboard/agent_{}'.format(args.exp_name, i))))

    # Init. Datapath
    data_path = os.path.abspath('data/{}'.format(args.exp_name))

    saveFigPath = os.path.join(data_path, "evaluate")

    if not os.path.exists(saveFigPath):
        os.mkdir(saveFigPath)
        os.mkdir(os.path.join(saveFigPath, "fig"))
        os.mkdir(os.path.join(saveFigPath, "svg"))
        os.mkdir(os.path.join(saveFigPath, "csv"))

    # Init. Training
    n_games: int = 2500
    best_score = [-np.inf for k in range(args.agent_num)]
    global_best_score = -np.inf
    score_history = {k: [] * n_games for k in range(args.agent_num)}
    score_history_test = {k: [] for k in range(args.agent_num)}
    global_score_history_test = []
    avg_history = {k: [] * n_games for k in range(args.agent_num)}
    logging_info: List[Dict[str, float]] = [] * n_games
    if args.random_choice:
        score_path_history_test = {k: [[] for _ in range(len(TOTAL_PATH_NUM))] for k in range(args.agent_num)}
        score_path_history_train = {k: [[] for _ in range(len(PATH_NUM))] for k in range(args.agent_num)}
        global_score_path_history_test = [[] for _ in range(len(TOTAL_PATH_NUM))]

    # if args.warm_start:
    #     warm_start(args, agent, tb, path_num, env, saveFigPath)

    Agents = [Agent(args, n_games, PATH_NUM, index, tbs[index]) for index in range(args.agent_num)]

    GlobalAgent = Agent(args, n_games, PATH_NUM, 'global', tb_global)

    for i in range(n_games):
        # 每个参与者训练一轮，记录每个智能体的score变化
        for index, item_agent in enumerate(Agents):
            temp_score, path_num = item_agent.play()
            score_history[index].append(temp_score)
            tbs[index].add_scalar('agent_{}_path_{}_score'.format(index, path_num), temp_score, i)

        # 记录每个参与者的score历史平均变化
        for item in range(args.agent_num):
            avg_score = np.mean(score_history[item][-100:])
            avg_history[item].append(avg_score)
            tbs[item].add_scalar('agent_{}_avg_score'.format(item), avg_score, i)

        # 模型融合后，重新评估
        if i > 0 and i % args.avg_loop == 0:
            print("============= Average the model parameters ===============")
            global_actor, global_critic = average_weights([_agent.get_state_dict() for _agent in Agents])
            global_target_actor, global_target_critic = average_weights([_agent.get_target_dict() for _agent in Agents])
            GlobalAgent.set_state_dict(global_actor, global_critic)
            print("Epoch = " + str(i) + "/" + str(n_games) + "Global averaging starts")

            for item_agent in Agents:
                item_agent.set_state_dict(global_actor, global_critic)

        if args.checkpoint_interval != 0 and i % args.checkpoint_interval == 0:
            for item_agent in Agents:
                item_agent.save(data_path)

        if i % args.evaluate_loop == 0:
            # 在参与者个人模型上评估
            for index, item_agent in enumerate(Agents):
                for path_item in TOTAL_PATH_NUM:
                    temp_score, path_num, total_path = item_agent.test_eva([path_item])
                    use_obca = False
                    if args.random_choice:
                        score_path_history_test[index][TOTAL_PATH_NUM.index(path_num)].append(temp_score)
                        if temp_score > np.mean(score_path_history_test[index][TOTAL_PATH_NUM.index(path_num)]):
                            # score_path_history_test[index][TOTAL_PATH_NUM.index(path_num)].append(temp_score)
                            item_agent.model_agent.reload_step_state_dict(True)
                            item_agent.save(data_path)
                            print(f'Test Episode:{i}'
                                  f'\t Agent Index:{index}'
                                  f'\t Path Num:{path_num}'
                                  f'\t ACC. Rewards: {temp_score:3.2f}'
                                  f'\t AVG. Rewards: {np.mean(score_path_history_test[index][TOTAL_PATH_NUM.index(path_num)]):3.2f}'
                                  f'\t *** MODEL SAVED SINGLE SCORE! ***')
                            use_obca = True
                        else:
                            item_agent.model_agent.reload_step_state_dict()
                            print(f'Test Episode:{i}'
                                  f'\t Agent Index:{index}'
                                  f'\t Path Num:{path_num}'
                                  f'\t ACC. Rewards: {temp_score:3.2f}'
                                  f'\t AVG. Rewards: {np.mean(score_path_history_test[index][TOTAL_PATH_NUM.index(path_num)]):3.2f}')
                    else:
                        if temp_score > best_score[index]:
                            best_score[index] = temp_score
                            score_history_test[index].append(temp_score)
                            item_agent.save(data_path)
                            print(f'Test Episode:{i}'
                                  f'\t Path Num:{path_num}'
                                  f'\t ACC. Rewards: {temp_score:3.2f}'
                                  f'\t AVG. Rewards: {np.mean(score_history_test[index]):3.2f}'
                                  f'\t *** MODEL SAVED SINGLE SCORE! ***')
                            item_agent.model_agent.reload_step_state_dict(True)
                            use_obca = True
                        else:
                            item_agent.model_agent.reload_step_state_dict()
                            print(f'Test Episode:{i}'
                                  f'\t Path Num:{path_num}'
                                  f'\t ACC. Rewards: {temp_score:3.2f}'
                                  f'\t AVG. Rewards: {np.mean(score_history_test[index]):3.2f}')
                    show(total_path, item_agent.env.case, path_num,
                         os.path.join(saveFigPath, "fig/Case-{}-{}-agent-{}.svg").format(path_num, i, index))
                    path_t = [item_agent.env.deltaT * k for k in range(len(total_path.x))]
                    saveCsv(path_t=path_t, path_x=total_path.x, path_y=total_path.y, path_v=total_path.v, path_yaw=total_path.yaw,
                            path_a=total_path.a,
                            path_steer=total_path.steer, path_steer_rate=total_path.steer_rate, init_x=item_agent.env.case.x0,
                            init_y=item_agent.env.case.y0,
                            sampleT=item_agent.env.deltaT, save_path=saveFigPath, i=i, j=index, case_num=path_num)
                    if use_obca and args.obca:
                        refineOBCA(total_path.x, total_path.y, total_path.yaw, item_agent.env.x_goal, item_agent.env.y_goal, item_agent.env.yaw_goal, item_agent.env,
                                   path_num=path_num, index=index)

            #  对GLOBAL模型评估
            for path_item in TOTAL_PATH_NUM:
                temp_score, path_num, total_path = GlobalAgent.test_eva([path_item])
                tb_global.add_scalar("Global_path_{}_score".format(path_num), temp_score, i)
                use_obca = False
                if args.random_choice:
                    global_score_path_history_test[TOTAL_PATH_NUM.index(path_num)].append(temp_score)
                    if temp_score > np.mean(global_score_path_history_test[TOTAL_PATH_NUM.index(path_num)]):
                        GlobalAgent.model_agent.reload_step_state_dict(True)
                        GlobalAgent.save(data_path)
                        print(f'Test Episode:{i}'
                              f'\t Agent Index: Global'
                              f'\t Path Num:{path_num}'
                              f'\t ACC. Rewards: {temp_score:3.2f}'
                              f'\t AVG. Rewards: {np.mean(global_score_path_history_test[TOTAL_PATH_NUM.index(path_num)]):3.2f}'
                              f'\t ***GLOBAL MODEL SAVED SINGLE SCORE! ***')
                        use_obca = True
                    else:
                        GlobalAgent.model_agent.reload_step_state_dict()
                        print(f'Test Episode:{i}'
                              f'\t Agent Index: Global'
                              f'\t Path Num:{path_num}'
                              f'\t ACC. Rewards: {temp_score:3.2f}'
                              f'\t AVG. Rewards: {np.mean(global_score_path_history_test[TOTAL_PATH_NUM.index(path_num)]):3.2f}')
                    tb_global.add_scalar("Global_path_{}_avg_score".format(path_num), np.mean(global_score_path_history_test[TOTAL_PATH_NUM.index(path_num)]), i)
                else:
                    if temp_score > global_best_score:
                        global_best_score = temp_score
                        global_score_history_test.append(temp_score)
                        GlobalAgent.save(data_path)
                        print(f'Test Episode:{i}'
                              f'\t Agent Index: Global'
                              f'\t Path Num:{path_num}'
                              f'\t ACC. Rewards: {temp_score:3.2f}'
                              f'\t AVG. Rewards: {np.mean(global_score_history_test):3.2f}'
                              f'\t ***GLOBAL MODEL SAVED SINGLE SCORE! ***')
                        GlobalAgent.model_agent.reload_step_state_dict(True)
                        use_obca = True
                    else:
                        GlobalAgent.model_agent.reload_step_state_dict()
                        print(f'Test Episode:{i}'
                              f'\t Agent Index: Global'
                              f'\t Path Num:{path_num}'
                              f'\t ACC. Rewards: {temp_score:3.2f}'
                              f'\t AVG. Rewards: {np.mean(global_score_history_test):3.2f}')

                    tb_global.add_scalar("Global_best_score", global_best_score, i)
                show(total_path, GlobalAgent.env.case, path_num,
                     os.path.join(saveFigPath, "fig/Case-{}-{}-agent-{}.svg").format(path_num, i, "global"))
                path_t = [GlobalAgent.env.deltaT * k for k in range(len(total_path.x))]
                saveCsv(path_t=path_t, path_x=total_path.x, path_y=total_path.y, path_v=total_path.v, path_yaw=total_path.yaw,
                        path_a=total_path.a,
                        path_steer=total_path.steer, path_steer_rate=total_path.steer_rate, init_x=GlobalAgent.env.case.x0,
                        init_y=GlobalAgent.env.case.y0,
                        sampleT=GlobalAgent.env.deltaT, save_path=saveFigPath, i=i, j="global", case_num=path_num)
                if use_obca and args.obca:
                    refineOBCA(total_path.x, total_path.y, total_path.yaw, GlobalAgent.env.x_goal, GlobalAgent.env.y_goal,
                               GlobalAgent.env.yaw_goal, GlobalAgent.env,
                               path_num=path_num, index="global")

        # Train Agent
        # print("========= Training Agent ============")
        # for index, item_agent in enumerate(Agents):
        #     item_agent.model_agent.optimize(tbs[index], item_agent.index)


        # episode_info = {
        #     'Episode': i,
        #     'Path Num': path_num,
        #     'Total Episodes': n_games,
        #     'Epidosic Summed Rewards': score,
        #     'Moving Mean of Episodic Rewards': avg_score
        # }

        # logging_info.append(episode_info)

        # Dump .json
        # with open(os.path.join(data_path, 'training_info.json'), 'w', encoding='utf8') as file:
        #     json.dump(logging_info, file, indent=4, ensure_ascii=False)

    # Close tensorboard writer
    tb_global.close()
    for item_tb in tbs:
        item_tb.close()
