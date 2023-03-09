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
import time
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

from rl.environment.env import Environment
from rl.algorithms.PDDPG import Agent
from rl.algorithms.PDDPG_C import Agent as PCAgent
from rl.algorithms.PDDPG_C_IB import Agent as IBDDPGAgent
from rl.algorithms.PDDPG_C_Fed import Agent as FedAgent
from rl.algorithms.SAC import Agent as SACAgent
from rl.algorithms.PPO import Agent as PPOAgent
from rl.algorithms.IBAC_PPO import Agent as IBPPOAgent
from rl.algorithms.augmentations import her_augmentation
from rl.sparse_rewards.prioritized.test import test
from rl.utils.savePath import show, saveCsv, showZone
from rl.utils.dealData import *
from Vehicle import OBCAPath, Vehicle, Path
from pyobca.search import VehicleConfig
from quadraticOBCA import quadraticPath

from case import Case
import matplotlib.pyplot as plt


class tPath(object):
    # x::Array{Float64} # x position [m]
    # y::Array{Float64} # y position [m]
    # yaw::Array{Float64} # yaw angle [rad]
    # a::Array{Float64} # acc [m/s2]
    # steer::Array{Float64} # steer [rad]
    def __init__(self, x, y, yaw, v, a, steer_rate, steer):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.a = a
        self.steer = steer
        self.steer_rate = steer_rate


def saveKin(path_v, path_a, path_steer, path_steer_rate, sampleT=0.1, path_num=7, index="global"):
    plt.figure(figsize=(20, 5), dpi=100)
    # fig2, ax2 = plt.subplots(1, 4)
    plt.subplots_adjust(hspace=0.35)
    t_v = [sampleT * k for k in range(len(path_v))]
    t_a = [sampleT * k for k in range(len(path_a))]
    t_steer = [sampleT * k for k in range(len(path_steer))]
    t_steer_rate = [sampleT * k for k in range(len(path_steer_rate))]
    # ax2[0].plot(t_v, path_v, label='v-t')
    # ax2[1].plot(t_a, path_a, label='a-t')
    # ax2[2].plot(t_steer, path_steer, label='steer-t')
    # ax2[3].plot(t_steer_rate, path_steer_rate, label='steer-rate-t')
    # ax2[0].legend()
    # ax2[1].legend()
    # ax2[2].legend()
    # ax2[3].legend()
    plt.subplot(1, 4, 1)
    plt.plot(t_v, path_v, label='v-t')
    plt.xlabel("Time", fontsize=10)
    plt.ylabel("Action Value", fontsize=10)
    plt.legend()

    plt.subplot(1, 4, 2)
    plt.plot(t_a, path_a, label='a-t')
    plt.xlabel("Time", fontsize=10)
    plt.ylabel("Action Value", fontsize=10)
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.plot(t_steer, path_steer, label='steer-t')
    plt.xlabel("Time", fontsize=10)
    plt.ylabel("Action Value", fontsize=10)
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.plot(t_steer_rate, path_steer_rate, label='steer-rate-t')
    plt.xlabel("Time", fontsize=10)
    plt.ylabel("Action Value", fontsize=10)
    plt.legend()
    plt.savefig(os.path.join(saveFigPath, "fig/OBCA-Kina-Case-{}-agent-{}.png").format(path_num, index))



def refineOBCA(path_x, path_y, path_yaw, goal_x, goal_y, goal_yaw, env, path_num=7, index="global"):
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
    saveKin(path_v, path_a, path_steer, path_steer_rate, sampleT, path_num, index)


def get_args():
    args = argparse.ArgumentParser("The args of TPCAP RL")
    args.add_argument('--alg', type=str, default="DDPG", help="The algorithm of model")
    args.add_argument('--random_choice', action="store_true", default=False, help="If random choice the path num?")
    args.add_argument('--exp_name', default="fed_ddpg", type=str, help="The name of exp")
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
    args.add_argument('--test_batch', type=int, default=10, help="Test batch")

    # ===================== Federated Learning ======================
    args.add_argument('--agent_num', type=int, default=4, help='The number of federated agent')
    args.add_argument('--avg_loop', type=int, default=10, help='The gap of federated')
    args.add_argument('--checkpoint_interval', type=int, default=0, help='The interval of save agent model(default:Never)')
    args.add_argument('--shared', action="store_true", default=False, help="Use the shared model to train.")


    # ===================== DDPG Bottleneck===========================
    args.add_argument('--ddpg_hidden', type=int, default=256, help="The hidden dim of DDPG IB")
    args.add_argument('--ddpg_std_init', type=float, default=0.3, metavar='G',
                      help='action std init (default: 0.6)')

    # ======================     PPO    =================================
    args.add_argument('--ppo_std_decay_rate', type=float, default=0.05, metavar='G',
                      help='linearly decay action_std (action_std = action_std - action_std_decay_rate)')
    args.add_argument('--ppo_min_action_std', type=float, default=0.00, metavar='G',
                      help='minimum action_std (stop decay after action_std <= min_action_std)')
    args.add_argument('--ppo_std_decay_freq', type=int, default=10000,
                      help='action_std decay frequency (in num timesteps)')

    return args.parse_args()


def testCase(env ,agent, path_num, testTime=20):
    OBS = env.reset(path_num)
    done = False
    total_x = [env.x_pos]
    total_y = [env.y_pos]
    total_yaw = [env.yaw]
    total_v = [0]
    total_steer_rate = [0]
    total_a = [0]
    total_steer = [0]
    score = 0
    # while not done and env.totalT:
    while env.totalT < testTime:
        state, curr_actgoal, curr_desgoal = OBS.values()
        assert len(curr_actgoal) == 3 and len(curr_desgoal) == 3, "Error!"
        obs = np.array(state)

        # Choose agent based action & make a transition
        action = agent.choose_action(obs, True)
        next_OBS, reward, done = env.step(action)
        if len(action) == 1:
            aValue, steerValue = env.action_list[int(action)][0], env.action_list[int(action)][1]
        else:
            aValue, steerValue = action[0] * agent.env.a_max, action[1] * agent.env.omega_max

        total_a.append(aValue)
        total_steer_rate.append(steerValue)

        next_state, next_actgoal, next_desgoal = next_OBS.values()
        assert len(next_actgoal) == 3 and len(next_desgoal) == 3, "Error!"
        next_obs = np.array(next_state)
        # total_x.append(next_obs[0])
        # total_y.append(next_obs[1])
        total_x.append(env.x_pos)
        total_y.append(env.y_pos)
        total_yaw.append(env.yaw)
        total_v.append(next_obs[-2])
        total_steer.append(next_obs[-1])

        OBS = next_OBS
        score += reward

    TotalPath = tPath(total_x, total_y, total_yaw, total_v, total_a, total_steer_rate, total_steer)
    return TotalPath


if __name__ == '__main__':
    args = get_args()
    pathNum = args.case
    if args.test_batch > 0:
        TIMES = [0] * args.test_batch
        for batch in range(args.test_batch):
            env = Environment(pathNum, args=args)
            agent = FedAgent(env, index="global", args=args)
            data_path = os.path.abspath('data/{}'.format(args.exp_name))
            saveFigPath = os.path.join(data_path, "evaluate")
            agent.load_models(data_path, "global")
            time1 = time.time()
            path = testCase(agent=agent, path_num=pathNum, env=env, testTime=20)
            time2 = time.time()
            TIMES[batch] += (time2 - time1)
            env.reset(path_num=pathNum)
            time1 = time.time()
            refineOBCA(path.x, path.y, path.yaw, env.x_goal, env.y_goal, env.yaw_goal, env, path_num=pathNum)
            time2 = time.time()
            TIMES[batch] += (time2 - time1)
        print("Total time is:")
        print(TIMES)
        print("Mean time is:")
        print(np.mean(TIMES))
    else:
        env = Environment(pathNum, args=args)
        agent = FedAgent(env, index="global", args=args)
        data_path = os.path.abspath('data/{}'.format(args.exp_name))
        saveFigPath = os.path.join(data_path, "evaluate")
        agent.load_models(data_path, "global")
        path = testCase(agent=agent, path_num=pathNum, env=env, testTime=20)
        env.reset(path_num=pathNum)
        refineOBCA(path.x, path.y, path.yaw, env.x_goal, env.y_goal, env.yaw_goal, env, path_num=pathNum)
