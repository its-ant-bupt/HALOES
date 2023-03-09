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
from rl.algorithms.SAC import Agent as SACAgent
from rl.algorithms.PPO import Agent as PPOAgent
from rl.algorithms.IBAC_PPO import Agent as IBPPOAgent
from rl.algorithms.augmentations import her_augmentation
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
    args.add_argument('--exp_name', default="test", type=str, help="The name of exp")
    args.add_argument('--maxT', type=int, default=40, help="The total time of simulation.")
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
    args.add_argument('--ppo_gamma', type=float, default=0.99, metavar='G',
                      help='discount factor for reward ppo')
    args.add_argument('--ppo_lr_actor', type=float, default=0.0003, metavar='G',
                      help='learning rate of ppo actor')
    args.add_argument('--ppo_lr_critic', type=float, default=0.001, metavar='G',
                      help='learning rate of ppo actor')
    args.add_argument('--ppo_adam_eps', type=float, default=1e-5, metavar='G',
                      help='the adam eps of ppo')
    args.add_argument('--ppo_k_epochs', type=int, default=80, metavar='N',
                      help='update policy for k epochs in one ppo update')
    args.add_argument('--ppo_eps_clip', type=float, default=0.2, metavar='G',
                      help='clip parameter for ppo')
    args.add_argument('--ppo_hidden_size', type=int, default=256, metavar='N',
                      help='hidden size (default: 256)')
    args.add_argument('--ppo_std_init', type=float, default=0.6, metavar='G',
                      help='action std init (default: 0.6)')
    args.add_argument('--ppo_batch_size', type=int, default=256, metavar='N',
                      help='batch size (default: 256)')
    args.add_argument('--ppo_update_timestep', type=int, default=800, metavar='N',
                      help='update timestep of ppo = max_len * 4')
    args.add_argument('--ppo_std_decay_rate', type=float, default=0.05, metavar='G',
                      help='linearly decay action_std (action_std = action_std - action_std_decay_rate)')
    args.add_argument('--ppo_min_action_std', type=float, default=0.00, metavar='G',
                      help='minimum action_std (stop decay after action_std <= min_action_std)')
    args.add_argument('--ppo_std_decay_freq', type=int, default=10000,
                      help='action_std decay frequency (in num timesteps)')

    # ======================    IBAC PPO    =============================
    args.add_argument('--ibppo_lr', type=float, default=0.0001,
                      help="the learning rate for optimizers.")
    args.add_argument('--ibppo_gae_lambda', type=float, default=0.95,
                      help="the factor of gae lambda.")
    args.add_argument('--ibppo_entropy_coef', type=float, default=0.01,
                      help="the factor of entropy coef.")
    args.add_argument('--ibppo_value_loss_coef', type=float, default=0.5,
                      help="the factor of value loss coef.")
    args.add_argument('--ibppo_max_grad_norm', type=float, default=0.5,
                      help="the factor of max grad norm.")
    args.add_argument('--ibppo_beta', type=float, default=1.0,
                      help="the factor of beta.")
    args.add_argument('--ibppo_sni_type', type=str, default="vib",
                      help="the type of sni.")
    args.add_argument('--ibppo_use_bottleneck', action="store_true", default=False,
                      help="use the bottleneck.")
    args.add_argument('--ibppo_use_l2a', action="store_true", default=False,
                      help="use the l2a.")
    args.add_argument('--ibppo_use_bn', action="store_true", default=False,
                      help="use the batch norm.")
    args.add_argument('--ibppo_use_dropout', action="store_true", default=False,
                      help="use the dropout.")
    args.add_argument('--ibppo_use_l2w', action="store_true", default=False,
                      help="use the l2 loss of weight.")
    args.add_argument('--res_net', action="store_true", default=False,
                      help="use the res net framework to actor")

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


def refineOBCA(path_x, path_y, path_yaw, goal_x, goal_y, goal_yaw, env, path_num=7):
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
    show(obcaPath, env.case, path_num, os.path.join(saveFigPath, "fig/OBCA-Warm-Start-Case-{}.svg").format(path_num))



if __name__ == '__main__':
    TOTAL_PATH_NUM = [1, 2, 3, 7, 8, 13, 14]
    PATH_NUM = [1, 2, 3, 7]
    args = get_args()
    # Init. Environment
    if args.random_choice:
        # if random.random() > 0.5:
        #     path_num = args.case
        # else:
        #     path_num = random.choice(PATH_NUM)
        path_num = random.choice(PATH_NUM)
        # 记录最好成绩
        test_num_score = {key: -math.inf for key in TOTAL_PATH_NUM}
        test_num_best_score = -math.inf
        train_num_score = {key: -math.inf for key in TOTAL_PATH_NUM}
    else:
        path_num = args.case
        test_best_score = -math.inf
    env = Environment(path_num, args=args)
    env.reset(path_num)
    # Init. tensorboard summary writer
    tb = SummaryWriter(log_dir=os.path.abspath('data/{}/tensorboard'.format(args.exp_name)))
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
    best_score = -np.inf
    score_history: List[float] = [] * n_games
    avg_history: List[float] = [] * n_games
    logging_info: List[Dict[str, float]] = [] * n_games

    # Init. Agent
    if args.action_c:
        if args.alg == "SAC":
            agent = SACAgent(env=env, n_games=n_games, args=args)
        elif args.alg == "PPO":
            agent = PPOAgent(env=env, n_games=n_games, args=args)
            args.ppo_update_timestep = int(args.maxT/0.1)
        elif args.alg == "IBPPO":
            agent = IBPPOAgent(env=env, n_games=n_games, args=args)
            args.ppo_update_timestep = int(args.maxT / 0.1)
        elif args.alg == "IBDDPG":
            agent = IBDDPGAgent(env=env, n_games=n_games, args=args)
        else:
            # default DDPG
            agent = PCAgent(env=env, n_games=n_games)
    else:
        if args.alg == "IBPPO":
            agent = IBPPOAgent(env=env, n_games=n_games, args=args)
            args.ppo_update_timestep = int(args.maxT / 0.1)
        else:
            # default DDPG
            agent = Agent(env=env, n_games=n_games)

    if args.warm_start:
        warm_start(args, agent, tb, path_num, env, saveFigPath)

    time_step = 0

    for i in range(n_games):
        done: bool = False
        score: float = 0.0

        states: List[Dict[str, np.ndarray]] = []
        actions: List[np.ndarray] = []
        one_hot_actions: List[np.ndarray] = []
        next_states: List[Dict[str, np.ndarray]] = []

        # Initial Reset of Environment
        if args.random_choice:
            # if random.random() > 0.5:
            #     path_num = args.case
            # else:
            #     path_num = random.choice(PATH_NUM)
            path_num = random.choice(PATH_NUM)
        else:
            path_num = args.case
        OBS: Dict[str, np.ndarray] = env.reset(path_num)
        next_OBS: Dict[str, np.array]


        while not done:
            # Unpack the observation
            # [状态，回放所用的目标点：当前的位置，真实的目标点]
            state, curr_actgoal, curr_desgoal = OBS.values()
            assert len(curr_actgoal) == 3 and len(curr_desgoal) == 3, "Error!"
            # obs = np.concatenate((state, curr_actgoal, curr_desgoal))
            obs = np.array(state)

            # Choose agent based action & make a transition
            if args.alg == "IBDDPG":
                action, log_prob = agent.choose_action(obs, evaluate=False)
            else:
                action = agent.choose_action(obs, evaluate=False)

            next_OBS, reward, done = env.step(action)


            next_state, next_actgoal, next_desgoal = next_OBS.values()
            assert len(next_actgoal) == 3 and len(next_desgoal) == 3, "Error!"
            # next_obs = np.concatenate((next_state, next_actgoal, next_desgoal))
            next_obs = np.array(next_state)

            if args.action_c:
                one_hot_action = action
            else:
                one_hot_action = torch.nn.functional.one_hot(torch.tensor(action), agent.n_actions).numpy()

            if args.alg == "PPO":
                agent.buffer.rewards.append(reward)
                agent.buffer.is_terminals.append(done)
            elif args.alg == "IBPPO":
                agent.buffer.rewards.append(torch.unsqueeze(torch.tensor(reward, device=agent.device), dim=-1))
                agent.buffer.masks.append(1 - done)
            elif args.alg == "IBDDPG":
                agent.memory.add(obs, one_hot_action, reward, log_prob, next_obs, done)
            else:
                agent.memory.add(obs, one_hot_action, reward, next_obs, done)

            if args.alg == "SAC":
                agent.optimize(tb)
            elif args.alg == "PPO":
                # if (time_step+1) % args.ppo_update_timestep == 0:
                if done:
                    loss = agent.optimize(tb)
                    tb.add_scalar('loss', loss, i)
                if (time_step+1) % args.ppo_std_decay_freq == 0:
                    agent.decay_action_std(args.ppo_std_decay_rate, args.ppo_min_action_std)
            elif args.alg == "IBPPO":
                if done:
                    entropy, value, policy_loss, value_loss, kl, loss = agent.optimize(tb)
                    tb.add_scalar('entropy', entropy, i)
                    tb.add_scalar('value', value, i)
                    tb.add_scalar('policy_loss', policy_loss, i)
                    tb.add_scalar('value_loss', value_loss, i)
                    tb.add_scalar('kl', kl, i)
                    tb.add_scalar('loss', loss, i)
                if (time_step+1) % args.ppo_std_decay_freq == 0:
                    agent.decay_action_std(args.ppo_std_decay_rate, args.ppo_min_action_std)
            elif args.alg == "IBDDPG":
                agent.optimize(tb)
                if (time_step+1) % args.ppo_std_decay_freq == 0:
                    agent.decay_action_std(args.ppo_std_decay_rate, args.ppo_min_action_std)
            else:
                # default DDPG
                agent.optimize(tb)


            states.append(OBS)
            next_states.append(next_OBS)
            actions.append(action)
            one_hot_actions.append(one_hot_action)

            OBS = next_OBS
            score += reward
            time_step += 1

        if args.her:
            her_augmentation(agent, states, one_hot_actions, next_states)
        score_history.append(score)
        avg_score: float = np.mean(score_history[-100:])
        avg_history.append(avg_score)

        tb.add_scalar('path_{}_score'.format(path_num), score, i)
        tb.add_scalar('avg_score', avg_score, i)

        if args.random_choice:
            if score > train_num_score[path_num]:
                train_num_score[path_num] = score
                agent.save_models(data_path)
                print(f'Episode:{i}'
                      f'\t Path Num:{path_num}'
                      f'\t ACC. Rewards: {score:3.2f}'
                      f'\t AVG. Rewards: {avg_score:3.2f}'
                      f'\t *** MODEL SAVED SINGLE SCORE! ***')
            elif avg_score > best_score:
                best_score = avg_score
                agent.save_models(data_path)
                print(f'Episode:{i}'
                      f'\t Path Num:{path_num}'
                      f'\t ACC. Rewards: {score:3.2f}'
                      f'\t AVG. Rewards: {avg_score:3.2f}'
                      f'\t *** MODEL SAVED AVERAGE SCORE! ***')
            else:
                print(f'Episode:{i}'
                      f'\t Path Num:{path_num}'
                      f'\t ACC. Rewards: {score:3.2f}'
                      f'\t AVG. Rewards: {avg_score:3.2f}')
        else:
            if avg_score > best_score:
                best_score = avg_score
                agent.save_models(data_path)
                print(f'Episode:{i}'
                      f'\t Path Num:{path_num}'
                      f'\t ACC. Rewards: {score:3.2f}'
                      f'\t AVG. Rewards: {avg_score:3.2f}'
                      f'\t *** MODEL SAVED! ***')
            else:
                print(f'Episode:{i}'
                      f'\t Path Num:{path_num}'
                      f'\t ACC. Rewards: {score:3.2f}'
                      f'\t AVG. Rewards: {avg_score:3.2f}')

        if i % args.evaluate_loop == 0:
            # 若是随机训练则每隔一定时间训练测试一次集合
            if args.random_choice:
                test_scores = []
                for path_num_item in TOTAL_PATH_NUM:
                    test_score, path = test(agent=agent, path_num=path_num_item, env=env, args=args)
                    test_scores.append(test_score)
                    show(path, env.case, path_num_item,
                         os.path.join(saveFigPath, "fig/Case-{}-{}-{}.svg").format(path_num_item, i, 0))
                    path_t = [env.deltaT * k for k in range(len(path.x))]
                    saveCsv(path_t=path_t, path_x=path.x, path_y=path.y, path_v=path.v, path_yaw=path.yaw,
                            path_a=path.a,
                            path_steer=path.steer, path_steer_rate=path.steer_rate, init_x=env.case.x0,
                            init_y=env.case.y0,
                            sampleT=env.deltaT, save_path=saveFigPath, i=i, j=0, case_num=path_num_item)
                    if test_score > test_num_score[path_num_item]:
                        print("Path Num: {} Occur The Best Score: {}".format(path_num_item, test_score))
                        test_num_score[path_num_item] = test_score
                        show(path, env.case, path_num_item,
                             os.path.join(saveFigPath, "fig/Best-Case-{}-{}-{}.svg").format(path_num_item, i, test_score))
                        if path_num_item == 3:
                            a = 1
                        refineOBCA(path.x, path.y, path.yaw, env.x_goal, env.y_goal, env.yaw_goal, env,
                                   path_num=path_num_item)
                    tb.add_scalar('test_path_{}_score'.format(path_num_item), test_score, int(i/args.evaluate_loop))
                tb.add_scalar('test_path_total_score', sum(test_scores), int(i / args.evaluate_loop))

            else:
                test_score, path = test(agent=agent, path_num=path_num, env=env, args=args)
                show(path, env.case, path_num, os.path.join(saveFigPath, "fig/Case-{}-{}-{}.svg").format(path_num, i, 0))
                path_t = [env.deltaT * k for k in range(len(path.x))]
                saveCsv(path_t=path_t, path_x=path.x, path_y=path.y, path_v=path.v, path_yaw=path.yaw, path_a=path.a,
                        path_steer=path.steer, path_steer_rate=path.steer_rate, init_x=env.case.x0, init_y=env.case.y0,
                        sampleT=env.deltaT, save_path=saveFigPath, i=i, j=0, case_num=path_num)
                if test_score > test_best_score:
                    test_best_score = test_score
                    print("Path Num: {} Occur The Best Score: {}".format(path_num, test_score))
                    show(path, env.case, path_num,
                         os.path.join(saveFigPath, "fig/Best-Case-{}-{}-{}.svg").format(path_num, i, test_score))
                    # 没获取到最优的路径后使用obca优化
                    refineOBCA(path.x, path.y, path.yaw, env.x_goal, env.y_goal, env.yaw_goal, env, path_num=path_num)
                tb.add_scalar('test_path_{}_score'.format(path_num), test_score, int(i / args.evaluate_loop))


        episode_info = {
            'Episode': i,
            'Path Num': path_num,
            'Total Episodes': n_games,
            'Epidosic Summed Rewards': score,
            'Moving Mean of Episodic Rewards': avg_score
        }

        logging_info.append(episode_info)

        # Add info. to tensorboard
        # tb.add_scalars('training_rewards',
        #                {'Epidosic Summed Rewards': score,
        #                 'Moving Mean of Episodic Rewards': avg_score}, i)

        # Dump .json
        with open(os.path.join(data_path, 'training_info.json'), 'w', encoding='utf8') as file:
            json.dump(logging_info, file, indent=4, ensure_ascii=False)

    # Close tensorboard writer
    tb.close()
