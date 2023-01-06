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
from rl.algorithms.augmentations import her_augmentation


def get_args():
    args = argparse.ArgumentParser("The args of TPCAP RL")
    args.add_argument('--random_choice', action="store_true", default=False, help="If random choice the path num?")
    args.add_argument('--exp_name', default="test", type=str, help="The name of exp")
    return args.parse_args()


if __name__ == '__main__':
    PATH_NUM = [1, 2, 3, 7, 8, 13, 14]
    args = get_args()
    # Init. Environment
    if args.random_choice:
        if random.random() > 0.5:
            path_num = 7
        else:
            path_num = random.choice(PATH_NUM)
    else:
        path_num = 7
    env = Environment(path_num)
    env.reset(path_num)
    # Init. tensorboard summary writer
    tb = SummaryWriter(log_dir=os.path.abspath('data/{}/tensorboard'.format(args.exp_name)))
    # Init. Datapath
    data_path = os.path.abspath('data/{}'.format(args.exp_name))

    # Init. Training
    n_games: int = 2500
    best_score = -np.inf
    score_history: List[float] = [] * n_games
    avg_history: List[float] = [] * n_games
    logging_info: List[Dict[str, float]] = [] * n_games

    # Init. Agent
    agent = Agent(env=env, n_games=n_games)

    for i in range(n_games):
        done: bool = False
        score: float = 0.0

        states: List[Dict[str, np.ndarray]] = []
        actions: List[np.ndarray] = []
        one_hot_actions: List[np.ndarray] = []
        next_states: List[Dict[str, np.ndarray]] = []

        # Initial Reset of Environment
        if args.random_choice:
            if random.random() > 0.5:
                path_num = 7
            else:
                path_num = random.choice(PATH_NUM)
        else:
            path_num = 7
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
            action = agent.choose_action(obs)
            next_OBS, reward, done = env.step(action)

            next_state, next_actgoal, next_desgoal = next_OBS.values()
            assert len(next_actgoal) == 3 and len(next_desgoal) == 3, "Error!"
            # next_obs = np.concatenate((next_state, next_actgoal, next_desgoal))
            next_obs = np.array(next_state)

            one_hot_action = torch.nn.functional.one_hot(torch.tensor(action), agent.n_actions).numpy()

            agent.memory.add(obs, one_hot_action, reward, next_obs, done)
            agent.optimize(tb)

            states.append(OBS)
            next_states.append(next_OBS)
            actions.append(action)
            one_hot_actions.append(one_hot_action)

            OBS = next_OBS
            score += reward

        her_augmentation(agent, states, one_hot_actions, next_states)

        score_history.append(score)
        avg_score: float = np.mean(score_history[-100:])
        avg_history.append(avg_score)

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

        episode_info = {
            'Episode': i,
            'Path Num': path_num,
            'Total Episodes': n_games,
            'Epidosic Summed Rewards': score,
            'Moving Mean of Episodic Rewards': avg_score
        }

        logging_info.append(episode_info)

        # Add info. to tensorboard
        tb.add_scalars('training_rewards',
                       {'Epidosic Summed Rewards': score,
                        'Moving Mean of Episodic Rewards': avg_score}, i)

        # Dump .json
        with open(os.path.join(data_path, 'training_info.json'), 'w', encoding='utf8') as file:
            json.dump(logging_info, file, indent=4, ensure_ascii=False)

    # Close tensorboard writer
    tb.close()
