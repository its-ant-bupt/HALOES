import torch
import random
import argparse
from torch.utils.tensorboard import SummaryWriter

from rl.environment.env import Environment
from rl.algorithms.PDDPG import Agent as PAgent
from rl.algorithms.PDDPG_C_Fed import Agent as PCAgent
from rl.algorithms.SAC import Agent as SACAgent
from rl.algorithms.augmentations import her_augmentation
from rl.sparse_rewards.prioritized.test import test
from rl.utils.savePath import show, saveCsv
from rl.utils.dealData import *
from Vehicle import OBCAPath, Vehicle, Path
from pyobca.search import VehicleConfig
from quadraticOBCA import quadraticPath


# 单个智能体网络
class Agent:
    def __init__(self, args, n_games, PATH_NUM, index, tb):
        self.PATH_NUM = PATH_NUM
        self.args = args
        self.env = Environment(7, args=args)
        self.n_games = n_games
        self.tb = tb  # tensorboard summary
        self.index = index
        self.count = 0
        self.model_agent = self.gen_model()

    def gen_model(self):
        if self.args.action_c:
            if self.args.alg == "SAC":
                agent = SACAgent(env=self.env, n_games=self.n_games, args=self.args)
            else:
                # default DDPG
                agent = PCAgent(env=self.env, n_games=self.n_games, index=self.index, args=self.args)
        else:
            # default DDPG
            agent = PAgent(env=self.env, n_games=self.n_games)
        return agent

    def play(self):
        done = False
        score = 0.0
        states = []
        actions = []
        one_hot_actions = []
        next_states = []
        time_step = 0

        # Initial Reset of Environment
        if self.args.random_choice:
            path_num = random.choice(self.PATH_NUM)
        else:
            path_num = self.args.case
        OBS = self.env.reset(path_num)
        next_OBS = []
        while not done:
            # Unpack the observation
            # [状态，回放所用的目标点：当前的位置，真实的目标点]
            state, curr_actgoal, curr_desgoal = OBS.values()
            assert len(curr_actgoal) == 3 and len(curr_desgoal) == 3, "Error!"
            # obs = np.concatenate((state, curr_actgoal, curr_desgoal))
            obs = np.array(state)

            # Choose agent based action & make a transition
            action = self.model_agent.choose_action(obs, evaluate=False)

            next_OBS, reward, done = self.env.step(action)

            next_state, next_actgoal, next_desgoal = next_OBS.values()
            assert len(next_actgoal) == 3 and len(next_desgoal) == 3, "Error!"
            # next_obs = np.concatenate((next_state, next_actgoal, next_desgoal))
            next_obs = np.array(next_state)

            if self.args.action_c:
                one_hot_action = action
            else:
                one_hot_action = torch.nn.functional.one_hot(torch.tensor(action), self.model_agent.n_actions).numpy()

            self.model_agent.memory.add(obs, one_hot_action, reward, next_obs, done)

            if self.args.alg == "SAC":
                self.model_agent.optimize(self.tb, index=self.index)
            else:
                self.model_agent.optimize(self.tb, index=self.index)

            states.append(OBS)
            next_states.append(next_OBS)
            actions.append(action)
            one_hot_actions.append(one_hot_action)

            OBS = next_OBS
            score += reward
            time_step += 1
            self.count += 1
        return score, path_num

    def test_eva(self, path_nums):
        if self.args.random_choice:
            path_num = random.choice(path_nums)
        else:
            path_num = self.args.case
        score, TotalPath = test(self.model_agent, path_num, self.env, self.args)
        return score, path_num, TotalPath

    def get_state_dict(self):
        return self.model_agent.actor.state_dict(), self.model_agent.critic.state_dict()

    def get_target_dict(self):
        return self.model_agent.target_actor.state_dict(), self.model_agent.target_critic.state_dict()

    def set_state_dict(self, new_actor_dict, new_critic_dict):
        self.model_agent.set_state_dict(new_actor_dict, new_critic_dict)

    def save(self, path):
        self.model_agent.save_models(path, str(self.index))
