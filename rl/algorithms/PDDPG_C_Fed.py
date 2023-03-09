# Library Imports
import numpy as np
import os

from gym import Env
from rl.environment.env import Environment

import torch
from torch.nn import functional as F
import random

from rl.replay_buffers.PER import PrioritizedReplayBuffer
from rl.replay_buffers.utils import LinearSchedule

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Critic(torch.nn.Module):
    """Defines a Critic Deep Learning Network"""

    def __init__(self, input_dim: int, beta: float, density: int = 512, name: str = 'critic'):
        super(Critic, self).__init__()

        self.model_name = name
        self.checkpoint = self.model_name

        # Architecture
        self.H1 = torch.nn.Linear(input_dim, density)
        self.H2 = torch.nn.Linear(density, density)
        self.drop = torch.nn.Dropout(p=0.1)
        self.H3 = torch.nn.Linear(density, density)
        self.H4 = torch.nn.Linear(density, density)
        self.Q = torch.nn.Linear(density, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.device = device
        self.to(device)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        value = torch.hstack((state, action))
        value = F.relu(self.H1(value))
        value = F.relu(self.H2(value))
        value = self.drop(value)
        value = F.relu(self.H3(value))
        value = F.relu(self.H4(value))
        value = self.Q(value)
        return value

    def save_model(self, path, index):
        torch.save(self.state_dict(), os.path.join(path, self.checkpoint) + index + '.pth')

    def load_model(self, path, index):
        self.load_state_dict(torch.load(os.path.join(path, self.checkpoint) + index + '.pth'))


class Actor(torch.nn.Module):
    """Defines a Actor Deep Learning Network"""

    def __init__(self, input_dim: int, n_actions: int, alpha: float, density: int = 512, name='actor'):
        super(Actor, self).__init__()

        self.model_name = name
        self.checkpoint = self.model_name

        # Architecture
        self.H1 = torch.nn.Linear(input_dim, density)
        self.H2 = torch.nn.Linear(density, density)
        self.drop = torch.nn.Dropout(p=0.1)
        self.H3 = torch.nn.Linear(density, density)
        self.H4 = torch.nn.Linear(density, density)
        self.mu = torch.nn.Linear(density, n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = device
        self.to(device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        value = F.relu(self.H1(state))
        value = F.relu(self.H2(value))
        value = self.drop(value)
        value = F.relu(self.H3(value))
        value = F.relu(self.H4(value))
        action = torch.tanh(self.mu(value))
        # action = torch.softmax(action, dim=-1)
        return action

    def save_model(self, path, index):
        torch.save(self.state_dict(), os.path.join(path, self.checkpoint) + index + '.pth')

    def load_model(self, path, index):
        self.load_state_dict(torch.load(os.path.join(path, self.checkpoint) + index + '.pth'))


class Agent:
    def __init__(self,
                 env: Environment,
                 training: bool = True,
                 alpha=1e-4,
                 beta=1e-3,
                 gamma=0.99,
                 tau=0.005,
                 batch_size: int = 256,
                 noise: float = 0.01,
                 n_games: int = 250,
                 index=None,
                 args=None):

        self.gamma = torch.tensor(gamma, dtype=torch.float32, device=device)
        self.tau = tau
        self.n_games = n_games
        self.env = env
        self.greedy = 0.9
        self.e_greedy = 0.0001
        self.greedy_min = 0.01
        self.index = index
        self.args = args

        self.max_size: int = (env._max_episode_steps * n_games * 2)
        self.memory = PrioritizedReplayBuffer(self.max_size, 0.6)
        self.beta_scheduler = LinearSchedule(n_games, 0.4, 0.9)
        self.batch_size = batch_size

        self.obs_shape: int = env.observation_space
        self.n_actions: int = 2

        self.max_action = torch.as_tensor(1, dtype=torch.float32, device=device)
        self.min_action = torch.as_tensor(-1, dtype=torch.float32, device=device)

        self.actor = Actor(self.obs_shape, self.n_actions, alpha, name='actor')
        self.critic = Critic(self.obs_shape + self.n_actions, beta, name='critic')
        self.target_actor = Actor(self.obs_shape, self.n_actions, alpha, name='target_actor')
        self.target_critic = Critic(self.obs_shape + self.n_actions, beta, name='target_critic')

        if args.shared:
            self.shared_actor = Actor(self.obs_shape, self.n_actions, alpha, name='shared_actor')
            self.shared_critic = Critic(self.obs_shape + self.n_actions, beta, name='shared_critic')

        self.actor_dict = self.actor.state_dict()
        self.critic_dict = self.critic.state_dict()
        self.target_actor_dict = self.target_actor.state_dict()
        self.target_critic_dict = self.target_critic.state_dict()

        self.mse = torch.nn.MSELoss()
        self.is_training = training
        self.noise = noise

        self.per_step: int = 0

        self.update_nums = 0

        self.soft_update_target_networks()

    def coral_func(self, src, tar):
        """
        inputs:
            -src(Variable) : features extracted from source data
            -tar(Variable) : features extracted from target data
        return coral loss between source and target features
        ref: Deep CORAL: Correlation Alignment for Deep Domain Adaptation \
             (https://arxiv.org/abs/1607.01719
        """
        ns, nt = src.data.shape[0], tar.data.shape[0]
        dim = src.data.shape[1]

        ones_s = torch.ones(1, ns, dtype=torch.float32).to(device)
        ones_t = torch.ones(1, nt, dtype=torch.float32).to(device)
        tmp_s = torch.matmul(ones_s, src)
        tmp_t = torch.matmul(ones_t, tar)
        cov_s = (torch.matmul(src.T, src) -
                 torch.matmul(tmp_s.T, tmp_s) / ns) / (ns - 1)
        cov_t = (torch.matmul(tar.T, tar) -
                 torch.matmul(tmp_t.T, tmp_t) / nt) / (nt - 1)

        coral = torch.sum(self.mse(cov_s, cov_t)) / (4 * dim * dim)
        return coral

    def reload_step_state_dict(self, better=True):
        if better:
            self.actor_dict = self.actor.state_dict()
            self.critic_dict = self.critic.state_dict()
            self.target_actor_dict = self.target_actor.state_dict()
            self.target_critic_dict = self.target_critic.state_dict()
        else:
            self.actor.load_state_dict(self.actor_dict)
            self.critic.load_state_dict(self.critic_dict)
            self.target_actor.load_state_dict(self.target_actor_dict)
            self.target_critic.load_state_dict(self.target_critic_dict)

    def set_state_dict(self, new_actor_dict, new_critic_dict):
        if self.args.shared:
            self.shared_actor.load_state_dict(new_actor_dict)
            self.shared_critic.load_state_dict(new_critic_dict)
            if self.index and self.index == "global":
                self.actor.load_state_dict(new_actor_dict)
                self.critic.load_state_dict(new_critic_dict)
        else:
            self.actor.load_state_dict(new_actor_dict)
            self.critic.load_state_dict(new_critic_dict)

    def get_state_dict(self):
        return self.actor.state_dict(), self.critic.state_dict()

    def get_target_dict(self):
        return self.target_actor.state_dict(), self.target_critic.state_dict()

    def set_target_dict(self, new_actor_dict, new_critic_dict):
        self.target_actor.load_state_dict(new_actor_dict)
        self.target_critic.load_state_dict(new_critic_dict)

    def soft_update_target_networks(self, tau: float = 1.0) -> None:

        for critic_weights, target_critic_weights in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_critic_weights.data.copy_(tau * critic_weights.data + (1.0 - tau) * target_critic_weights.data)

        for actor_weights, target_actor_weights in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_actor_weights.data.copy_(tau * actor_weights.data + (1.0 - tau) * target_actor_weights.data)

    def update_target_network(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _add_exploration_noise(self, action: torch.Tensor) -> torch.Tensor:
        noise = np.random.uniform(0.0, self.noise, action.shape)
        action += torch.as_tensor(noise, dtype=torch.float32, device=device)

        return action

    def _action_scaling(self, action: torch.Tensor) -> torch.Tensor:
        neural_min = -1.0 * torch.ones_like(action)
        neural_max = 1.0 * torch.ones_like(action)

        env_min = self.min_action * torch.ones_like(action)
        env_max = self.max_action * torch.ones_like(action)

        return ((action - neural_min) / (neural_max - neural_min)) * (env_max - env_min) + env_min

    def greedy_action(self, action):
        if random.random() < self.greedy:
            self.greedy = max(self.greedy_min, self.greedy-self.e_greedy)
            return torch.tensor(random.choice(list(range(self.n_actions))))
        else:
            self.greedy = max(self.greedy_min, self.greedy - self.e_greedy)
            return torch.argmax(action, dim=-1)

    def choose_action(self, observation: np.ndarray, evaluate: bool = False) -> np.ndarray:
        self.actor.eval()

        state = torch.as_tensor(observation, dtype=torch.float32, device=device)
        action = self.actor.forward(state)

        # 分类动作使用greedy生成探索东所，连续动作使用噪声
        # action = self.greedy_action(action)

        if self.is_training:
            action = self._add_exploration_noise(action)

        action = self._action_scaling(action)

        return action.detach().cpu().numpy()

    def save_models(self, path, index) -> None:
        self.actor.save_model(path, index)
        self.target_actor.save_model(path, index)
        self.critic.save_model(path, index)
        self.target_critic.save_model(path, index)

    def load_models(self, path, index) -> None:
        self.actor.load_model(path, index)
        # self.target_actor.load_model(path, index)
        self.critic.load_model(path, index)
        # self.target_critic.load_model(path, index)

    def optimize(self, tb_summary=None, index=-1):
        if len(self.memory._storage) < self.batch_size:
            return

        beta = self.beta_scheduler.value(self.per_step)
        state, action, reward, new_state, done, weights, indices = self.memory.sample(self.batch_size, beta)

        state = torch.as_tensor(np.vstack(state), dtype=torch.float32, device=device)
        action = torch.as_tensor(np.vstack(action), dtype=torch.float32, device=device)
        done = torch.as_tensor(np.vstack(1 - done), dtype=torch.float32, device=device)
        reward = torch.as_tensor(np.vstack(reward), dtype=torch.float32, device=device)
        new_state = torch.as_tensor(np.vstack(new_state), dtype=torch.float32, device=device)
        weights = torch.as_tensor(np.hstack(weights), dtype=torch.float32, device=device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.train()
        self.actor.train()

        Q_target = self.target_critic.forward(new_state, self.target_actor.forward(new_state))
        Y = reward + (done * self.gamma * Q_target)
        Q = self.critic.forward(state, action)
        TD_errors = torch.sub(Y, Q).squeeze(dim=-1)

        weighted_TD_Errors = TD_errors * torch.sqrt(weights)
        critic_smooth_loss = smooth_l1_loss(weighted_TD_Errors, torch.zeros_like(weighted_TD_Errors))

        if self.args.shared:
            critic_coral_loss = self.coral_func(self.critic.forward(state, action), self.shared_critic.forward(state, action))
            critic_loss = critic_smooth_loss + 1e-2 * critic_coral_loss
        else:
            critic_loss = critic_smooth_loss
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        tb_summary.add_scalar("agent_{}_".format(index) + 'critic_loss', critic_loss.detach().cpu(), self.update_nums)

        # Compute & Update Actor losses
        actor_smooth_loss = torch.mean(-1.0 * self.critic.forward(state, self.actor(state)))
        if self.args.shared:
            actor_coral_loss = self.coral_func(self.actor(state), self.shared_actor(state))
            actor_loss = actor_smooth_loss + 1e-2 * actor_coral_loss
        else:
            actor_loss = actor_smooth_loss
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        tb_summary.add_scalar("agent_{}_".format(index) + 'actor_loss', actor_loss.detach().cpu(), self.update_nums)

        td_errors: np.ndarray = TD_errors.detach().cpu().numpy()
        new_priorities = np.abs(td_errors) + 1e-6
        self.memory.update_priorities(indices, new_priorities)
        self.update_nums += 1

        self.soft_update_target_networks(self.tau)


def smooth_l1_loss(input, target, reduce=True, normalizer=1.0):
    beta = 1.
    diff = torch.abs(input - target)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    if reduce:
        return torch.sum(loss) / normalizer
    return torch.sum(loss, dim=1) / normalizer

