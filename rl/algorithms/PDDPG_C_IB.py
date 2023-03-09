# Library Imports
import numpy as np
import os

from gym import Env
from rl.environment.env import Environment

import torch
from torch.nn import functional as F
import torch.nn as nn
import random

from rl.replay_buffers.PER import PrioritizedReplayBufferIB
from rl.replay_buffers.utils import LinearSchedule
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.distributions.categorical import Categorical
from torch.distributions import MultivariateNormal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def xavier_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        print("Initiating bottleneck")
        nn.init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))
        m.bias.data.zero_()


class BottleNeckEncoder(nn.Module):
    #Define the Information Bottleneck for DDPG
    def __init__(self, input_size, output_size, hidden_dim, name="bottleneck"):
        super(BottleNeckEncoder, self).__init__()
        self.output_size = output_size
        self.checkpoint = name

        self.embedding = nn.Sequential(
            nn.Linear(input_size, 2 * hidden_dim),
            # nn.ReLU(),
            nn.ELU(),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.ELU(),
            # nn.ReLU(),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.ELU()
            # nn.ReLU()
        )

        self.encode = nn.Linear(2 * hidden_dim, 2 * output_size)
        # self.weight_init()

    def forward(self, x):
        device = x.device
        embedding = self.embedding(x)
        stats = self.encode(embedding)

        mu = stats[:, :self.output_size]
        std = F.softplus(stats[:, self.output_size:])

        # if self.noisy_prior:
        #     prior_0 = Normal(torch.zeros(self.output_size).to(device), torch.ones(self.output_size).to(device) / math.sqrt(2.))
        #     prior_mu = prior_0.sample()
        #     prior = Normal(prior_mu, torch.ones(self.output_size).to(device) / math.sqrt(2.))
        prior = Normal(torch.zeros(self.output_size).to(device), torch.ones(self.output_size).to(device))

        dist = Normal(mu, std)
        kl = kl_divergence(dist, prior)
        kl = torch.sum(kl, dim=1)

        return mu, dist.rsample(), kl

    def weight_init(self):
        for m in self._modules:
            xavier_init(self._modules[m])

    def save_model(self, path):
        torch.save(self.state_dict(), os.path.join(path, self.checkpoint) + '.pth')

    def load_model(self, path):
        self.load_state_dict(torch.load(os.path.join(path, self.checkpoint) + '.pth'))


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

    def forward(self, _value) -> torch.Tensor:
        # value = torch.hstack((state, action))
        value = F.relu(self.H1(_value))
        value = F.relu(self.H2(value))
        value = self.drop(value)
        value = F.relu(self.H3(value))
        value = F.relu(self.H4(value))
        value = self.Q(value)
        return value

    def save_model(self, path):
        torch.save(self.state_dict(), os.path.join(path, self.checkpoint) + '.pth')

    def load_model(self, path):
        self.load_state_dict(torch.load(os.path.join(path, self.checkpoint) + '.pth'))


class Actor(torch.nn.Module):
    """Defines a Actor Deep Learning Network"""

    def __init__(self, input_dim: int, n_actions: int, alpha: float, density: int = 512, name='actor', args=None):
        super(Actor, self).__init__()

        self.model_name = name
        self.checkpoint = self.model_name
        self.args = args

        # Bottleneck
        self.bottleneck = BottleNeckEncoder(input_dim, args.ddpg_hidden, args.ddpg_hidden).to(device)

        # Architecture
        self.H1 = torch.nn.Linear(args.ddpg_hidden+input_dim, density)
        self.H2 = torch.nn.Linear(density, density)
        self.drop = torch.nn.Dropout(p=0.1)
        self.H3 = torch.nn.Linear(density, density)
        self.H4 = torch.nn.Linear(density, density)
        self.mu = torch.nn.Linear(density, n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = device
        self.to(device)

    def forward(self, state: torch.Tensor, train=False):
        # ============ BottleNeck =================
        bot_mean, bot, kl = self.bottleneck(state)
        # =========================================
        if self.args.res_net:
            x_in = torch.cat([bot_mean, state], dim=-1).to(device)
            if train:
                x_in_train = torch.cat([bot, state], dim=-1).to(device)
        else:
            x_in = bot_mean
            if train:
                x_in_train = bot
        
        if train:
            value = F.relu(self.H1(x_in))
            value = F.relu(self.H2(value))
            value = self.drop(value)
            value = F.relu(self.H3(value))
            value = F.relu(self.H4(value))
            action = torch.tanh(self.mu(value))
            # action = torch.softmax(action, dim=-1)

            train_value = F.relu(self.H1(x_in_train))
            train_value = F.relu(self.H2(train_value))
            train_value = self.drop(train_value)
            train_value = F.relu(self.H3(train_value))
            train_value = F.relu(self.H4(train_value))
            train_action = torch.tanh(self.mu(train_value))
            return action, train_action, kl
        else:
            value = F.relu(self.H1(x_in))
            value = F.relu(self.H2(value))
            value = self.drop(value)
            value = F.relu(self.H3(value))
            value = F.relu(self.H4(value))
            action = torch.tanh(self.mu(value))
            # action = torch.softmax(action, dim=-1)
            return action

    def save_model(self, path):
        torch.save(self.state_dict(), os.path.join(path, self.checkpoint) + '.pth')

    def load_model(self, path):
        self.load_state_dict(torch.load(os.path.join(path, self.checkpoint) + '.pth'))


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
                 args=None):

        self.gamma = torch.tensor(gamma, dtype=torch.float32, device=device)
        self.tau = tau
        self.n_games = n_games
        self.env = env
        self.greedy = 0.9
        self.e_greedy = 0.0001
        self.greedy_min = 0.01
        self.args = args

        self.max_size: int = (env._max_episode_steps * n_games * 2)
        self.memory = PrioritizedReplayBufferIB(self.max_size, 0.6)
        self.beta_scheduler = LinearSchedule(n_games, 0.4, 0.9)
        self.batch_size = batch_size

        self.obs_shape: int = env.observation_space
        self.n_actions: int = 2

        self.max_action = torch.as_tensor(1, dtype=torch.float32, device=device)
        self.min_action = torch.as_tensor(-1, dtype=torch.float32, device=device)

        # self.bottleneck = BottleNeckEncoder(self.obs_shape, args.ddpg_hidden, args.ddpg_hidden).to(device)
        self.actor = Actor(self.obs_shape, self.n_actions, alpha, name='actor', args=args)
        self.critic = Critic(self.obs_shape + self.n_actions, beta, name='critic')
        self.target_actor = Actor(self.obs_shape, self.n_actions, alpha, name='target_actor', args=args)
        self.target_critic = Critic(self.obs_shape + self.n_actions, beta, name='target_critic')

        self.action_std = args.ddpg_std_init
        self.action_var = torch.full((self.n_actions,), args.ddpg_std_init * args.ddpg_std_init).to(device)

        self.is_training = training
        self.noise = noise

        self.per_step: int = 0

        self.update_nums = 0

        self._update_networks()

    def _update_networks(self, tau: float = 1.0) -> None:

        for critic_weights, target_critic_weights in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_critic_weights.data.copy_(tau * critic_weights.data + (1.0 - tau) * target_critic_weights.data)

        for actor_weights, target_actor_weights in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_actor_weights.data.copy_(tau * actor_weights.data + (1.0 - tau) * target_actor_weights.data)

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

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if self.action_std <= min_action_std:
            self.action_std = min_action_std
            print("setting actor output action_std to min_action_std : ", self.action_std)
        else:
            print("setting actor output action_std to : ", self.action_std)

        self.set_action_std(self.action_std)

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.n_actions,), new_action_std * new_action_std).to(device)

    def choose_action(self, observation: np.ndarray, evaluate: bool = False):
        self.actor.eval()

        state = torch.as_tensor(observation, dtype=torch.float32, device=device).view(1, -1)

        x_dist = self.actor.forward(state)
        if self.args.action_c:
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(x_dist, cov_mat)
        else:
            dist = Categorical(logits=F.log_softmax(x_dist, dim=1))

        sample = dist.sample()
        action = torch.clamp(sample, -1, 1)

        # value = self.critic.forward(torch.cat([x_in, action], dim=-1).to(device))

        # 分类动作使用greedy生成探索东所，连续动作使用噪声
        # action = self.greedy_action(action)

        if self.is_training:
            action = self._add_exploration_noise(action)

        action = self._action_scaling(action)

        return action.detach().cpu().numpy().flatten(), dist.log_prob(action).detach().cpu().numpy()

    def save_models(self, path) -> None:
        self.actor.save_model(path)
        self.target_actor.save_model(path)
        self.critic.save_model(path)
        self.target_critic.save_model(path)

    def load_models(self, path) -> None:
        self.actor.load_model(path)
        self.target_actor.load_model(path)
        self.critic.load_model(path)
        self.target_critic.load_model(path)

    def optimize(self, tb_summary=None):
        if len(self.memory._storage) < self.batch_size:
            return

        beta = self.beta_scheduler.value(self.per_step)
        state, action, reward, log_prob, new_state, done, weights, indices = self.memory.sample(self.batch_size, beta)

        state = torch.as_tensor(np.vstack(state), dtype=torch.float32, device=device)
        action = torch.as_tensor(np.vstack(action), dtype=torch.float32, device=device)
        done = torch.as_tensor(np.vstack(1 - done), dtype=torch.float32, device=device)
        reward = torch.as_tensor(np.vstack(reward), dtype=torch.float32, device=device)
        log_prob = torch.as_tensor(np.vstack(log_prob), dtype=torch.float32, device=device)
        new_state = torch.as_tensor(np.vstack(new_state), dtype=torch.float32, device=device)
        weights = torch.as_tensor(np.hstack(weights), dtype=torch.float32, device=device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.train()
        self.actor.train()

        x_dist_run, x_dist_train, kl = self.actor(state, True)
        new_x_dist_run, new_x_dist_train, kl_train = self.target_actor(new_state, True)

        if self.args.action_c:
            action_var = self.action_var.expand_as(x_dist_run)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist_run = MultivariateNormal(x_dist_run, cov_mat)
            new_dist_run = MultivariateNormal(new_x_dist_run, cov_mat)
            action_var = self.action_var.expand_as(x_dist_train)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist_train = MultivariateNormal(x_dist_train, cov_mat)
            new_dist_train = MultivariateNormal(new_x_dist_train, cov_mat)
        else:
            dist_run = Categorical(logits=F.log_softmax(x_dist_run, dim=1))
            dist_train = Categorical(logits=F.log_softmax(x_dist_train, dim=1))
            new_dist_run = Categorical(logits=F.log_softmax(new_x_dist_run, dim=1))
            new_dist_train = Categorical(logits=F.log_softmax(new_x_dist_train, dim=1))

        action_run = dist_run.sample()
        action_train = dist_train.sample()
        new_action_run = dist_run.sample()
        new_action_train = dist_train.sample()

        Q_target = self.target_critic.forward(torch.cat([new_state, new_action_run], dim=-1).to(device))
        Y = reward + (done * self.gamma * Q_target)
        Q = self.critic.forward(torch.cat([state, action], dim=-1).to(device))
        TD_errors = torch.sub(Y, Q).squeeze(dim=-1)

        weighted_TD_Errors = TD_errors * torch.sqrt(weights)
        critic_loss = smooth_l1_loss(weighted_TD_Errors, torch.zeros_like(weighted_TD_Errors))

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        tb_summary.add_scalar('critic_loss', critic_loss.detach().cpu(), self.update_nums)

        # ================== Compute & Update Actor losses ===================
        kl = kl.mean()

        ratio_r = torch.exp(dist_run.log_prob(action) - log_prob)
        ratio_t = torch.exp(dist_train.log_prob(action) - log_prob)
        surr1_r = ratio_r
        surr1_t = ratio_t

        surr2_r = torch.clamp(ratio_r, 1.0 - 0.2, 1.0 + 0.2)
        surr2_t = torch.clamp(ratio_t, 1.0 - 0.2, 1.0 + 0.2)

        policy_loss_r = -torch.min(surr1_r, surr2_r).mean()
        policy_loss_t = -torch.min(surr1_t, surr2_t).mean()

        policy_loss = (policy_loss_r + policy_loss_t) / 2.

        actor_loss = torch.mean(-1.0 * self.critic.forward(torch.cat([state, action_run], dim=-1).to(device)))

        total_actor_loss = policy_loss + actor_loss + 0.5 * kl
        self.actor.optimizer.zero_grad()

        total_actor_loss.backward()
        self.actor.optimizer.step()

        tb_summary.add_scalar('actor_loss', actor_loss.detach().cpu(), self.update_nums)

        td_errors: np.ndarray = TD_errors.detach().cpu().numpy()
        new_priorities = np.abs(td_errors) + 1e-6
        self.memory.update_priorities(indices, new_priorities)

        self._update_networks(self.tau)


def smooth_l1_loss(input, target, reduce=True, normalizer=1.0):
    beta = 1.
    diff = torch.abs(input - target)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    if reduce:
        return torch.sum(loss) / normalizer
    return torch.sum(loss, dim=1) / normalizer

