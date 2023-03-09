import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
import torch.nn as nn
from rl.replay_buffers.PER import PrioritizedReplayBuffer
from rl.replay_buffers.utils import LinearSchedule
from rl.algorithms.PPO_model import ActorCritic
import numpy as np


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class Agent:
    def __init__(self, env, n_games, args):
        self.action_std = args.ppo_std_init

        self.gamma = args.ppo_gamma
        self.eps_clip = args.ppo_eps_clip
        self.K_epochs = args.ppo_k_epochs

        self.env = env
        self.obs_shape: int = env.observation_space
        self.n_actions: int = 2

        # self.max_size = (env._max_episode_steps * n_games * 2)
        # self.memory = PrioritizedReplayBuffer(self.max_size, 0.6)
        # self.beta_scheduler = LinearSchedule(n_games, 0.4, 0.9)

        self.buffer = RolloutBuffer()

        self.batch_size = args.ppo_batch_size

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.policy = ActorCritic(self.obs_shape, self.n_actions,
                                  args.ppo_hidden_size, self.action_std, self.device).to(self.device)

        # self.optimizer = torch.optim.Adam([
        #     {'params': self.policy.actor.parameters(), 'lr': args.ppo_lr_actor},
        #     {'params': self.policy.critic.parameters(), 'lr': args.ppo_lr_critic}
        # ])

        self.optimizer = torch.optim.Adam(self.policy.parameters())

        self.policy_old = ActorCritic(self.obs_shape, self.n_actions,
                                      args.ppo_hidden_size, self.action_std, self.device).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        self.update = 0

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if self.action_std <= min_action_std:
            self.action_std = min_action_std
            print("setting actor output action_std to min_action_std : ", self.action_std)
        else:
            print("setting actor output action_std to : ", self.action_std)

        self.set_action_std(self.action_std)

    def choose_action(self, state, evaluate=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, state_val = self.policy_old(state)
            if not evaluate:
                self.buffer.states.append(state)
                self.buffer.actions.append(action)
                self.buffer.logprobs.append(action_logprob)
                self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()

    def optimize(self, tb):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        total_loss = []

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # tb.add_scalar('loss', loss.item(), self.update)
            # self.update += 1

            total_loss.append(loss.mean().item())

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

        return sum(total_loss)/len(total_loss)

    # Save model parameters
    def save_models(self, path):
        torch.save(self.policy_old.state_dict(), os.path.join(path, "policy") + '.pth')

    # Load model parameters
    def load_models(self, path, evaluate=False):
        self.policy_old.load_state_dict(torch.load(os.path.join(path, "policy") + '.pth', map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(os.path.join(path, "policy") + '.pth', map_location=lambda storage, loc: storage))

        if evaluate:
            self.policy.eval()
            self.policy_old.eval()
        else:
            self.policy.train()
            self.policy_old.train()





