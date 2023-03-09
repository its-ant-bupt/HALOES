import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from rl.algorithms.SAC_model import GaussianPolicy, QNetwork, DeterministicPolicy
from rl.replay_buffers.PER import PrioritizedReplayBuffer
from rl.replay_buffers.utils import LinearSchedule
import numpy as np


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class Agent(object):
    def __init__(self, env, n_games, args):

        self.gamma = args.sac_gamma
        self.tau = args.sac_tau
        self.alpha = args.sac_alpha
        
        self.env = env
        self.obs_shape: int = env.observation_space
        self.n_actions: int = 2

        self.max_size: int = (env._max_episode_steps * n_games * 2)
        self.memory = PrioritizedReplayBuffer(self.max_size, 0.6)
        self.beta_scheduler = LinearSchedule(n_games, 0.4, 0.9)
        self.batch_size = args.sac_batch_size

        self.per_step: int = 0

        self.update = 0

        self.policy_type = args.sac_policy
        self.target_update_interval = args.sac_target_update_interval
        self.automatic_entropy_tuning = args.sac_automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critic = QNetwork(self.obs_shape, self.n_actions, args.sac_hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.sac_lr)

        self.critic_target = QNetwork(self.obs_shape, self.n_actions, args.sac_hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor([self.n_actions]).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.sac_lr)

            self.policy = GaussianPolicy(self.obs_shape, self.n_actions, args.sac_hidden_size, None).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.sac_lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(self.obs_shape, self.n_actions, args.sac_hidden_size, None).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def choose_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def optimize(self, tb, index=-1):
        # Sample a batch from memory
        if len(self.memory._storage) < self.batch_size:
            return
        beta = self.beta_scheduler.value(self.per_step)
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch, weights, indices = self.memory.sample(batch_size=self.batch_size, beta=beta)


        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(np.vstack(1 - mask_batch)).to(self.device)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        if self.update % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        tb.add_scalar('loss/critic_1', qf1_loss.item(), self.update)
        tb.add_scalar('loss/critic_2', qf2_loss.item(), self.update)
        tb.add_scalar('loss/policy', policy_loss.item(), self.update)
        tb.add_scalar('loss/entropy_loss', alpha_loss.item(), self.update)
        tb.add_scalar('entropy_temprature/alpha', alpha_tlogs.item(), self.update)
        self.update += 1

        return

    # Save model parameters
    def save_models(self, path):

        torch.save(self.policy.state_dict(), os.path.join(path, "policy") + '.pth')
        torch.save(self.critic.state_dict(), os.path.join(path, "critic") + '.pth')
        torch.save(self.critic_target.state_dict(), os.path.join(path, "critic_target") + '.pth')
        torch.save(self.critic_optim.state_dict(), os.path.join(path, "critic_optim") + '.pth')
        torch.save(self.policy_optim.state_dict(), os.path.join(path, "policy_optim") + '.pth')

    # Load model parameters
    def load_models(self, path, evaluate=False):
        self.policy.load_state_dict(torch.load(os.path.join(path, "policy") + '.pth'))
        self.critic.load_state_dict(torch.load(os.path.join(path, "critic") + '.pth'))
        self.critic_target.load_state_dict(torch.load(os.path.join(path, "critic_target") + '.pth'))
        self.critic_optim.load_state_dict(torch.load(os.path.join(path, "critic_optim") + '.pth'))
        self.policy_optim.load_state_dict(torch.load(os.path.join(path, "policy_optim") + '.pth'))

        if evaluate:
            self.policy.eval()
            self.critic.eval()
            self.critic_target.eval()
        else:
            self.policy.train()
            self.critic.train()
            self.critic_target.train()