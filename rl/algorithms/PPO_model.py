
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical


class ActionHead(nn.Module):
    def __init__(self, input_dim):
        super(ActionHead, self).__init__()
        self.name2dim = {"a": 1, "steer": 1}
        self.heads = nn.ModuleDict({
            "a": nn.Linear(input_dim, 1),
            "steer": nn.Linear(input_dim, 1)
        })

    def forward(self, x):
        out = {name: self.heads[name](x) for name in self.name2dim}
        return out


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_std_init, device):
        super(ActorCritic, self).__init__()
        self.device = device
        self.action_dim = num_actions
        self.action_var = torch.full((num_actions,), action_std_init * action_std_init).to(self.device)

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, 2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

        # self.actor_linear1 = nn.Linear(num_inputs, 2*hidden_dim)
        # self.actor_linear2 = nn.Linear(2 * hidden_dim, hidden_dim)
        # self.actor_linear3 = nn.Linear(hidden_dim, num_actions)

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)

    def forward(self, state):
        action_mean = self.actor(state)
        action_mean = torch.tanh(action_mean)

        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_mean = torch.tanh(action_mean)

        # x1 = F.relu(self.actor_linear1(state))
        # x1 = F.relu(self.actor_linear2(x1))
        # x1 = F.relu(self.actor_linear3(x1))
        # action_mean = torch.tanh(x1)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)

        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy





