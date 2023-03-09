import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from rl.algorithms.bottleneck import Bottleneck
from torch.distributions import MultivariateNormal


def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class IBACModel(nn.Module):
    def __init__(self, input_size, action_space, hidden_dim, use_bottleneck=False,
                 dropout=0, use_l2a=False, use_bn=False, sni_type=None, device=None, action_std_init=None, args=None):
        super(IBACModel, self).__init__()
        # Decide which components are enabled
        self.use_bottleneck = use_bottleneck
        self.use_l2a = use_l2a
        self.dropout = dropout
        self.sni_type = sni_type
        self.args = args
        self.action_dim = action_space

        self.device = device

        self.action_var = torch.full((action_space,), action_std_init * action_std_init).to(device)

        self.embedding = nn.Sequential(
            nn.Linear(input_size, 2 * hidden_dim),
            # nn.ReLU(),
            nn.Sigmoid(),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.Sigmoid(),
            # nn.ReLU(),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.Sigmoid()
            # nn.ReLU()
        )

        if use_bottleneck:
            self.reg_layer = Bottleneck(2 * hidden_dim, hidden_dim)
        else:
            self.reg_layer = nn.Linear(2 * hidden_dim, hidden_dim)

        self.dropout_layer = nn.Dropout(p=self.dropout, inplace=False)
        if args.res_net:
            # 将观测信息添加进处理好的信息中
            self.actor = nn.Sequential(
                nn.Linear(hidden_dim+input_size, 2*hidden_dim),
                nn.ELU(),
                nn.Linear(2*hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, action_space),
                nn.Tanh()
            )
            self.critic = nn.Sequential(
                nn.Linear(hidden_dim+input_size, 2*hidden_dim),
                nn.ELU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(hidden_dim, action_space),
                nn.Tanh()
            )
            self.critic = nn.Sequential(
                nn.Linear(hidden_dim, 1)
            )

        # self.critic = nn.Sequential(
        #     nn.Linear(hidden_dim, 1),
        #     nn.Tanh()
        # )


        self.apply(initialize_parameters)

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)

    def encode(self, obs):
        embedding = self.embedding(obs)

        if self.use_bottleneck:
            bot_mean, bot, kl = self.reg_layer(embedding)
            kl = torch.sum(kl, dim=1)
        elif self.use_l2a:
            bot_mean = bot = self.reg_layer(embedding)
            kl = torch.sum(bot ** 2, dim=1)
        else:
            bot_mean = self.reg_layer(embedding)
            bot = self.dropout_layer(bot_mean)
            kl = torch.Tensor([0])

        return bot_mean, bot, kl

    def compute_run(self, obs):
        bot_mean, bot, kl = self.encode(obs)

        if self.sni_type is not None:
            # For any SNI type, the rollouts values are deterministic
            if self.args.res_net:
                x_in = torch.cat([bot_mean, obs], dim=-1).to(self.device)
            else:
                x_in = bot_mean
            x_dist = self.actor(x_in)
            # dist = Categorical(logits=F.log_softmax(x_dist, dim=1))
            value = self.critic(x_in).squeeze(1)
        else:
            if self.args.res_net:
                x_in = torch.cat([bot, obs], dim=-1).to(self.device)
            else:
                x_in = bot
            x_dist = self.actor(x_in)
            # dist = Categorical(logits=F.log_softmax(x_dist, dim=1))
            value = self.critic(x_in).squeeze(1)

        if self.args.action_c:
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(x_dist, cov_mat)
        else:
            dist = Categorical(logits=F.log_softmax(x_dist, dim=1))

        return dist, value, kl

    def compute_train(self, obs):
        bot_mean, bot, kl = self.encode(obs)

        if self.sni_type == 'vib':
            # Need both policies for training, but still only one value function:
            if self.args.res_net:
                x_in_run = torch.cat([bot_mean, obs], dim=-1).to(self.device)
                x_in_train = torch.cat([bot, obs], dim=-1).to(self.device)
            else:
                x_in_run = bot_mean
                x_in_train = bot
            x_dist_run = self.actor(x_in_run)
            if self.args.action_c:
                action_var = self.action_var.expand_as(x_dist_run)
                cov_mat = torch.diag_embed(action_var).to(self.device)
                dist_run = MultivariateNormal(x_dist_run, cov_mat)
            else:
                dist_run = Categorical(logits=F.log_softmax(x_dist_run, dim=1))
            value = self.critic(x_in_run).squeeze(1)

            x_dist_train = self.actor(x_in_train)
            if self.args.action_c:
                action_var = self.action_var.expand_as(x_dist_train)
                cov_mat = torch.diag_embed(action_var).to(self.device)
                dist_train = MultivariateNormal(x_dist_train, cov_mat)
            else:
                dist_train = Categorical(logits=F.log_softmax(x_dist_train, dim=1))

            return dist_run, dist_train, value, kl
        elif self.sni_type == 'dropout' or self.sni_type is None:
            # Random policy AND value function
            if self.args.res_net:
                x_in_train = torch.cat([bot, obs], dim=-1).to(self.device)
            else:
                x_in_train = bot
            x_dist_train = self.actor(x_in_train)
            if self.args.action_c:
                action_var = self.action_var.expand_as(x_dist_train)
                cov_mat = torch.diag_embed(action_var).to(self.device)
                dist_train = MultivariateNormal(x_dist_train, cov_mat)
            else:
                dist_train = Categorical(logits=F.log_softmax(x_dist_train, dim=1))
            value = self.critic(x_in_train).squeeze(1)
            return dist_train, value, kl



