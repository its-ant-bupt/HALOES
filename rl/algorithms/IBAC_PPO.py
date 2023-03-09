import torch
import numpy as np
from rl.algorithms.IBAC import IBACModel
import os


class RolloutBuffer:
    def __init__(self):
        self.obss = []
        self.masks = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.advantages = []
        self.log_probs = []
        self.returnn = []

    def clear(self):
        self.obss = []
        self.masks = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.advantages = []
        self.log_probs = []
        self.returnn = []


class Agent:
    def __init__(self, env, n_games, args):
        self.env = env
        self.args = args

        self.obs_shape: int = env.observation_space
        if args.action_c:
            self.n_actions: int = 2
        else:
            self.n_actions: int = len(env.action_list)

        self.batch = args.ppo_batch_size
        self.K_epochs = args.ppo_k_epochs
        self.discount = args.ppo_gamma
        self.lr = args.ibppo_lr
        self.gae_lambda = args.ibppo_gae_lambda
        self.entropy_coef = args.ibppo_entropy_coef
        self.value_loss_coef = args.ibppo_value_loss_coef
        self.max_grad_norm = args.ibppo_max_grad_norm
        self.recurrence = args.ppo_update_timestep

        self.clip_eps = args.ppo_eps_clip
        self.beta = args.ibppo_beta
        self.sni_type = args.ibppo_sni_type

        self.action_std = args.ppo_std_init

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.buffer = RolloutBuffer()

        self.acmodel = IBACModel(input_size=self.obs_shape, action_space=self.n_actions, hidden_dim=args.ppo_hidden_size,
                                 use_bottleneck=args.ibppo_use_bottleneck, dropout=args.ibppo_use_dropout,
                                 use_l2a=args.ibppo_use_l2a, use_bn=args.ibppo_use_bn, sni_type=args.ibppo_sni_type,
                                 device=self.device, action_std_init=0.6, args=args).to(self.device)

        if args.ibppo_use_l2w:
            self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr=self.lr, eps=args.ppo_adam_eps,
                                              weight_decay=self.beta)
        else:
            self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr=self.lr, eps=args.ppo_adam_eps)

        self.update = 0

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.acmodel.set_action_std(new_action_std)

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
            state = torch.FloatTensor(state).to(self.device).view(1, -1)
            dist, value, kl = self.acmodel.compute_run(state)
            if self.args.action_c:
                action = torch.clamp(dist.sample(), -1, 1)
            else:
                action = dist.sample()
            if not evaluate:
                self.buffer.obss.append(state)
                self.buffer.actions.append(action)
                self.buffer.values.append(value)
                self.buffer.log_probs.append(dist.log_prob(action))

            return action.detach().cpu().numpy().flatten()

    def operate_advantage(self):
        last_state = torch.FloatTensor(self.buffer.obss[-1].cpu()).to(self.device).view(1, -1)
        with torch.no_grad():
            _, next_value, _ = self.acmodel.compute_run(last_state)

        batch_size = len(self.buffer.obss)
        self.buffer.advantages = [0] * batch_size
        for i in reversed(range(batch_size)):
            next_mask = self.buffer.masks[i+1] if i < batch_size - 1 else self.buffer.masks[-1]
            next_value = self.buffer.values[i+1] if i < batch_size - 1 else next_value
            next_advantage = self.buffer.advantages[i+1] if i < batch_size - 1 else 0

            delta = self.buffer.rewards[i] + self.discount * next_value * next_mask - self.buffer.values[i]

            self.buffer.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # self.buffer.returnn = list(np.array(self.buffer.rewards) + np.array(self.buffer.advantages))
        self.buffer.returnn = torch.squeeze(torch.stack(self.buffer.rewards, dim=0) + torch.stack(self.buffer.advantages, dim=0)).detach().to(self.device)

        return

    def optimize(self, tb=None):
        self.operate_advantage()
        obs = torch.squeeze(torch.stack(self.buffer.obss, dim=0)).detach().to(self.device)
        action = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        value = torch.squeeze(torch.stack(self.buffer.values, dim=0)).detach().to(self.device)
        reward = torch.squeeze(torch.stack(self.buffer.rewards, dim=0)).detach().to(self.device)
        advantage = torch.squeeze(torch.stack(self.buffer.advantages, dim=0)).detach().to(self.device)
        # returnn = torch.squeeze(torch.stack(self.buffer.returnn, dim=0)).detach().to(self.device)
        returnn = self.buffer.returnn
        log_prob = torch.squeeze(torch.stack(self.buffer.log_probs, dim=0)).detach().to(self.device)

        BatchEntropy = []
        BatchValue = []
        BatchPolicyLoss = []
        BatchValueLoss = []
        BatchKl = []
        Loss = []

        for _ in range(self.K_epochs):

            if self.args.ibppo_sni_type == 'vib':
                dist_run, dist_train, value_train, kl = self.acmodel.compute_train(obs)

                entropy = (dist_run.entropy().mean() + dist_train.entropy().mean()) / 2.
                kl = kl.mean()

                ratio_r = torch.exp(dist_run.log_prob(action) - log_prob)
                ratio_t = torch.exp(dist_train.log_prob(action) - log_prob)
                surr1_r = ratio_r * advantage
                surr1_t = ratio_t * advantage

                surr2_r = torch.clamp(ratio_r, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantage
                surr2_t = torch.clamp(ratio_t, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantage

                policy_loss_r = -torch.min(surr1_r, surr2_r).mean()
                policy_loss_t = -torch.min(surr1_t, surr2_t).mean()

                policy_loss = (policy_loss_r + policy_loss_t) / 2.

                value_clipped = value + torch.clamp(value_train - value, -self.clip_eps, self.clip_eps)

                surr1 = (value_train - returnn).pow(2)
                surr2 = (value_clipped - returnn).pow(2)
                value_loss = torch.max(surr1, surr2).mean()
            else:
                # dropout
                dist, value_train, kl = self.acmodel.compute_train(obs)
                entropy = dist.entropy().mean()
                kl = kl.mean()

                ratio = torch.exp(dist.log_prob(action) - log_prob)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantage
                policy_loss = -torch.min(surr1, surr2).mean()

                value_clipped = value + torch.clamp(value_train - value, -self.clip_eps, self.clip_eps)
                surr1 = (value_train - returnn).pow(2)
                surr2 = (value_clipped - returnn).pow(2)
                value_loss = torch.max(surr1, surr2).mean()
            
            # reference: https://arxiv.org/abs/1910.12911 (IBAC-SNI) formulation (12)
            loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss + self.beta * kl

            # Update batch values

            batch_entropy = entropy.item()
            batch_value = value.mean().item()
            batch_policy_loss = policy_loss.item()
            batch_value_loss = value_loss.item()
            batch_kl = kl.item()
            batch_loss = loss

            BatchEntropy.append(batch_entropy)
            BatchValue.append(batch_value)
            BatchPolicyLoss.append(batch_policy_loss)
            BatchValueLoss.append(batch_value_loss)
            BatchKl.append(batch_kl)
            Loss.append(batch_loss.item())

            self.optimizer.zero_grad()
            batch_loss.backward()
            grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters()) ** 0.5
            torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # tb.add_scalar('entropy', batch_entropy, self.update)
            # tb.add_scalar('value', batch_value, self.update)
            # tb.add_scalar('policy_loss', batch_policy_loss, self.update)
            # tb.add_scalar('value_loss', batch_value_loss, self.update)
            # tb.add_scalar('kl', batch_kl, self.update)
            # tb.add_scalar('loss', loss.item(), self.update)
            #
            # self.update += 1
        self.buffer.clear()
        return sum(BatchEntropy)/len(BatchEntropy), sum(BatchValue)/len(BatchValue),\
               sum(BatchPolicyLoss)/len(BatchPolicyLoss), sum(BatchValueLoss)/len(BatchValueLoss),\
               sum(BatchKl)/len(BatchKl), sum(Loss)/len(Loss)

    # Save model parameters
    def save_models(self, path):
        torch.save(self.acmodel.state_dict(), os.path.join(path, "policy") + '.pth')

    # Load model parameters
    def load_models(self, path, evaluate=False):
        self.acmodel.load_state_dict(
            torch.load(os.path.join(path, "policy") + '.pth', map_location=lambda storage, loc: storage))

        if evaluate:
            self.acmodel.eval()
        else:
            self.acmodel.train()














