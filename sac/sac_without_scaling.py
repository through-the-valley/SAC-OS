import random

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import utils.rl_utils as rl_utils
import env.temp as my_env
import os
import pickle


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_log_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        log_std = self.fc_log_std(x)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # rsample()是重参数化采样
        log_prob = dist.log_prob(normal_sample)
        mt_1 = (((torch.tanh(normal_sample[:, 0]) + 1) / 2) * self.action_bound['max_mt_1']).view(-1, 1)
        mt_2 = (((torch.tanh(normal_sample[:, 1]) + 1) / 2) * self.action_bound['max_mt_2']).view(-1, 1)
        mt_3 = (((torch.tanh(normal_sample[:, 2]) + 1) / 2) * self.action_bound['max_mt_3']).view(-1, 1)
        mt_4 = (((torch.tanh(normal_sample[:, 3]) + 1) / 2) * self.action_bound['max_mt_4']).view(-1, 1)
        ess_1 = (torch.tanh(normal_sample[:, 4]) * self.action_bound['bes_1']).view(-1, 1)
        ess_2 = (torch.tanh(normal_sample[:, 5]) * self.action_bound['bes_2']).view(-1, 1)

        action = torch.cat([mt_1, mt_2, mt_3, mt_4, ess_1, ess_2], dim=1)

        return action, log_prob


class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class SACContinuous:
    ''' 处理连续动作的SAC算法 '''

    def __init__(self, actor_lr, critic_lr, alpha_lr, hidden_dim, num_episodes, batch_size):
        self.env_name = 'MicroGrid for Continuous'
        self.mode_dict = {
            'with_datetime': False,
            'r_mode': 'self_balance_plus_emission'
        }
        self.env = my_env.MicroGridForContinuous(self.mode_dict)

        self.state_dim = len(self.env.observation_space.spaces)
        self.action_dim = len(self.env.action_space.spaces)
        self.action_bound = self.env.get_action_bound()  # 动作最大值

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.num_episodes = num_episodes
        self.hidden_dim = hidden_dim
        self.gamma = 0.98
        self.tau = 0.005  # 软更新参数
        self.buffer_size = 100000
        self.minimal_size = 1000
        self.batch_size = batch_size
        self.target_entropy = - len(self.env.action_space.spaces)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
            "cpu")

        self.replay_buffer = rl_utils.ReplayBuffer(self.buffer_size)

        self.actor = PolicyNetContinuous(self.state_dim, self.hidden_dim, self.action_dim,
                                         self.action_bound).to(self.device)  # 策略网络
        self.critic_1 = QValueNetContinuous(self.state_dim, self.hidden_dim,
                                            self.action_dim).to(self.device)  # 第一个Q网络
        self.critic_2 = QValueNetContinuous(self.state_dim, self.hidden_dim,
                                            self.action_dim).to(self.device)  # 第二个Q网络
        self.target_critic_1 = QValueNetContinuous(self.state_dim,
                                                   self.hidden_dim, self.action_dim).to(
            self.device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNetContinuous(self.state_dim,
                                                   self.hidden_dim, self.action_dim).to(
            self.device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=self.critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=self.critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=self.alpha_lr)


        self.q_value_net_path = '../model/sac_without_scaling/critic/actor_lr_' + str(self.actor_lr) + '_critic_lr_' + \
                                str(self.critic_lr) + '.pth'
        self.policy_net_path = '../model/sac_without_scaling/actor/actor_lr_' + str(self.actor_lr) + '_critic_lr_' + \
                               str(self.critic_lr) + '.pth'

        self.return_list_path = '../list/sac_without_scaling/object/return_list.pickle'
        self.mv_return_list_path = '../list/sac_without_scaling/mv_object/mv_return_list.pickle'
        self.total_cost_list_path = '../list/sac_without_scaling/cost/total_cost_list.pickle'
        self.mv_total_cost_list_path = '../list/sac_without_scaling/mv_cost/mv_total_cost_list.pickle'
        self.self_balance_list_path = '../list/sac_without_scaling/self_balance/self_balance_list.pickle'
        self.mv_self_balance_list_path = '../list/sac_without_scaling/mv_self_balance/mv_self_balance_list.pickle'
        self.trade_list_path = '../list/sac_without_scaling/trade/trade_list.pickle'
        self.mv_trade_list_path = '../list/sac_without_scaling/mv_trade/trade_list.pickle'
        self.out_range_list_path = '../list/sac_without_scaling/out_range/out_range_list.pickle'
        self.p_out_range_list_path = '../list/sac_without_scaling/p_out_range/p_out_range_list.pickle'

        self.total_cost_list = []
        self.mv_total_cost_list = []
        self.episode_list = []
        self.self_balance = []
        self.mv_self_balance = []
        self.self_balance_rate = []
        self.mv_self_balance_rate = []
        self.trade_list = []
        self.mv_trade_list = []

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action, log_prob = self.actor(state)
        return action.detach()

    def calc_target(self, rewards, next_states, dones):  # 计算目标Q值
        next_actions, log_prob = self.actor(next_states)

        entropy = -log_prob
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value,
                               q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        td_target = torch.mean(td_target, dim=1, keepdim=True)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.cat(transition_dict['actions'], dim=1).view(-1, 6)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones).detach()
        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(states, actions), td_target))
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(states, actions), td_target))
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_1_loss.backward()
        critic_2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=1)
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm=1)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # 更新策略网络
        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.2)
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

    def train(self, show=True):
        return_list = []
        total_cost_list = []
        trade_list = []
        out_range_list = []
        p_out_range_list = []
        for i in range(10):
            with tqdm(total=int(self.num_episodes / 10), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(self.num_episodes / 10)):  # 每一次循环是一个回合，共num_episodes/10=50次循环，而这个的上级循环要走10次
                    episode_return = 0
                    episode_total_cost = 0
                    episode_self_balance = 0
                    episode_trade = 0
                    episode_out_range = 0
                    episode_p_out_rage = 0
                    state = self.env.reset()
                    done = 0
                    while not done:
                        action = self.take_action(state)
                        next_state, reward, done, _ = self.env.step(action)
                        self.replay_buffer.add(state, action, reward, next_state, done)
                        state = next_state
                        episode_return += sum(reward).item()
                        episode_total_cost += _['total_cost']
                        episode_trade += _['trade']
                        episode_out_range += _['out_range']
                        episode_p_out_rage += _['p_out_rage']
                        # 当buffer数据的数量超过一定值后,才进行Q网络训练
                        if self.replay_buffer.size() > self.minimal_size:
                            b_s, b_a, b_r, b_ns, b_d = self.replay_buffer.sample(self.batch_size)
                            transition_dict = {
                                'states': b_s,
                                'actions': b_a,
                                'next_states': b_ns,
                                'rewards': b_r,
                                'dones': b_d
                            }
                            self.update(transition_dict)
                    return_list.append(episode_return)
                    total_cost_list.append(episode_total_cost)
                    trade_list.append(episode_trade)
                    out_range_list.append(episode_out_range)
                    p_out_range_list.append(episode_p_out_rage)
                    if (i_episode + 1) % 10 == 0:
                        pbar.set_postfix({
                            'episode':
                                '%d' % (self.num_episodes / 10 * i + i_episode + 1),
                            'return':
                                '%.3f' % np.mean(return_list[-10:])
                        })
                    pbar.update(1)



        mv_trade_list = rl_utils.moving_average(trade_list, 9)


        with open(self.trade_list_path, 'wb') as file:
            pickle.dump(trade_list, file)
        with open(self.mv_trade_list_path, 'wb') as file:
            pickle.dump(mv_trade_list, file)
        with open(self.out_range_list_path, 'wb') as file:
            pickle.dump(out_range_list, file)
        with open(self.p_out_range_list_path, 'wb') as file:
            pickle.dump(p_out_range_list, file)


