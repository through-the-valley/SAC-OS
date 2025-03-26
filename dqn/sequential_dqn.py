import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import utils.rl_utils as rl_utils
import env.microgrid_for_dqn as my_env
import pickle


class Q_U_Net(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim_1, hidden_dim_2, action_dim):
        super(Q_U_Net, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim_1)
        self.fc2 = torch.nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = torch.nn.Linear(hidden_dim_2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Q_L_1_Net(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim_1, hidden_dim_2, action_dim):
        super(Q_L_1_Net, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim_1)
        self.fc2 = torch.nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = torch.nn.Linear(hidden_dim_2, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Q_L_2_Net(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim_1, hidden_dim_2, action_dim):
        super(Q_L_2_Net, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + 1, hidden_dim_1)
        self.fc2 = torch.nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = torch.nn.Linear(hidden_dim_2, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Q_L_3_Net(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim_1, hidden_dim_2, action_dim):
        super(Q_L_3_Net, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + 2, hidden_dim_1)
        self.fc2 = torch.nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = torch.nn.Linear(hidden_dim_2, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Q_L_4_Net(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim_1, hidden_dim_2, action_dim):
        super(Q_L_4_Net, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + 3, hidden_dim_1)
        self.fc2 = torch.nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = torch.nn.Linear(hidden_dim_2, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Q_L_5_Net(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim_1, hidden_dim_2, action_dim):
        super(Q_L_5_Net, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + 4, hidden_dim_1)
        self.fc2 = torch.nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = torch.nn.Linear(hidden_dim_2, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Q_L_6_Net(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim_1, hidden_dim_2, action_dim):
        super(Q_L_6_Net, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + 5, hidden_dim_1)
        self.fc2 = torch.nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = torch.nn.Linear(hidden_dim_2, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DoubleSDQNAgent:
    def __init__(self, discrete_bins, batch_size, hidden_dim, lr):
        '''
        :param action_dim: 离散化的自由度，其分为多少个可能的取值
        :param learning_rate: q net的学习率
        :param gamma: 计算target时的超参数
        :param epsilon: e贪婪中的超参数
        :param f_update_target: 目标网络更新的频率
        :param f_train: 存一次后，训练多少次
        :param device:
        '''

        self.mode_dict = {
            'with_datetime': False
        }
        self.discrete_bins = discrete_bins
        self.env = my_env.MicroGridForDQN(self.discrete_bins, self.mode_dict)

        self.tau = 0.005  # 软更新参数
        self.num_episodes = 500
        self.buffer_size = 100000
        self.minimal_size = 1000
        self.batch_size = batch_size
        self.env_name = 'MicroGrid for DQN'

        self.hidden_dim_1 = hidden_dim
        self.hidden_dim_2 = hidden_dim
        self.state_dim = len(self.env.observation_space.spaces)
        self.action_dim = len(self.env.action_space.spaces)
        self.epsilon = 0.05
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
            "cpu")
        self.gamma = 0.98
        self.lr = lr

        self.replay_buffer = rl_utils.ReplayBuffer(self.buffer_size)

        self.q_u_net = Q_U_Net(self.state_dim, self.hidden_dim_1, self.hidden_dim_2, self.action_dim).to(self.device)
        self.target_q_u_net = Q_U_Net(self.state_dim, self.hidden_dim_1, self.hidden_dim_2, self.action_dim).to(
            self.device)
        self.q_l_1_net = Q_L_1_Net(self.state_dim, self.hidden_dim_1, self.hidden_dim_2, self.action_dim).to(
            self.device)
        self.target_q_l_1_net = Q_L_1_Net(self.state_dim, self.hidden_dim_1, self.hidden_dim_2, self.action_dim).to(
            self.device)
        self.q_l_2_net = Q_L_2_Net(self.state_dim, self.hidden_dim_1, self.hidden_dim_2, self.action_dim).to(
            self.device)
        self.target_q_l_2_net = Q_L_2_Net(self.state_dim, self.hidden_dim_1, self.hidden_dim_2, self.action_dim).to(
            self.device)
        self.q_l_3_net = Q_L_3_Net(self.state_dim, self.hidden_dim_1, self.hidden_dim_2, self.action_dim).to(
            self.device)
        self.target_q_l_3_net = Q_L_3_Net(self.state_dim, self.hidden_dim_1, self.hidden_dim_2, self.action_dim).to(
            self.device)
        self.q_l_4_net = Q_L_4_Net(self.state_dim, self.hidden_dim_1, self.hidden_dim_2, self.action_dim).to(
            self.device)
        self.target_q_l_4_net = Q_L_4_Net(self.state_dim, self.hidden_dim_1, self.hidden_dim_2, self.action_dim).to(
            self.device)
        self.q_l_5_net = Q_L_5_Net(self.state_dim, self.hidden_dim_1, self.hidden_dim_2, self.action_dim).to(
            self.device)
        self.target_q_l_5_net = Q_L_5_Net(self.state_dim, self.hidden_dim_1, self.hidden_dim_2, self.action_dim).to(
            self.device)
        self.q_l_6_net = Q_L_6_Net(self.state_dim, self.hidden_dim_1, self.hidden_dim_2, self.action_dim).to(
            self.device)
        self.target_q_l_6_net = Q_L_6_Net(self.state_dim, self.hidden_dim_1, self.hidden_dim_2, self.action_dim).to(
            self.device)

        self.q_u_optimizer = torch.optim.Adam(self.q_u_net.parameters(), lr=self.lr)
        self.q_l_1_optimizer = torch.optim.Adam(self.q_l_1_net.parameters(), lr=self.lr)
        self.q_l_2_optimizer = torch.optim.Adam(self.q_l_2_net.parameters(), lr=self.lr)
        self.q_l_3_optimizer = torch.optim.Adam(self.q_l_3_net.parameters(), lr=self.lr)
        self.q_l_4_optimizer = torch.optim.Adam(self.q_l_4_net.parameters(), lr=self.lr)
        self.q_l_5_optimizer = torch.optim.Adam(self.q_l_5_net.parameters(), lr=self.lr)
        self.q_l_6_optimizer = torch.optim.Adam(self.q_l_6_net.parameters(), lr=self.lr)


        self.return_list_path = '../list/dqn/return_list.pickle'
        self.mv_return_list_path = '../list/dqn/mv_return_list.pickle'
        self.total_cost_list_path = '../list/dqn/total_cost_list.pickle'
        self.mv_total_cost_list_path = '../list/dqn/mv_total_cost_list.pickle'
        self.self_balance_list_path = '../list/dqn/self_balance_rate_list.pickle'
        self.mv_self_balance_list_path = '../list/dqn/mv_self_balance_rate_list.pickle'
        self.trade_list_path = '../list/dqn/trade_list.pickle'
        self.mv_trade_list_path = '../list/dqn/trade_list.pickle'

    def non_epsilon_take_action(self, state, source='sample'):
        if source == 'sample':
            state = torch.tensor([state], dtype=torch.float).to(self.device)
        else:
            test = 0
        q_1 = self.q_l_1_net(state)
        mt_1 = q_1.argmax(dim=1).view(-1, 1)
        if source == 'sample':
            temp = torch.cat([state, mt_1], dim=1).view(1, -1)
        else:
            temp = torch.cat([state, mt_1], dim=1)

        q_2 = self.q_l_2_net(temp)
        mt_2 = q_2.argmax(dim=1).view(-1, 1)
        if source == 'sample':
            temp = torch.cat([temp, mt_2], dim=1).view(1, -1)
        else:
            temp = torch.cat([temp, mt_2], dim=1)

        q_3 = self.q_l_3_net(temp)
        mt_3 = q_3.argmax(dim=1).view(-1, 1)
        if source == 'sample':
            temp = torch.cat([temp, mt_3], dim=1).view(1, -1)
        else:
            temp = torch.cat([temp, mt_3], dim=1)

        q_4 = self.q_l_4_net(temp)
        mt_4 = q_4.argmax(dim=1).view(-1, 1)
        if source == 'sample':
            temp = torch.cat([temp, mt_4], dim=1).view(1, -1)
        else:
            temp = torch.cat([temp, mt_4], dim=1)

        q_5 = self.q_l_5_net(temp)
        ess_1 = q_5.argmax(dim=1).view(-1, 1)
        if source == 'sample':
            temp = torch.cat([temp, ess_1], dim=1).view(1, -1)
        else:
            temp = torch.cat([temp, ess_1], dim=1)

        q_6 = self.q_l_6_net(temp)
        ess_2 = q_6.argmax(dim=1).view(-1, 1)
        if source == 'sample':
            temp = torch.cat([temp, ess_2], dim=1).view(1, -1)
        else:
            temp = torch.cat([temp, ess_2], dim=1)

        action = torch.cat([mt_1, mt_2, mt_3, mt_4, ess_1, ess_2], dim=1)

        return action

    def epsilon_take_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.action_dim, (1, 6))
            action = torch.from_numpy(action).to(self.device)
        else:
            action = self.non_epsilon_take_action(state)
        return action

    def optim_update(self, optim, loss):
        optim.zero_grad()
        loss.backward(retain_graph=True)
        optim.step()

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        # actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).view(-1, 6).to(self.device)
        actions = torch.cat(transition_dict['actions'], dim=1).view(-1, 6)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        mt_1, mt_2, mt_3, mt_4, ess_1, ess_2 = actions[:, 0].view(-1, 1), actions[:, 1].view(-1, 1), actions[:, 2].view(
            -1, 1) \
            , actions[:, 3].view(-1, 1), actions[:, 4].view(-1, 1), actions[:, 5].view(-1, 1)

        temp1, temp2, temp3, temp4, temp5, temp6 = torch.cat([states, mt_1], dim=1), torch.cat([states, mt_1, mt_2],
                                                                                               dim=1), \
            torch.cat([states, mt_1, mt_2, mt_3], dim=1), torch.cat([states, mt_1, mt_2, mt_3, mt_4], dim=1), \
            torch.cat([states, mt_1, mt_2, mt_3, mt_4, ess_1], dim=1), torch.cat(
            [states, mt_1, mt_2, mt_3, mt_4, ess_1, ess_2], dim=1)

        q_u = self.q_u_net(temp6)
        next_a = self.non_epsilon_take_action(next_states, source='train')
        max_target_q = self.target_q_u_net(torch.cat([next_states, next_a], dim=1))
        target_value = rewards + self.gamma * max_target_q * (1 - dones)
        q_u_net_loss = F.mse_loss(q_u, target_value)
        q_u_for_q_l_6 = self.q_u_net(temp6)

        q_l_1 = self.q_l_1_net(states)
        q_l_1_a = q_l_1.gather(1, mt_1)
        q_l_2 = self.q_l_2_net(temp1)
        q_l_2_a = q_l_2.gather(1, mt_2)
        q_l_3 = self.q_l_3_net(temp2)
        q_l_3_a = q_l_3.gather(1, mt_3)
        q_l_4 = self.q_l_4_net(temp3)
        q_l_4_a = q_l_4.gather(1, mt_4)
        q_l_5 = self.q_l_5_net(temp4)
        q_l_5_a = q_l_5.gather(1, ess_1)
        q_l_6 = self.q_l_6_net(temp5)
        q_l_6_a = q_l_6.gather(1, ess_2)

        max_q_l_2 = self.target_q_l_2_net(temp1).max(dim=1)[0].view(-1, 1)
        max_q_l_3 = self.target_q_l_3_net(temp2).max(dim=1)[0].view(-1, 1)
        max_q_l_4 = self.target_q_l_4_net(temp3).max(dim=1)[0].view(-1, 1)
        max_q_l_5 = self.target_q_l_5_net(temp4).max(dim=1)[0].view(-1, 1)
        max_q_l_6 = self.target_q_l_6_net(temp5).max(dim=1)[0].view(-1, 1)

        q_l_1_net_loss = F.mse_loss(q_l_1_a, max_q_l_2)
        q_l_2_net_loss = F.mse_loss(q_l_2_a, max_q_l_3)
        q_l_3_net_loss = F.mse_loss(q_l_3_a, max_q_l_4)
        q_l_4_net_loss = F.mse_loss(q_l_4_a, max_q_l_5)
        q_l_5_net_loss = F.mse_loss(q_l_5_a, max_q_l_6)
        q_l_6_net_loss = F.mse_loss(q_l_6_a, q_u_for_q_l_6)

        self.q_u_optimizer.zero_grad()
        self.q_l_1_optimizer.zero_grad()
        self.q_l_2_optimizer.zero_grad()
        self.q_l_3_optimizer.zero_grad()
        self.q_l_4_optimizer.zero_grad()
        self.q_l_5_optimizer.zero_grad()
        self.q_l_6_optimizer.zero_grad()
        q_u_net_loss.backward()
        q_l_1_net_loss.backward()
        q_l_2_net_loss.backward()
        q_l_3_net_loss.backward()
        q_l_4_net_loss.backward()
        q_l_5_net_loss.backward()
        q_l_6_net_loss.backward()
        self.q_u_optimizer.step()
        self.q_l_1_optimizer.step()
        self.q_l_2_optimizer.step()
        self.q_l_3_optimizer.step()
        self.q_l_4_optimizer.step()
        self.q_l_5_optimizer.step()
        self.q_l_6_optimizer.step()

        self.soft_update(self.q_u_net, self.target_q_u_net)
        self.soft_update(self.q_l_1_net, self.target_q_l_1_net)
        self.soft_update(self.q_l_2_net, self.target_q_l_2_net)
        self.soft_update(self.q_l_3_net, self.target_q_l_3_net)
        self.soft_update(self.q_l_4_net, self.target_q_l_4_net)
        self.soft_update(self.q_l_5_net, self.target_q_l_5_net)
        self.soft_update(self.q_l_6_net, self.target_q_l_6_net)

    def train(self, show=True):
        return_list = []
        total_cost_list = []
        self_balance_list = []
        trade_list = []
        for i in range(10):
            with tqdm(total=int(self.num_episodes / 10), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(self.num_episodes / 10)):  # 每一次循环是一个回合，共num_episodes/10=50次循环，而这个的上级循环要走10次
                    episode_return = 0
                    episode_total_cost = 0
                    episode_self_balance = 0
                    episode_trade = 0
                    state = self.env.reset()
                    done = 0
                    while not done:
                        action = self.epsilon_take_action(state)
                        next_state, reward, done, _ = self.env.step(action)
                        self.replay_buffer.add(state, action, reward, next_state, done)
                        state = next_state
                        episode_return += reward.item()
                        episode_total_cost += _['total_cost']
                        episode_self_balance += _['r_self_balance']
                        episode_trade += _['trade']
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
                    self_balance_list.append(episode_self_balance)
                    trade_list.append(episode_trade)
                    if (i_episode + 1) % 10 == 0:
                        pbar.set_postfix({
                            'episode':
                                '%d' % (self.num_episodes / 10 * i + i_episode + 1),
                            'return':
                                '%.3f' % np.mean(return_list[-10:])
                        })
                    pbar.update(1)


        mv_return = rl_utils.moving_average(return_list, 9)

        mv_costs = rl_utils.moving_average(total_cost_list, 9)
        mv_self_balance_list = rl_utils.moving_average(self_balance_list, 9)
        mv_trade_list = rl_utils.moving_average(trade_list, 9)

        with open(self.return_list_path, 'wb') as file:
            pickle.dump(return_list, file)
        with open(self.mv_return_list_path, 'wb') as file:
            pickle.dump(mv_return, file)
        with open(self.total_cost_list_path, 'wb') as file:
            pickle.dump(total_cost_list, file)
        with open(self.mv_total_cost_list_path, 'wb') as file:
            pickle.dump(mv_costs, file)
        with open(self.self_balance_list_path, 'wb') as file:
            pickle.dump(self_balance_list, file)
        with open(self.mv_self_balance_list_path, 'wb') as file:
            pickle.dump(mv_self_balance_list, file)
        with open(self.trade_list_path, 'wb') as file:
            pickle.dump(trade_list, file)
        with open(self.mv_trade_list_path, 'wb') as file:
            pickle.dump(mv_trade_list, file)


