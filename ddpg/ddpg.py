
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import utils.rl_utils as rl_utils
import env.microgrid_dg_considered as my_env
import pickle



class OrnsteinUhlenbeckProcess:
    def __init__(self, size, mu=0, theta=0.15, sigma=0.2, dt=1e-2):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()

    def reset(self):
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) * self.dt + \
             self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.state += dx
        return self.state


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim_1, hidden_dim_2, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim_1)
        self.fc2 = torch.nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc_mt_1 = torch.nn.Linear(hidden_dim_2, 1)
        self.fc_mt_2 = torch.nn.Linear(hidden_dim_2, 1)
        self.fc_mt_3 = torch.nn.Linear(hidden_dim_2, 1)
        self.fc_mt_4 = torch.nn.Linear(hidden_dim_2, 1)
        self.fc_ess_1 = torch.nn.Linear(hidden_dim_2, 1)
        self.fc_ess_2 = torch.nn.Linear(hidden_dim_2, 1)
        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mt_1 = (torch.tanh(self.fc_mt_1(x)) + 1) / 2 * self.action_bound['max_mt_1']
        mt_2 = (torch.tanh(self.fc_mt_2(x)) + 1) / 2 * self.action_bound['max_mt_2']
        mt_3 = (torch.tanh(self.fc_mt_3(x)) + 1) / 2 * self.action_bound['max_mt_3']
        mt_4 = (torch.tanh(self.fc_mt_4(x)) + 1) / 2 * self.action_bound['max_mt_4']
        ess_1 = torch.tanh(self.fc_ess_1(x))
        ess_2 = torch.tanh(self.fc_ess_2(x))
        action = torch.cat([mt_1, mt_2, mt_3, mt_4, ess_1, ess_2], dim=1)
        return action


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim_1, hidden_dim_2, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim_1)
        self.fc2 = torch.nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc_out = torch.nn.Linear(hidden_dim_2, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1) # 拼接状态和动作
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class DDPG:
    ''' DDPG算法 '''
    def __init__(self, actor_lr, critic_lr, num_episodes, hidden_dim, batch_size):
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.num_episodes = num_episodes
        self.hidden_dim = hidden_dim
        self.gamma = 0.98
        self.tau = 0.05  # 软更新参数
        self.buffer_size = 10000
        self.minimal_size = 1000
        self.batch_size = batch_size
        self.sigma = 0.01  # 高斯噪声标准差
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.env_name = 'MicroGrid for Continuous'
        self.mode_dict = {
            'with_datetime': False,
            'r_mode': 'self_balance_plus_emission'
        }
        self.env = my_env.MicroGridForContinuous(self.mode_dict)

        self.replay_buffer = rl_utils.ReplayBuffer(self.buffer_size)
        self.state_dim = len(self.env.observation_space.spaces)
        self.action_dim = len(self.env.action_space.spaces)
        self.action_bound = self.env.get_action_bound()  # 动作最大值

        self.actor = PolicyNet(self.state_dim, self.hidden_dim, self.hidden_dim, self.action_bound).to(self.device)
        self.critic = QValueNet(self.state_dim, self.hidden_dim, self.hidden_dim, self.action_dim).to(self.device)
        self.target_actor = PolicyNet(self.state_dim, self.hidden_dim, self.hidden_dim, self.action_bound).to(self.device)
        self.target_critic = QValueNet(self.state_dim, self.hidden_dim, self.hidden_dim, self.action_dim).to(self.device)
        # 初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网络并设置和策略相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.noise = OrnsteinUhlenbeckProcess(size=6)

        self.q_value_net_path = '../model/ddpg/critic/actor_lr_' + str(self.actor_lr) + '_critic_lr_' + \
                                str(self.critic_lr) + '.pth'
        self.policy_net_path = '../model/ddpg/actor/actor_lr_' + str(self.actor_lr) + '_critic_lr_' + \
                               str(self.critic_lr) + '.pth'

        self.return_list_path = '../list/ddpg/object/return_list.pickle'
        self.mv_return_list_path = '../list/ddpg/mv_object/mv_return_list.pickle'
        self.total_cost_list_path = '../list/ddpg/cost/total_cost_list.pickle'
        self.mv_total_cost_list_path = '../list/ddpg/mv_cost/mv_total_cost_list.pickle'
        self.self_balance_list_path = '../list/ddpg/self_balance/self_balance_list.pickle'
        self.mv_self_balance_list_path = '../list/ddpg/mv_self_balance/mv_self_balance_list.pickle'
        self.trade_list_path = '../list/ddpg/trade/trade_list.pickle'
        self.mv_trade_list_path = '../list/ddpg/mv_trade/trade_list.pickle'

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
        action = self.actor(state).detach()
        n = torch.from_numpy(self.noise.sample().astype(np.float32)).view(-1,6).to(self.device)
        # 给动作添加噪声，增加探索
        action = action + n
        action[:, 4] = torch.clamp(action[:, 5], -1.0, 1.0)
        action[:, 5] = torch.clamp(action[:, 5], -1.0, 1.0)
        return action


    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float32).to(self.device)
        actions = torch.cat(transition_dict['actions'], dim=1).view(-1, 6)
        # actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 2).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络

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
                        action = self.take_action(state)
                        next_state, reward, done, _ = self.env.step(action)
                        self.replay_buffer.add(state, action, reward.item(), next_state, done)
                        state = next_state
                        episode_return += sum(reward).item()
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

        episodes_list = list(range(len(return_list)))

        mv_return = rl_utils.moving_average(return_list, 9)

        mv_costs = rl_utils.moving_average(total_cost_list, 9)

        mv_self_balance_list = rl_utils.moving_average(self_balance_list, 9)

        mv_trade_list = rl_utils.moving_average(trade_list, 9)

        self.total_cost_list = total_cost_list
        self.mv_total_cost_list = mv_costs
        self.episode_list = episodes_list
        self.self_balance = self_balance_list
        self.mv_self_balance = mv_self_balance_list

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

        torch.save(self.actor.state_dict(), self.policy_net_path)
        torch.save(self.critic.state_dict(), self.q_value_net_path)


