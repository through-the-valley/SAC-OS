import gym
from gym import spaces
from gym.spaces import Dict, Discrete
import numpy as np
import pandas as pd
import random
import datetime


def save_to_dict(dict, data, name):
    if isinstance(data, float) or isinstance(data, int):
        dict[name] = data
    else:
        dict[name] = data.cpu().item()

def dis_to_con(discrete_action, low, high, bins):  # 离散动作转回连续的函数 这个函数主要是环境得到动作要去算reward的时候使用
    action_lowbound = low  # 连续动作的最小值
    action_upbound = high  # 连续动作的最大值
    return action_lowbound + (discrete_action /
                              (bins)) * (action_upbound -
                                         action_lowbound)


# 将连续值离散化，返回离散化后所有的点，最终列表长度其实是num_bins+1
def discretize_range(start, end, num_bins):
    # 计算bin的数量和大小
    bin_size = (end - start) / num_bins
    bins = np.linspace(start, end, num_bins + 1)
    # 使用numpy的histogram函数来离散化数据
    discretized_values = np.histogram(np.random.randn(10000), bins=bins)[1]
    return discretized_values


def divide_dataset(dataset, r=None):
    if r is None:
        r = [0.7, 0.2, 0.1]
    length = len(dataset)
    train, eval, test = int(length * r[0]), int(length * r[1]), int(length * r[2])
    train_set, eval_set, test_set = [], [], []
    for i in range(train):
        sample = random.choice(dataset)
        index = dataset.index(sample)
        temp = dataset.pop(index)
        train_set.append(temp)
    for i in range(eval):
        sample = random.choice(dataset)
        index = dataset.index(sample)
        temp = dataset.pop(index)
        eval_set.append(temp)
    for i in range(test):
        sample = random.choice(dataset)
        index = dataset.index(sample)
        temp = dataset.pop(index)
        test_set.append(temp)
    return train_set, eval_set, test_set


class MicroGridForDQN(gym.Env):
    def __init__(self, discrete_bins, mode_dict, dataset_range=None, run_mode='train'):
        if dataset_range is None:
            dataset_range = [0.7, 0.2, 0.1]
        self.data_path = '../data/your_data.csv'  # 改为你的数据集路径
        self.df = pd.read_csv(self.data_path)
        self.discrete_bins = discrete_bins
        self.mode_dict = mode_dict
        self.run_mode = run_mode
        self.train_set, self.eval_set, self.test_set = divide_dataset(list(range(1440)), dataset_range)

        self.r_cost_factor = 0.7
        self.r_self_balance_factor = 0.3

        # 状态空间
        self.min_pv = 0
        self.max_pv = self.df['pv'].max()
        self.min_wind = 0
        self.max_wind = self.df['wt'].max()
        self.min_price = self.df['price'].min()
        self.max_price = self.df['price'].max()
        self.min_load = 0
        self.max_load = self.df['Global_active_power'].max()

        self.min_soc_1 = 0.2
        self.max_soc_1 = 0.9
        self.capacity_1 = 25
        self.ita_charge_1 = 0.98
        self.ita_discharge_1 = 0.95
        self.c_bes_1 = 0.8

        self.min_soc_2 = 0.3
        self.max_soc_2 = 0.95
        self.capacity_2 = 50
        self.ita_charge_2 = 0.9
        self.ita_discharge_2 = 0.85
        self.c_bes_2 = 0.6

        # 动作空间
        self.min_mt_1 = 0.
        self.max_mt_1 = 20.
        self.min_mt_2 = 0.
        self.max_mt_2 = 40.
        self.min_mt_3 = 0.
        self.max_mt_3 = 40.
        self.min_mt_4 = 0.
        self.max_mt_4 = 50.
        self.a_1, self.a_2, self.a_3, self.a_4 = 0.005, 0.006, 0.0175, 0.0625
        self.b_1, self.b_2, self.b_3, self.b_4 = 3., 2.497, 0.748, 0.5
        self.c_1, self.c_2, self.c_3, self.c_4 = 0.42234332, 1., 0., 0.

        self.observation_space = spaces.Dict({
            'load': spaces.Box(low=self.min_load, high=self.max_load, shape=(1,)),
            'pv': spaces.Box(low=self.min_pv, high=self.max_pv, shape=(1,)),
            'wind': spaces.Box(low=self.min_wind, high=self.max_wind, shape=(1,)),
            'soc_1': spaces.Box(low=self.min_soc_1, high=self.max_soc_1, shape=(1,)),
            'soc_2': spaces.Box(low=self.min_soc_2, high=self.max_soc_2, shape=(1,)),
            'price': spaces.Box(low=self.min_price, high=self.max_price, shape=(1,)),
        })

        self.max_charge_1 = 17.857142857142858
        self.max_charge_2 = 36.1111111111111
        self.max_discharge_1 = 16.624999999999996
        self.max_discharge_2 = 27.624999999999993
        self.trade_sold_discount_factor = 0.9
        self.max_trade = 25
        self.min_trade = -self.max_trade
        self.action_space = spaces.Dict({
            'mt_1': spaces.Discrete(self.discrete_bins[0]),
            'mt_2': spaces.Discrete(self.discrete_bins[1]),
            'mt_3': spaces.Discrete(self.discrete_bins[2]),
            'mt_4': spaces.Discrete(self.discrete_bins[3]),
            'ess_1': spaces.Discrete(self.discrete_bins[4]),
            'ess_2': spaces.Discrete(self.discrete_bins[5]),
        })

        if self.run_mode == 'train':
            day = random.choice(self.train_set)
        elif self.run_mode == 'eval':
            day = random.choice(self.eval_set)
        else:
            day = random.choice(self.test_set)
        self.idx = 24 * day
        load, pv, wind, price = self.df['Global_active_power'][self.idx], self.df['pv'][self.idx], self.df['wt'][
            self.idx], self.df['price'][self.idx]
        soc_1 = random.uniform(self.min_soc_1, self.max_soc_1)
        soc_2 = random.uniform(self.min_soc_2, self.max_soc_2)

        self.step_count = 0  # 默认每个episode从0点开始，每一小时进行决策观察，一个episode包含24步

        self.state = {}
        self.set_state(load, pv, wind, soc_1, soc_2, price)

    def reset(self):
        if self.run_mode == 'train':
            day = random.choice(self.train_set)
        elif self.run_mode == 'eval':
            day = random.choice(self.eval_set)
        else:
            day = random.choice(self.test_set)
        if day - 1 >= 0:
            day = day - 1
        self.idx = 24 * day
        load, pv, wind, price = self.df['Global_active_power'][self.idx], self.df['pv'][self.idx], self.df['wt'][
            self.idx], self.df['price'][self.idx]
        soc_1 = random.uniform(self.min_soc_1, self.max_soc_1)
        soc_2 = random.uniform(self.min_soc_2, self.max_soc_2)

        self.set_state(load, pv, wind, soc_1, soc_2, price)
        return list(self.state.values())

    def step(self, action):
        status_variable = {}

        dis_mt_1, dis_mt_2, dis_mt_3, dis_mt_4, ess_1, ess_2 = action[:, 0], action[:, 1], action[:, 2], action[:,
                                                                                                         3], action[:,
                                                                                                             4], action[
                                                                                                                 :, 5]

        mt_1 = dis_to_con(dis_mt_1, self.min_mt_1, self.max_mt_1, self.discrete_bins[0])
        mt_2 = dis_to_con(dis_mt_2, self.min_mt_2, self.max_mt_2, self.discrete_bins[1])
        mt_3 = dis_to_con(dis_mt_3, self.min_mt_3, self.max_mt_3, self.discrete_bins[2])
        mt_4 = dis_to_con(dis_mt_4, self.min_mt_4, self.max_mt_4, self.discrete_bins[3])
        up_1, low_1, up_2, low_2 = self.calc_ess_boundary()

        p_ess_1 = dis_to_con(ess_1, low_1, up_1, self.discrete_bins[4])
        p_ess_2 = dis_to_con(ess_2, low_2, up_2, self.discrete_bins[5])

        p_trade = self.state['load'] + p_ess_1 + p_ess_2 - self.state['pv'] - self.state['wind'] - mt_1 - mt_2 - mt_3 - mt_4

        save_to_dict(status_variable, p_trade, 'trade')

        next_soc_1 = self.calc_soc(self.state['soc_1'], self.capacity_1, self.ita_charge_1, self.ita_discharge_1,
                                   p_ess_1)
        next_soc_2 = self.calc_soc(self.state['soc_2'], self.capacity_2, self.ita_charge_2, self.ita_discharge_2,
                                   p_ess_2)

        mt_1_cost = self.a_1 * (mt_1 ** 2) + self.b_1 * mt_1 + self.c_1
        mt_2_cost = self.a_2 * (mt_2 ** 2) + self.b_2 * mt_2 + self.c_2
        mt_3_cost = self.a_3 * (mt_3 ** 2) + self.b_3 * mt_3 + self.c_3
        mt_4_cost = self.a_4 * (mt_4 ** 2) + self.b_4 * mt_4 + self.c_4
        mt_cost = mt_1_cost + mt_2_cost + mt_3_cost + mt_4_cost

        bes_1_cost = self.c_bes_1 * p_ess_1.abs()
        bes_2_cost = self.c_bes_2 * p_ess_2.abs()
        bes_cost = bes_1_cost + bes_2_cost

        trade_cost = p_trade * self.state['price']
        if p_trade <= 0:
            trade_cost = self.trade_sold_discount_factor * trade_cost


        self_balance_scale_rate = 5
        r_self_balance = self_balance_scale_rate * abs(p_trade)

        save_to_dict(status_variable, r_self_balance, 'r_self_balance')

        total_cost = trade_cost + mt_cost + bes_cost
        save_to_dict(status_variable, total_cost, 'total_cost')

        r = -(self.r_cost_factor * total_cost + self.r_self_balance_factor * r_self_balance)

        self.step_count += 1
        if self.step_count == 24:
            done = 1
            self.step_count = 0
            self.reset()
        else:
            done = 0

        temp = self.idx + self.step_count
        load, pv, wind, price = self.df['Global_active_power'][temp], self.df['pv'][temp], self.df['wt'][
            temp], self.df['price'][temp]

        next_state = [load, pv, wind, next_soc_1, next_soc_2, price]

        self.set_state(load, pv, wind, next_soc_1, next_soc_2, price)

        return next_state, r, done, status_variable

    def render(self, mode='human'):
        pass

    def calc_ess_boundary(self):
        # 充电是正值，放电是负值，对应上限和下限。
        p_charge_threshold_1 = ((self.max_soc_1 - self.state['soc_1']) * self.capacity_1) / self.ita_charge_1
        p_discharge_threshold_1 = (self.min_soc_1 - self.state['soc_1']) * self.ita_discharge_1 * self.capacity_1
        p_charge_threshold_2 = ((self.max_soc_2 - self.state['soc_2']) * self.capacity_2) / self.ita_charge_2
        p_discharge_threshold_2 = (self.min_soc_2 - self.state['soc_2']) * self.ita_discharge_2 * self.capacity_2
        return p_charge_threshold_1, p_discharge_threshold_1, p_charge_threshold_2, p_discharge_threshold_2

    def calc_soc(self, soc, capacity, ita_charge, ita_discharge, p_ess):
        if p_ess >= 0:
            next_soc = soc + p_ess * ita_charge * 1.0 / capacity
        else:
            next_soc = soc + p_ess * 1.0 / (capacity * ita_discharge)
        return next_soc

    def set_state(self, load, pv, wind, soc_1, soc_2, price):
        self.state['load'] = load
        self.state['pv'] = pv
        self.state['wind'] = wind
        self.state['soc_1'] = soc_1
        self.state['soc_2'] = soc_2
        self.state['price'] = price


