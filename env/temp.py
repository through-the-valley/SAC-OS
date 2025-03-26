import gym
from gym import spaces
from gym.spaces import Dict, Discrete
import numpy as np
import pandas as pd
import random
import datetime


def process_date(time):
    time_obj = datetime.datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
    # year = time_obj.year
    month = time_obj.month
    day = time_obj.day
    hour = time_obj.hour
    return month, day, hour

def save_to_dict(dict, data, name):
    if isinstance(data, float) or isinstance(data, int):
        dict[name] = data
    else:
        dict[name] = data.cpu().item()


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


class MicroGridForContinuous(gym.Env):
    def __init__(self, mode_dict, dataset_range=None, run_mode='train'):
        if dataset_range is None:
            dataset_range = [0.7, 0.2, 0.1]
        self.data_path = '../data/your_data.csv'  # 改为你的数据集路径
        self.df = pd.read_csv(self.data_path)
        self.mean_pv = self.df['pv'].mean()
        self.mean_wt = self.df['wt'].mean()
        self.mean_price = self.df['price'].mean()
        self.mean_load = self.df['Global_active_power'].mean()
        self.var_pv = max(1e-6,self.df['pv'].var())
        self.var_wt = max(1e-6,self.df['wt'].var())
        self.var_price = max(1e-6,self.df['price'].var())
        self.var_load = max(1e-6,self.df['Global_active_power'].var())
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

        # if self.mode_dict['with_datetime']:
        #     self.observation_space = spaces.Dict({
        #         'load': spaces.Box(low=self.min_load, high=self.max_load, shape=(1,)),
        #         'pv': spaces.Box(low=self.min_pv, high=self.max_pv, shape=(1,)),
        #         'wind': spaces.Box(low=self.min_wind, high=self.max_wind, shape=(1,)),
        #         'soc_1': spaces.Box(low=self.min_soc_1, high=self.max_soc_1, shape=(1,)),
        #         'soc_2': spaces.Box(low=self.min_soc_2, high=self.max_soc_2, shape=(1,)),
        #         'price': spaces.Box(low=self.min_price, high=self.max_price, shape=(1,)),
        #         'month': spaces.Discrete(12),
        #         'day': spaces.Discrete(31),
        #         'hour': spaces.Discrete(24)
        #     })
        # else:
        self.observation_space = spaces.Dict({
            'load': spaces.Box(low=self.min_load, high=self.max_load, shape=(1,)),
            'pv': spaces.Box(low=self.min_pv, high=self.max_pv, shape=(1,)),
            'wind': spaces.Box(low=self.min_wind, high=self.max_wind, shape=(1,)),
            'soc_1': spaces.Box(low=self.min_soc_1, high=self.max_soc_1, shape=(1,)),
            'soc_2': spaces.Box(low=self.min_soc_2, high=self.max_soc_2, shape=(1,)),
            'price': spaces.Box(low=self.min_price, high=self.max_price, shape=(1,)),
            # 'mt_1_pre': spaces.Box(low=self.min_mt_1, high=self.max_mt_1, shape=(1,)),
            # 'mt_2_pre': spaces.Box(low=self.min_mt_2, high=self.max_mt_2, shape=(1,)),
            # 'mt_3_pre': spaces.Box(low=self.min_mt_3, high=self.max_mt_3, shape=(1,)),
            # 'mt_4_pre': spaces.Box(low=self.min_mt_4, high=self.max_mt_4, shape=(1,)),
            # 'trade': spaces.Box(low=-50, high=50, shape=(1,))
        })



        self.max_charge_1 = 17.857142857142858
        self.max_charge_2 = 36.1111111111111
        self.max_discharge_1 = 16.624999999999996
        self.max_discharge_2 = 27.624999999999993
        self.trade_sold_discount_factor = 0.9
        self.action_space = spaces.Dict({
            'mt_1': spaces.Box(low=self.min_mt_1, high=self.max_mt_1, shape=(1,)),
            'mt_2': spaces.Box(low=self.min_mt_2, high=self.max_mt_2, shape=(1,)),
            'mt_3': spaces.Box(low=self.min_mt_3, high=self.max_mt_3, shape=(1,)),
            'mt_4': spaces.Box(low=self.min_mt_4, high=self.max_mt_4, shape=(1,)),
            'ess_1': spaces.Box(low=self.max_discharge_1, high=self.max_charge_1, shape=(1,)),
            'ess_2': spaces.Box(low=self.max_discharge_2, high=self.max_charge_2, shape=(1,))
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
        month, day, hour = process_date(self.df['datetime'][self.idx])
        soc_1 = random.uniform(self.min_soc_1, self.max_soc_1)
        soc_2 = random.uniform(self.min_soc_2, self.max_soc_2)

        self.step_count = 0  # 默认每个episode从0点开始，每一小时进行决策观察，一个episode包含24步

        self.state = {}
        # if self.mode_dict['with_datetime']:
        #     self.state = [load, pv, wind, soc_1, soc_2, price, month, day, hour]
        # else:
        #     self.state = [load, pv, wind, soc_1, soc_2, price]
        # 0：load, 1:pv, 2:wind, 3:soc1, 4:soc2, 5: price, 6: month, 7:day, 8: hour
        # self.state['mt_1_pre'] = 0
        # self.state['mt_2_pre'] = 0
        # self.state['mt_3_pre'] = 0
        # self.state['mt_4_pre'] = 0
        # self.state['trade'] = 0
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
        month, day, hour = process_date(self.df['datetime'][self.idx])
        soc_1 = random.uniform(self.min_soc_1, self.max_soc_1)
        soc_2 = random.uniform(self.min_soc_2, self.max_soc_2)
        # load = (load-self.mean_load)/self.var_load
        # pv = (pv - self.mean_pv) / self.var_pv
        # wind = (wind - self.mean_wt) / self.var_wt
        # price = (price - self.mean_price) / self.var_price

        # if self.mode_dict['with_datetime']:
        #     self.state = [load, pv, wind, soc_1, soc_2, price, month, day, hour]
        # else:
        # #     self.state = [load, pv, wind, soc_1, soc_2, price]
        # self.state['mt_1_pre'] = 0
        # self.state['mt_2_pre'] = 0
        # self.state['mt_3_pre'] = 0
        # self.state['mt_4_pre'] = 0
        # self.state['trade'] = 0
        self.set_state(load, pv, wind, soc_1, soc_2, price)
        state = list(self.state.values())
        # if not self.mode_dict['with_datetime']:
        #     state.pop()
        #     state.pop()
        #     state.pop()
        return state

    def step(self, action):
        bes_confine_factor = 100
        bes_confine = 0
        status_variable = {}
        # action = action.detach()
        # 充电为正值，放电为负值，买电为正值，卖电为负值。
        mt_1, mt_2, mt_3, mt_4, p_ess_1, p_ess_2 = action[:, 0], action[:, 1], action[:, 2], action[:,3], action[:,
                                                                                                             4], action[
                                                                                                                 :, 5]
        out_range = 0
        up_1, low_1, up_2, low_2 = self.calc_ess_boundary()
        # # 这样应该是可以确保ess的范围不会越界。
        # p_ess_1 = (up_1 - low_1)/(1 - (-1)) * (ess_1 - (-1)) + low_1
        # p_ess_2 = (up_2 - low_2)/(1 - (-1)) * (ess_2 - (-1)) + low_2

        trade = self.state['load'] + p_ess_1 + p_ess_2 - self.state['pv'] - self.state['wind'] - mt_1 - mt_2 - mt_3 - mt_4

        next_soc_1 = self.calc_soc(self.state['soc_1'], self.capacity_1, self.ita_charge_1, self.ita_discharge_1,
                                   p_ess_1)
        next_soc_2 = self.calc_soc(self.state['soc_2'], self.capacity_2, self.ita_charge_2, self.ita_discharge_2,
                                   p_ess_2)
        if next_soc_1 > self.max_soc_1:
            bes_confine += bes_confine_factor * abs(next_soc_1 - self.max_soc_1)
            next_soc_1 = self.max_soc_1
            p_ess_1_exceed = p_ess_1 - up_1
            p_ess_1 = up_1
            out_range += 1
        elif next_soc_1 < self.min_soc_1:
            bes_confine += bes_confine_factor * abs(self.min_soc_1 - next_soc_1)
            next_soc_1 = self.min_soc_1
            p_ess_1_exceed = p_ess_1 - low_1
            p_ess_1 = low_1
            out_range += 1
        else:
            p_ess_1_exceed = 0
            bes_confine += 0

        if next_soc_2 > self.max_soc_2:
            bes_confine += bes_confine_factor * abs(next_soc_2 - self.max_soc_2)
            next_soc_2 = self.max_soc_2
            p_ess_2_exceed = p_ess_2 - up_2
            p_ess_2 = up_2
            out_range += 1
        elif next_soc_2 < self.min_soc_2:
            bes_confine += bes_confine_factor * abs(self.min_soc_2 - next_soc_2)
            next_soc_2 = self.min_soc_2
            p_ess_2_exceed = p_ess_2 - low_2
            p_ess_2 = low_2
            out_range += 1
        else:
            p_ess_2_exceed = 0
            bes_confine += 0

        p_out_rage = abs(p_ess_2_exceed) + abs(p_ess_1_exceed)
        save_to_dict(status_variable, out_range, 'out_range')
        save_to_dict(status_variable, p_out_rage, 'p_out_range')

        mt_1_cost = self.a_1 * (mt_1 ** 2) + self.b_1 * mt_1 + self.c_1
        mt_2_cost = self.a_2 * (mt_2 ** 2) + self.b_2 * mt_2 + self.c_2
        mt_3_cost = self.a_3 * (mt_3 ** 2) + self.b_3 * mt_3 + self.c_3
        mt_4_cost = self.a_4 * (mt_4 ** 2) + self.b_4 * mt_4 + self.c_4
        mt_cost = mt_1_cost + mt_2_cost + mt_3_cost + mt_4_cost

        bes_1_cost = self.c_bes_1 * abs(p_ess_1)
        bes_2_cost = self.c_bes_2 * abs(p_ess_2)
        bes_cost = bes_1_cost + bes_2_cost

        trade = trade + p_ess_1_exceed + p_ess_2_exceed
        save_to_dict(status_variable, trade, 'trade')

        emission_cost = (mt_1 + mt_2 + mt_3 + mt_4) * 0.7
        save_to_dict(status_variable, emission_cost, 'emission_cost')

        trade_cost = trade * self.state['price']
        if trade <= 0:
            trade_cost = self.trade_sold_discount_factor * trade_cost

        # if trade >= 0 : # 说明需求需要依靠电网来补足
        #     r_self_balance = trade
        # else:
        #     r_self_balance = 0
        self_balance_scale_rate = 5
        r_self_balance = self_balance_scale_rate * abs(trade)

        total_cost = trade_cost + mt_cost + bes_cost

        save_to_dict(status_variable, total_cost, 'total_cost')

        emission_cost = (mt_1 + mt_2 + mt_3 + mt_4) * 0.7

        r = -(self.r_cost_factor * total_cost + self.r_self_balance_factor * r_self_balance + bes_confine)

        self.step_count += 1
        if self.step_count == 24:
            # if self.step_count % 24 == 0:
            done = 1
            self.step_count = 0
            self.reset()
        else:
            done = 0

        temp = self.idx + self.step_count
        load, pv, wind, price = self.df['Global_active_power'][temp], self.df['pv'][temp], self.df['wt'][
            temp], self.df['price'][temp]
        month, day, hour = process_date(self.df['datetime'][temp])
        # next_state = [load, pv, wind, next_soc_1, next_soc_2, price, mt_1, mt_2, mt_3, mt_4, trade]
        next_state = [load, pv, wind, next_soc_1, next_soc_2, price]
        # if not self.mode_dict['with_datetime']:
        #     next_state.pop()
        #     next_state.pop()
        #     next_state.pop()

        # self.state['mt_1_pre'] = mt_1
        # self.state['mt_2_pre'] = mt_2
        # self.state['mt_3_pre'] = mt_3
        # self.state['mt_4_pre'] = mt_4
        # self.state['trade'] = trade
        # load = (load-self.mean_load)/self.var_load
        # pv = (pv - self.mean_pv) / self.var_pv
        # wind = (wind - self.mean_wt) / self.var_wt
        # price = (price - self.mean_price) / self.var_price
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
        # self.state['month'] = month
        # self.state['day'] = day
        # self.state['hour'] = hour

    def get_action_bound(self):
        action_bound = {
            'max_mt_1': self.max_mt_1,
            'max_mt_2': self.max_mt_2,
            'max_mt_3': self.max_mt_3,
            'max_mt_4': self.max_mt_4,
            # 'bes_1': max(self.max_charge_1, self.max_discharge_1),
            # 'bes_2': max(self.max_charge_2, self.max_discharge_2)
            'bes_1': 40,
            'bes_2': 50
        }
        return action_bound