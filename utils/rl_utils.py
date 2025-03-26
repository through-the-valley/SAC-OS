from tqdm import tqdm
import numpy as np
import torch
import collections
import random


'''
清单： 
    dis_to_con
    replay buffer
    moving average
    sumtree
    memory(与sumtree配套使用)
    
'''


def dis_to_con(discrete_action, low, high, action_dim):  # 离散动作转回连续的函数 这个函数主要是环境得到动作要去算reward的时候使用
    """
    :param discrete_action: 应该是网络输出的动作代号（0-自由度）
    :param env:
    :param action_dim: 动作自由度
    这里是由于动作空间只有一个，如果有复数个，action_space.low/high返回的便是列表了，要用到索引
    :return: 转为真正的动作值，本质还是离散
    """
    action_lowbound = low  # 连续动作的最小值
    action_upbound = high  # 连续动作的最大值
    return action_lowbound + (discrete_action /
                              (action_dim - 1)) * (action_upbound -
                                                   action_lowbound)


class ReplayBuffer:
    # 不带优先级的经验回放 先进先出 构建时需设定缓存大小
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    # 增加时传入5元组： s，a，r，s，done
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # 均匀采集采样
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        # 在这里*作为解包运算符。zip（*xxx）其实是zip的逆操作。
        state, action, reward, next_state, done = zip(*transitions)
        # return np.array(state), action, reward, np.array(next_state), done
        return state, action, reward, next_state, done

    def size(self):
        return len(self.buffer)

    def clear_all(self):
        self.buffer.clear()


def moving_average(a, window_size):

    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size

    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]

    return np.concatenate((begin, middle, end))

    # np.cumsum(a,axis=None,dtype=None,out=None) 计算特定方向上的元素累计和，如果axis不是None或a是一维那么返回的array的size和shape都与a一致
    # 当axis=None时 会将输入a展平成一维的再计算累计和
    # 当axis=0时为按行累加。当axis=1时为按列累加。这里的按行/列累加的含义是本行/列=本行/列+上一行/列 第一行/列没有前一行/列所以等于本身。
    # a=np.insert(arr, obj, values, axis)
    # arr原始数组，可一可多，obj插入元素位置，values是插入内容，axis是按行按列插入（0：行、1：列）。

    # np.arrange与python自带函数range类似 留头去尾，区别是它返回的时ndarray 与np.arrange类似的还有np.linspace，但linspcae指定的不是间隔而是你需要多少个点。
    # np.concatenate可以在指定维度上拼接ndarray

'''
这是原作者的代码
'''
# class SumTree:
#     write = 0
#
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.tree = np.zeros( 2*capacity - 1 )
#         self.data = np.zeros( capacity, dtype=object )
#
#     def _propagate(self, idx, change):  # 前向传播
#         parent = (idx - 1) // 2
#
#         self.tree[parent] += change   # p值更新后，其上面的父节点也都要更新
#
#         if parent != 0:
#             self._propagate(parent, change)
#
#     def update(self, idx, p):
#         change = p - self.tree[idx]
#
#         self.tree[idx] = p
#         self._propagate(idx, change)
#
#     def add(self, p, data):
#         idx = self.write + self.capacity - 1
#
#         self.data[self.write] = data
#         self.update(idx, p)
#
#         self.write += 1
#         if self.write >= self.capacity:  # 叶结点存满了就从头开始覆写
#             self.write = 0
#
#     def _retrieve(self, idx, s):  # 检索s值
#         left = 2 * idx + 1
#         right = left + 1
#
#         if left >= len(self.tree):  # 说明已经到叶结点了
#             return idx
#
#         if s <= self.tree[left]:
#             return self._retrieve(left, s)  # 递归调用
#         else:
#             return self._retrieve(right, s-self.tree[left])
#
#     def get(self, s):
#         idx = self._retrieve(0, s)
#         dataIdx = idx - self.capacity + 1
#
#         return (idx, self.tree[idx], self.data[dataIdx])
#
#     def total(self):
#         return self.tree[0]
#
#
# class Memory:   # stored as ( s, a, r, s_ ) in SumTree
#     e = 0.01
#     a = 0.6
#
#     def __init__(self, capacity):
#         self.tree = SumTree(capacity)
#         self.size = 0
#
#     def _getPriority(self, error):
#         return (error + self.e) ** self.a
#     #   还有一种算p的方法是1/rank（i）ranki是error在所有error中的排序值，这种方法不会过度关注TD-error特别大的样本
#
#     def add(self, error, sample):
#         p = self._getPriority(error)
#         self.tree.add(p, sample)
#         if self.size < self.tree.capacity:
#             self.size += 1
#
#     def sample(self, n):
#         batch = []
#         segment = self.tree.total() / n
#
#         for i in range(n):
#             a = segment * i
#             b = segment * (i + 1)
#
#             s = random.uniform(a, b)
#             (idx, p, data) = self.tree.get(s)
#             batch.append( (idx, data) )
#
#         return batch
#
#     def update(self, idx, error):
#         p = self._getPriority(error)
#         self.tree.update(idx, p)


'''
这是国人修改过的版本
'''


class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        # data是那个五元组，p是优先级
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        # 这个支持批量更新吗？
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        # 根据v进行抽样。返回叶节点编号，优先级，数据（五元组）
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total_p(self):
        return self.tree[0]  # the root，根节点的值是所有优先级的累加和


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree    本项目中其实是（s，a，r，s，d）五元组
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    # p的范围应在[epsilon,abs_err_upper]之间
    epsilon = 0.01  # small amount to avoid zero priority 保证所有transition都有概率能被选到
    alpha = 0.6  # [0~1] convert the importance of TD error to priority 优先级的影响程度，0相当于没有优先级
    beta = 0.4  # importance-sampling, from initial value increasing to 1 用于修正采样偏差的超参数 慢慢增加到1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error warmup阶段给所有样本的初始优先级 结合update来看，这个值也是优先级的下界
    add_count = 0

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        # 初始必然存入1即self.abs_err_upper。之后每次存入的是优先级中的最大值，除非进行了p的更新，否则这个最大值应该一直都是self.abs_err_upper
        # 即对于第一条存储的数据，我们认为它的优先级P是最大的，同时，对于新来的数据，我们也认为它的优先级与当前树中优先级最大的经验相同。
        # 可以说存入的时候我们不用在意优先级，而是在训练完更新的时候才需要？
        # 看了下解释说是在warmup收集数据阶段无法计算TD-error也就无法计算优先值。所以可以先给所有样本一个统一的常数如1，也有人认为可以将每个transition的r作为优先级
        # warmup结束后，对于新数据论文中认为是存当前buffer数据的最大优先级
        max_p = np.max(self.tree.tree[-self.tree.capacity:]) # 目前存储的优先级中的最大值。由于初始化时设置的全0，一开始相当于必然存入的优先级是1
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max p for new p
        self.add_count += 1
        if self.add_count >= self.tree.capacity:
            self.add_count = self.tree.capacity

    # def store(self, transition, abs_error):
    #     # max_p = np.max(self.tree.tree[-self.tree.capacity:])
    #     # if max_p == 0:
    #     #     max_p = self.abs_err_upper
    #     p = (abs_error + self.epsilon) ** self.alpha
    #     self.tree.add(p, transition)   # set the max p for new p
    #     self.add_count += 1
    #     if self.add_count >= self.tree.capacity:
    #         self.add_count = self.tree.capacity

    def sample(self, n):
        # n是batch大小
        temp = self.tree.data[0]
        b_memory = []
        b_idx, ISWeights = np.empty((n,), dtype=np.int32), np.empty(
            (n, 1))
        pri_seg = self.tree.total_p() / n  # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        # 这里对计算IS WEIGHT进行了简化，只需要用到min（Pi）和Pj和超参数beta
        # 这里有个问题，当树还没填满时，这个分子必然是0，此时min_prob就会变为0，而这个值在下面是作为分母使用的，会出计算问题
        # 有说法是应该在batch内取max（原公式中的分母是max）而不是在buffer所有内容中取
        p_list = []
        prob_list = []
        # min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            # 晚点用于计算batch内的min
            p_list.append(p)
            prob = p / self.tree.total_p()
            prob_list.append(prob)
            # ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i] = idx
            b_memory.append(data)
        min_prob = min(p_list) / self.tree.total_p()
        for i in range(n):
            ISWeights[i, 0] = np.power(prob_list[i] / min_prob, -self.beta)
        return b_idx, b_memory, ISWeights

    # def sample(self, n):
    #     b_idx = np.empty((n,), dtype=np.int32)
    #     b_memory = []
    #     # b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
    #     pri_seg = self.tree.total_p() / n       # priority segment
    #     self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1
    #
    #     # 做归一化也就是除以p的最大值，这里应该在batch内取
    #     IS_pre = []
    #     # min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p()     # for later calculate ISweight
    #     for i in range(n):
    #         a, b = pri_seg * i, pri_seg * (i + 1)
    #         v = np.random.uniform(a, b)
    #         idx, p, data = self.tree.get_leaf(v)
    #         # prob = p / self.tree.total_p()
    #         # ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
    #         IS_pre.append(np.power((self.add_count * p), (-self.beta)))
    #         b_idx[i] = idx
    #         b_memory.append(data)
    #     is_np = np.array(IS_pre)
    #     IS = is_np / np.max(is_np)
    #     return b_idx, b_memory, IS

    def batch_update(self, tree_idx, abs_errors):
        # 传入的abs_errors就是td-error的abs, 看起来预期是输入numpy形式
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper) # 1相当于是下界
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    # def batch_update(self, tree_idx, abs_errors):
    #     # abs_errors += self.epsilon  # convert to abs and avoid 0
    #     # clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
    #     # ps = np.power(abs_errors, self.alpha)
    #     abs_errors = abs_errors.numpy().reshape(64)
    #     ps = np.power((abs_errors + self.epsilon), self.alpha)
    #     for ti, p in zip(tree_idx, ps):
    #         self.tree.update(ti, p)


def compute_advantage(gamma, lmbda, td_delta):
    # pytroch中detach方法用于返回一个新的tensor，这个tensor和原来的tensor共享的内存空间，但不参与反向传播，是一种有效的处理中间结果的方式
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
