# %%
import numpy as np
from env import Game
from policy import UniformPolicy, AlwaysCallPolicy

from collections import defaultdict


# %%
def collect_data(game=None, policies=[], collect_index=0, n_collect=1):

    data = defaultdict(list)
    
    for c in range(n_collect):

        game.reset()

        while not game.over:
            i = game.whose_turn

            step, obs, reward, done, valid_act_filter = game.to_player_view()

            act_probs = policies[i](obs, valid_act_filter)
            act = np.random.choice(game.acts, p=act_probs.reshape(-1))
            act_prob = act_probs[:, act]

            act = game.receive_act(act).reshape(1, -1)

            if i == collect_index:
                
                data['step'].append(step[0])
                data['obs'].append(obs[0])
                data['reward'].append(reward[0])
                data['done'].append(done[0])
                data['valid_act_filter'].append(valid_act_filter[0])

                if step[0, 0] > 0:
                    data['next_step'].append(step[0])
                    data['next_obs'].append(obs[0])
                    data['next_reward'].append(reward[0])
                    data['next_done'].append(done[0])
                    data['next_valid_act_filter'].append(valid_act_filter[0])

                data['act_probs'].append(act_probs[0])
                data['act'].append(act[0])
                data['act_prob'].append(act_prob[0])

        for i in range(game.n_player):
            if i == collect_index:
                step, obs, reward, done, valid_act_filter = game.to_player_view(i)
                data['next_step'].append(step[0])
                data['next_obs'].append(obs[0])
                data['next_reward'].append(reward[0])
                data['next_done'].append(done[0])
                data['next_valid_act_filter'].append(valid_act_filter[0])

    

    for k, v in data.items():
        data[k] = np.array(v)

    data['batch_size'] = len(data['obs'])

    return data

game = Game(max_n_player=6, n_player=6)
policies = [AlwaysCallPolicy(n_act=game.n_act, dtype=game.dtype)] * game.n_player
data = collect_data(game, policies, n_collect=10, collect_index=0)
data


class Buffer():

    def __init__(self, max_size=10000):
        self.__MAX_SIZE = max_size
        self.__size = 0
        self.__buffer = dict()

    def add(self, data):
        for k, v in data.items():
            if k == 'batch_size':
                self.__size = min(self.__size + v, self.__MAX_SIZE)
                self.__buffer[k] = self.__size
            elif k in self.__buffer:
                self.__buffer[k] = np.concatenate((self.__buffer[k], v), axis=0)[-self.__MAX_SIZE:]
            else:
                self.__buffer[k] = v[-self.__MAX_SIZE:]

    def sample(self, n=1, replace=True):
        if n > self.__size:
            replace = True
        all_indices = np.arange(self.__size, dtype=np.int_)
        sampled_indices = np.random.choice(all_indices, size=n, replace=replace)
        sampled_data = {}
        for k, v in self.__buffer.items():
            if k == 'batch_size':
                sampled_data[k] = n
            else:
                sampled_data[k] = v[sampled_indices]
        return sampled_data
    
    def clear(self):
        self.__size = 0
        self.__buffer = dict()

    @property
    def max_size(self):
        return self.__MAX_SIZE
    
    @property
    def buffer(self):
        return self.__buffer
    
    def __len__(self):
        return self.__size
    

    
buffer = Buffer(100)
buffer.add(data)
buffer.sample(5)
# %%


# %%
