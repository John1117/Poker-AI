# %%
import numpy as np
from env import Game
from policy import UniformPolicy, AlwaysCallPolicy

from collections import defaultdict


# %%
def collect_data(game=None, policies=[], collect_index=0, n_collect=1):

    data = defaultdict(list)
    
    for _ in range(n_collect):

        game.reset()

        while not game.over:
            i = game.whose_turn

            step, observation, reward, done, valid_action_filter = game.to_player_view()

            action_probs = policies[i](observation, valid_action_filter)
            action = np.random.choice(game.actions, p=action_probs.reshape(-1))
            action_prob = action_probs[:, action]

            action = game.receive_action(action).reshape(1, -1)


            if i == collect_index:
                data['step'].append(step[0])
                data['observation'].append(observation[0])
                data['reward'].append(reward[0])
                data['done'].append(done[0])
                data['valid_action_filter'].append(valid_action_filter[0])

                if step[0, 0] > 0:
                    data['next_step'].append(step[0])
                    data['next_observation'].append(observation[0])
                    data['next_reward'].append(reward[0])
                    data['next_done'].append(done[0])
                    data['next_valid_action_filter'].append(valid_action_filter[0])

                data['action_probs'].append(action_probs[0])
                data['action'].append(action[0])
                data['action_prob'].append(action_prob[0])

        for i in range(game.n_player):
            if i == collect_index:
                step, observation, reward, done, valid_action_filter = game.to_player_view(i)
                data['next_step'].append(step[0])
                data['next_observation'].append(observation[0])
                data['next_reward'].append(reward[0])
                data['next_done'].append(done[0])
                data['next_valid_action_filter'].append(valid_action_filter[0])

    
    for k, v in data.items():
        data[k] = np.array(v)

    data['batch_size'] = len(data['observation'])

    return data

game = Game(max_n_player=6, n_player=6)
policies = [AlwaysCallPolicy(n_action=game.n_action, dtype=game.dtype)] * game.n_player
data = collect_data(game, policies, n_collect=10, collect_index=0)
data


class Buffer():

    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.size = 0
        self.buffer = dict()

    def add(self, data):
        for k, v in data.items():
            if k == 'batch_size':
                self.size = min(self.size + v, self.max_size)
                self.buffer[k] = self.size
            elif k in self.buffer:
                self.buffer[k] = np.concatenate((self.buffer[k], v), axis=0)[-self.max_size:]
            else:
                self.buffer[k] = v[-self.max_size:]

    def sample(self, n=1, replace=True):
        if n > self.size:
            replace = True
        all_indices = np.arange(self.size, dtype=np.int_)
        sampled_indices = np.random.choice(all_indices, size=n, replace=replace)
        sampled_data = {}
        for k, v in self.buffer.items():
            if k == 'batch_size':
                sampled_data[k] = n
            else:
                sampled_data[k] = v[sampled_indices]
        return sampled_data
    
    def clear(self):
        self.size = 0
        self.buffer = dict()
    
buffer = Buffer(100)
buffer.add(data)
buffer.sample(5)
# %%

