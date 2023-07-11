import numpy as np
import warnings
import torch
from tensordict.tensordict import TensorDict
from torch.distributions.categorical import Categorical
from env import DEFAULT_DTYPE


def coll_data(game, player_lst, num_of_coll=1):
    num_of_player = len(player_lst)
    num_of_nn_player = 0
    idx_map = []
    for player in player_lst:
        if player.use_nn:
            idx_map.append(num_of_nn_player)
            num_of_nn_player += 1
        else:
            idx_map.append(None)
    dict_lst = [{
        'observation': torch.tensor([], dtype=DEFAULT_DTYPE), 
        'reward': torch.tensor([], dtype=DEFAULT_DTYPE), 
        'done': torch.tensor([], dtype=torch.bool),
        'step_count': torch.tensor([], dtype=torch.int), 
        'action': torch.tensor([], dtype=torch.int),
        'next': {
            'observation': torch.tensor([], dtype=DEFAULT_DTYPE),
            'reward': torch.tensor([], dtype=DEFAULT_DTYPE), 
            'done': torch.tensor([], dtype=torch.bool),
            'step_count': torch.tensor([], dtype=torch.int)
            }
        } for _ in range(num_of_nn_player)]
    
    for _ in range(num_of_coll):
        game.reset()
        while not game.is_over:
            wt = game.whose_turn
            obs, rwd, done, step = game.to_player_pov(wt)

            if player_lst[wt].use_nn:

                if game.act_str_lst[wt] != 'I':
                    #if nn_player took act before, record the next state info
                    dict_lst[idx_map[wt]]['next']['observation'] = torch.cat((dict_lst[idx_map[wt]]['next']['observation'], obs))
                    dict_lst[idx_map[wt]]['next']['reward'] = torch.cat((dict_lst[idx_map[wt]]['next']['reward'], rwd))
                    dict_lst[idx_map[wt]]['next']['done'] = torch.cat((dict_lst[idx_map[wt]]['next']['done'], done))
                    dict_lst[idx_map[wt]]['next']['step_count'] = torch.cat((dict_lst[idx_map[wt]]['next']['step_count'], step))

                act_probs = player_lst[wt].policy(obs)
                act_distr = Categorical(act_probs)
                act = act_distr.sample(torch.Size([1]))
                game.next(act)
                
                dict_lst[idx_map[wt]]['observation'] = torch.cat((dict_lst[idx_map[wt]]['observation'], obs))
                dict_lst[idx_map[wt]]['reward'] = torch.cat((dict_lst[idx_map[wt]]['reward'], rwd))
                dict_lst[idx_map[wt]]['done'] = torch.cat((dict_lst[idx_map[wt]]['done'], done))
                dict_lst[idx_map[wt]]['step_count'] = torch.cat((dict_lst[idx_map[wt]]['step_count'], step))
                dict_lst[idx_map[wt]]['action'] = torch.cat((dict_lst[idx_map[wt]]['action'], act))
            else:
                act_probs = player_lst[wt].policy(obs)
                act_distr = Categorical(act_probs)
                act = act_distr.sample(torch.Size([1]))
                game.next(act)
    
        #final next state
        for wt in range(num_of_player):
            if player_lst[wt].use_nn:
                obs, rwd, done, step = game.to_player_pov(wt)
                dict_lst[idx_map[wt]]['next']['observation'] = torch.cat((dict_lst[idx_map[wt]]['next']['observation'], obs))
                dict_lst[idx_map[wt]]['next']['reward'] = torch.cat((dict_lst[idx_map[wt]]['next']['reward'], rwd))
                dict_lst[idx_map[wt]]['next']['done'] = torch.cat((dict_lst[idx_map[wt]]['next']['done'], done))
                dict_lst[idx_map[wt]]['next']['step_count'] = torch.cat((dict_lst[idx_map[wt]]['next']['step_count'], step))

    tsd_lst = []
    for d in dict_lst:
        tsd_lst.append(TensorDict(d, batch_size=d['observation'].shape[0]))
    return torch.cat(tsd_lst)
