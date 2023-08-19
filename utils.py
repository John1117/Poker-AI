import numpy as np
import warnings
import torch
from tensordict.tensordict import TensorDict
from torch.distributions.categorical import Categorical
from env import DEFAULT_DTYPE


def coll_data(game, player_lst, num_of_coll=1, device='cpu'):
    num_of_player = len(player_lst)
    num_of_nn_player = 0
    idx_map = [] # map the whose_turn idx of nn_player to the lst idx of dict_lst
    for player in player_lst:
        if player.use_nn:
            idx_map.append(num_of_nn_player)
            num_of_nn_player += 1
        else:
            idx_map.append(None)
    # e.g. if the sit_idx of nn_player are [0, 2, 3] in six player game
    # the idx map would be [0, None, 1, 2, None, None] which maps [0, 2, 3] to [0, 1, 2]

    dict_lst = [{'obs': [], 'val': [], 'act': [], 'act_log_prob': [], 'rwd': [], 'done': []} for _ in range(num_of_nn_player)]
    
    for _ in range(num_of_coll):
        game.reset()
        while not game.is_over:
            wt = game.whose_turn
            obs, rwd, done = game.to_player_pov(wt)

            if player_lst[wt].use_nn:
                
                if game.player_lines[wt].item() != 1:
                    #if nn_player took act before, collect the next state info
                    dict_lst[idx_map[wt]]['rwd'].append(rwd)
                    dict_lst[idx_map[wt]]['done'].append(done)
                
                val = player_lst[wt].eval(obs)
                act_probs = player_lst[wt].policy(obs)
                act_distr = Categorical(act_probs)
                act = act_distr.sample(torch.Size([1])).to(torch.int64)
                act_log_prob = act_distr.log_prob(act)
                real_act = game.next(act)
                #print(real_act)
                
                # collect the this state info
                dict_lst[idx_map[wt]]['obs'].append(obs)
                dict_lst[idx_map[wt]]['val'].append(val)
                dict_lst[idx_map[wt]]['act'].append(act)
                dict_lst[idx_map[wt]]['act_log_prob'].append(act_log_prob)

            else:
                act_probs = player_lst[wt].policy(obs)
                act_distr = Categorical(act_probs)
                act = act_distr.sample(torch.Size([1])).to(torch.int64)
                game.next(act)

        # collect the final next state info after the game is over
        for wt in range(num_of_player):
            if player_lst[wt].use_nn:
                obs, rwd, done = game.to_player_pov(wt)
                dict_lst[idx_map[wt]]['rwd'].append(rwd)
                dict_lst[idx_map[wt]]['done'].append(done)

    # concat all the data from all of the nn_player who use the same nn
    td_lst = []
    for d in dict_lst:
        for k, v in d.items():
            d[k] = torch.cat(v, dim=0)
        td_lst.append(TensorDict(d, batch_size=d['obs'].size(dim=0)))
    return torch.cat(td_lst).to(device).detach()
