import numpy as np
from env import FakeGameEnv, ACT_DICT
from player import RandomPolicy, NNPolicy, Player

class Buffer():
    def __init__(self):
        self.bag = []
        self.num_of_episo_in_bag = 0
        self.idx_table = []

    def add(self, episo_lst):
        self.bag += episo_lst
        num_of_episo = len(episo_lst)
        for i in range(num_of_episo):
            for j in episo_lst[i]['idx']:
                self.idx_table.append([self.num_of_episo_in_bag + i, j])

        self.num_of_episo_in_bag += num_of_episo

    def sample(self):
        pass

buf = Buffer()
ep = [{'idx': [0, 1]}] * 2
buf.add(ep)
buf.add(ep)
print(buf.bag)
print(buf.idx_table)

def coll_episo(game, player_lst):

    num_of_nn_player = 0
    nn_idx_table = []
    for player in player_lst:
        if player.use_nn:
            nn_idx_table.append(num_of_nn_player)
            num_of_nn_player += 1
        else:
            nn_idx_table.append(None)

    episo_lst = [{'idx': [], 'obs': [], 'act': [], 'rwd': []}] * num_of_nn_player
    game.reset()
    while not game.is_over:
        wt = game.whose_turn
        obs, rwd = game.to_player_pov(wt)

        if player_lst[wt].use_nn:
            act_distr = player_lst[wt].policy(obs)
            act = np.random.choice(num_of_act, p=act_distr)
            game.nxt(act)
            episo_lst[nn_idx_table[wt]]

        else:
            act_distr = player_lst[wt].policy(obs)
            act = np.random.choice(num_of_act, p=act_distr)
            game.nxt(0)

        #print(f'S{game.street} P{wt} {ACT_DICT[act]}')
    




num_of_player = 6
num_of_act = 22
nn_player_sit_idx_lst = [0, 2, 3]

player_lst = []
for sit_idx in range(num_of_player):
    if sit_idx in nn_player_sit_idx_lst:
        player_lst.append(Player(NNPolicy()))
    else:
        player_lst.append(Player(RandomPolicy()))

num_of_nn_player = 0
nn_idx_table = []
for player in player_lst:
    if player.use_nn:
        nn_idx_table.append(num_of_nn_player)
        num_of_nn_player += 1
    else:
        nn_idx_table.append(None)
print(nn_idx_table)

game = FakeGameEnv(num_of_player)


    