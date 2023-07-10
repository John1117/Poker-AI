import numpy as np
import torch


# obs = [hand[0:2], flop[2:5], turn[5:6], river[6:7], pot[7:8], bets[8:14], chips[14,20], line[20:26]]
# act = [fold[0:1], call[1:2], bets[2:21], all-in[21:22]]

def make_act_code(odd_incr=0.05):
    odd = np.arange(odd_incr, 1, odd_incr)
    bet = np.int32(odd/(1-odd)*100)

    n_code = len(bet) + 3
    code = {}
    for i in range(n_code):
        if i==0:
            code[i] = 'F'
        elif i==1:
            code[i] = 'C'
        elif i==n_code-1:
            code[i] = 'A'
        else:
            code[i] = f'B_{bet[i-2]}%'
    return code
    
ACT_DICT = make_act_code(odd_incr=0.05)

def encode_act_str(act_str='I'):
    assert isinstance(act_str, str)

    act_bin = '1' # default inaction
    for a in act_str:
        if a == 'C': # check, call, limp
            act_bin += '0'
        elif a == 'B': # open, bet, raise, all-in
            act_bin += '1'
        elif a == 'N': # no player
            act_bin = '0'
    
    code = int(act_bin, 2)
    if act_str[-1] == 'F':
        code *= -1
    return code


DEFAULT_DTYPE = torch.float32
class FakeGameTorchEnv():
    def __init__(self, num_of_player=6, dtype=DEFAULT_DTYPE, device=None):
        self.dtype = dtype
        self.device = device

        self.num_of_player = num_of_player
        self.sb_idx = -1

        self.reset()

    def reset(self):
        self.is_over = False

        self.sb_idx = (self.sb_idx + 1) % 6 #sit idx of SB
        self.whose_turn = (self.sb_idx + 2) % 6 #sit idx of UTG
        self.street = 0

        self.deck = torch.randperm(52, dtype=self.dtype, device=self.device) + 1
        self.hand_arr = self.deck[:12].reshape(6, 2).sort(dim=-1).values
        self.flop = self.deck[12:15].sort().values
        self.turn = self.deck[15:16]
        self.river = self.deck[16:17]

        self.pot = torch.tensor([1.5], dtype=self.dtype, device=self.device)
        self.bet_arr = torch.zeros(self.num_of_player, dtype=self.dtype, device=self.device)
        self.bet_arr[self.whose_turn - 2] = 0.5  #SB
        self.bet_arr[self.whose_turn - 1] = 1.0  #BB
        self.chip_arr = 100.0 - self.bet_arr #all initialize to 100

        self.act_str_lst = ['I' for _ in range(self.num_of_player)]
        self.act_code_arr = torch.ones(self.num_of_player, dtype=self.dtype, device=self.device)  # all players inaction, act line code = 1

        self.num_of_call = 0
        self.fold_arr = torch.zeros(self.num_of_player, dtype=torch.bool, device=self.device)

        self.rwd_arr = torch.zeros((self.num_of_player, 1, 1), dtype=self.dtype, device=self.device)
        self.done_arr = torch.zeros((self.num_of_player, 1, 1), dtype=torch.bool, device=self.device)

    def to_player_pov(self, whose_pov=None):
        if whose_pov is None:
            whose_pov = self.whose_turn
        
        obs = torch.zeros((1, self.num_of_player * 3 + 8), dtype=self.dtype, device=self.device)
        obs[0, 0:2] = self.hand_arr[whose_pov]

        # board cards
        if self.street > 0:
            obs[0, 2:5] = self.flop
        if self.street > 1:
            obs[0, 5:6] = self.turn
        if self.street > 2:
            obs[0, 5:7] = self.river

        obs[0, 7:] = torch.cat((self.pot, self.bet_arr, self.chip_arr, self.act_code_arr))

        return obs, self.rwd_arr[whose_pov], self.done_arr[whose_pov]
    
    def next(self, act_key):
        act_key = act_key.item() #to python int
        assert act_key in ACT_DICT
        act_val = ACT_DICT[act_key] #act str

        wt = self.whose_turn

        foo = torch.rand(1, dtype=self.dtype, device=self.device)
        if act_val[0] == 'F':
            self.pot = foo
            self.bet_arr[wt] = foo
            self.chip_arr[wt] = foo

            self.act_str_lst[wt] += 'F'
            self.act_code_arr[wt] *= -1
            self.fold_arr[wt] = True
        
        elif act_val[0] == 'C' or act_val[0] == 'B':          
            self.pot = foo
            self.bet_arr[wt] = foo
            self.chip_arr[wt] = foo

            self.act_str_lst[wt] += 'C'
            self.act_code_arr[wt] = encode_act_str(self.act_str_lst[wt])
            self.num_of_call += 1

        elif act_val[0] == 'A':
            self.pot = foo
            self.bet_arr[wt] = foo
            self.chip_arr[wt] = foo

            self.act_str_lst[wt] += 'B'
            self.act_code_arr[wt] = encode_act_str(self.act_str_lst[wt])
            self.num_of_call = 0

        all_fold = self.fold_arr.sum() == self.num_of_player - 1
        all_call = self.num_of_call == self.num_of_player - self.fold_arr.sum()
        if all_fold:
            return self.distirbute_pot()
        elif all_call:
            if self.street == 3:
                return self.distirbute_pot()
            else:
                self.num_of_call = 0
                wt = self.sb_idx % 6 # to SB
                self.street += 1
                self.bet_arr = torch.zeros(self.num_of_player, dtype=self.dtype, device=self.device)
        else:
            wt = (wt + 1) % self.num_of_player # to nxt player

        while self.fold_arr[wt]:
            wt = (wt + 1) % 6
        self.whose_turn = wt

    def distirbute_pot(self):
        self.rwd_arr = torch.rand((self.num_of_player, 1, 1), dtype=self.dtype, device=self.device)
        self.done_arr = torch.ones((self.num_of_player, 1, 1), dtype=torch.bool, device=self.device)
        self.is_over = True

if __name__ == '__main__':
    game = FakeGameTorchEnv()
    game.reset()
    while not game.is_over:
        obs, rwd, done = game.to_player_pov()
        act = torch.randint(0, 22, (1,))
        game.nxt(act)

DEFAULT_NPTYPE = np.float32
class FakeGameNumpyEnv():
    def __init__(self, num_of_player=6, dtype=DEFAULT_NPTYPE):
        self.dtype = dtype

        self.num_of_player = num_of_player
        self.sb_idx = -1

        self.reset()

    def reset(self):
        self.is_over = False

        self.sb_idx = (self.sb_idx + 1) % 6 #sit idx of SB
        self.whose_turn = (self.sb_idx + 2) % 6 #sit idx of UTG
        self.street = 0

        self.deck = np.arange(1, 53, dtype=self.dtype)
        np.random.shuffle(self.deck)
        self.hand_arr = np.sort(np.reshape(self.deck[:12], (6, 2)), axis=-1)

        self.pot = 1.5
        self.bet_arr = np.zeros(self.num_of_player, self.dtype)
        self.bet_arr[self.whose_turn - 2] = 0.5  #SB
        self.bet_arr[self.whose_turn - 1] = 1.0  #BB
        self.chip_arr = 100.0 - self.bet_arr #all initialize to 100

        self.act_str_lst = ['I'] * self.num_of_player
        self.act_code_arr = np.ones(self.num_of_player, self.dtype)  # all players inaction, act line code = 1

        self.num_of_call = 0
        self.fold_arr = np.zeros(6, bool)

        self.rwd_arr = np.zeros((6, 1), self.dtype)
        self.done_arr = np.zeros((6, 1), bool)

    def to_player_pov(self, whose_pov=None):
        if self.is_over:
            self.reset()

        if whose_pov is None:
            whose_pov = self.whose_turn
        
        obs = np.zeros(self.num_of_player * 3 + 8, dtype=self.dtype)
        obs[0:2] = np.sort(self.hand_arr[whose_pov])

        # board cards
        if self.street > 0:
            obs[2:5] = np.sort(self.deck[12:15])
            if self.street == 2:
                obs[5:6] = self.deck[15:16]
            elif self.street == 3:
                obs[5:7] = self.deck[15:17]

        obs[7:] = np.concatenate(([self.pot], self.bet_arr, self.chip_arr, self.act_code_arr))

        return obs, self.rwd_arr[whose_pov], self.done_arr[whose_pov]
    
    def nxt(self, act_key):
        assert act_key in ACT_DICT
        act_val = ACT_DICT[act_key]

        wt = self.whose_turn

        foo = np.random.rand()
        if act_val[0] == 'F':
            self.pot = foo
            self.bet_arr[wt] = foo
            self.chip_arr[wt] = foo

            self.act_str_lst[wt] += 'F'
            self.act_code_arr[wt] *= -1
            self.fold_arr[wt] = True
        
        elif act_val[0] == 'C' or act_val[0] == 'B':          
            self.pot = foo
            self.bet_arr[wt] = foo
            self.chip_arr[wt] = foo

            self.act_str_lst[wt] += 'C'
            self.act_code_arr[wt] = encode_act_str(self.act_str_lst[wt])
            self.num_of_call += 1

        elif act_val[0] == 'A':
            self.pot = foo
            self.bet_arr[wt] = foo
            self.chip_arr[wt] = foo

            self.act_str_lst[wt] += 'B'
            self.act_code_arr[wt] = encode_act_str(self.act_str_lst[wt])
            self.num_of_call = 0

        all_fold = self.fold_arr.sum() == self.num_of_player - 1
        all_call = self.num_of_call == self.num_of_player - self.fold_arr.sum()
        if all_fold:
            return self.distirbute_pot()
        elif all_call:
            if self.street == 3:
                return self.distirbute_pot()
            else:
                self.num_of_call = 0
                wt = self.sb_idx % 6 # to SB
                self.street += 1
                self.bet_arr = np.zeros(self.num_of_player, self.dtype)
        else:
            wt = (wt + 1) % self.num_of_player # to nxt player

        while self.fold_arr[wt]:
            wt = (wt + 1) % 6
        self.whose_turn = wt

    def distirbute_pot(self):
        self.rwd_arr = np.random.rand(self.num_of_player, 1)
        self.is_over = True
