import numpy as np

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


DEFAULT_DTYPE = np.float32

class FakeGameEnv():
    def __init__(self, num_of_player=6, dtype=DEFAULT_DTYPE):
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

        return obs, self.rwd_arr[whose_pov]
    
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

if __name__ == '__main__':
    game = FakeGameEnv()
    while not game.is_over:

        wt = game.whose_turn
        obs, rwd = game.to_player_pov()
        act = np.random.choice(22)
        game.nxt(act)

        print(f'S{game.street} P{wt} {ACT_DICT[act]}')

"""
class GameEnv():

    def __init__(self, nn_posi=[0, 2, 3], dtype=np.float32):
        self.dtype = dtype

        self.i_game = -1
        self.sits = np.arange(0, 6, dtype=np.int32)
        self.nn_posi = nn_posi
        self.n_player_using_nn_policy = len(nn_posi)

        self.reset()

    def reset(self):
        print('=============================================================')
        self.i_game += 1
        self.whose_turn = (self.i_game + 2) % 6  # UTG
        self.street = 0

        self.deck = np.arange(1, 53, dtype=self.dtype)
        np.random.shuffle(self.deck)

        self.hands = np.sort(np.reshape(self.deck[:12], (6, 2)), axis=-1)

        self.pot = 1.5
        self.to_call = 1.0
        self.min_raise = 2.0
        self.bets = np.zeros(6, self.dtype)
        self.bets[self.whose_turn - 2] = 0.5  # SB
        self.bets[self.whose_turn - 1] = 1.0  # BB
        self.chips = np.ones(6, self.dtype) * 100.0
        self.chips[self.whose_turn - 2] = 99.5  # SB
        self.chips[self.whose_turn - 1] = 99.0  # BB

        self.act_lines = [''] * 6
        self.lines = np.ones(6, self.dtype)  # all players inaction, act line code = 1

        self.rwds = np.zeros((6, 1), self.dtype)
        self.all_ins = np.zeros(6, bool)
        self.folds = np.zeros(6, bool)
        self.dones = np.zeros(6, bool)

    def to_player_pov(self, whose_pov=None):
        if whose_pov is None:
            whose_pov = self.whose_turn
        
        obs = np.zeros(26, dtype=self.dtype)
        obs[0:2] = self.hands[whose_pov]

        # board cards
        if self.street > 0:
            obs[2:5+self.street-1] = self.deck[12:15+self.street-1]

        obs[7:26] = np.concatenate(([self.pot], self.bets, self.chips, self.lines))

        return obs, self.rwds[whose_pov], self.dones[whose_pov]
    
    def nxt(self, act_key):

        assert act_key in ACT_DICT
        act_val = ACT_DICT[act_key]

        wt = self.whose_turn

        if act_val[0] == 'F':
            self.bets[wt] = 0.0
            self.act_lines[wt] += 'F'
            self.lines[wt] *= -1
            self.rwds[wt, 0] = self.chips[wt] - 100.0
            self.folds[wt] = True
            self.dones[wt] = True
            
        elif act_val[0] == 'C':
            make_up = min(self.to_call - self.bets[wt], self.chips[wt])
            
            self.pot += make_up
            self.bets[wt] += make_up
            self.chips[wt] -= make_up
            self.act_lines[wt] += 'C'
            self.lines[wt] = encode_act_line(self.act_lines[wt])


        elif act_val[0] == 'B':
            my_chip = self.chips[wt]
            my_pre_bet = self.bets[wt]

            if act_val[0] == 'B':
                make_up_call = min(self.to_call - my_pre_bet, my_chip)
                my_bet = self.pot * float(act_val[2:-1]) / 100.0
                make_up_bet = min(my_bet - my_pre_bet, my_chip)
            elif act_val[0] == 'A':
                make_up_call = my_chip
                make_up_bet = my_chip
                my_bet = my_pre_bet + make_up_bet
                self.all_ins[wt] = True

            if my_bet < (self.to_call + self.min_raise) / 2:
                #call
                self.pot += make_up_call
                self.bets[wt] += make_up_call
                self.chips[wt] -= make_up_call
                self.act_lines[wt] += 'C'
                self.lines[wt] = encode_act_line(self.act_lines[wt])

            elif my_bet >= (self.to_call + self.min_raise) / 2 and my_bet < self.min_raise:
                if my_chip < self.min_raise:
                    #call
                    self.as_call(my_chip)
                else:
                    self.as_bet(self.min_raise)
            else:
                self.as_bet(my_bet)

        fold_to_one = sum(self.folds) == 5
        call_to_one = all([b==self.to_call or f for b, f in zip(self.bets, self.folds)]) # may equal to fold_to_one
        if fold_to_one:
            self.choose_winner()

        elif call_to_one:
            if self.street == 3:
                self.choose_winner()
            else:
                print('-------------------------------')
                wt = self.i_game % 6 # to SB
                while self.folds[wt] or self.all_ins[wt]:
                    wt = (wt + 1) % 6
                self.whose_turn = wt
                self.street += 1
                self.bets = np.zeros(6, self.dtype)
        
        else:
            wt = (wt + 1) % 6 # to nxt player
            while self.folds[wt] or self.all_ins[wt]:
                wt = (wt + 1) % 6
            self.whose_turn = wt
    
    def as_call(self, to_make_up):
        wt = self.whose_turn
        my_chip = self.chips[wt]
        my_pre_bet = self.bets[wt]
        to_make_up = min(self.to_call - my_pre_bet, my_chip)

        self.pot += to_make_up
        self.bets[wt] += to_make_up
        self.chips[wt] -= to_make_up
        self.act_lines[wt] += 'C'
        self.lines[wt] = encode_act_line(self.act_lines[wt])

    def as_bet(self, my_bet):
        wt = self.whose_turn
        self.pot += my_bet
        self.bets[wt] = my_bet
        self.chips[wt] -= my_bet
        self.min_raise = 2 * my_bet - self.to_call
        self.to_call = my_bet
        self.act_lines[wt] += 'B'
        self.lines[wt] = encode_act_line(self.act_lines[wt])

    def choose_winner(self):
        win = np.random.choice(self.sits[~self.folds])
        for wt in self.sits[~self.folds]:
            if wt == win:
                self.chips[wt] += self.pot
                self.rwds[wt] = self.chips[wt] - 100.0
            else:
                self.rwds[wt] = self.chips[wt] - 100.0
            self.dones[wt] = True
        self.reset()

game = FakeGameEnv()
for j in range(100):
    wt = game.whose_turn
    act = np.random.choice([1])
    game.nxt(act)

    print(game.act_lines)
    print(game.bets)
"""
