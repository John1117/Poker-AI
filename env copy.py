# %%
import numpy as np
import pandas as pd
from itertools import combinations
from container import ArrayDict



# %%
SIMPLE_CARDS = np.array([
    '', 
    '2s', '2h', '2d', '2c',
    '3s', '3h', '3d', '3c',
    '4s', '4h', '4d', '4c',
    '5s', '5h', '5d', '5c',
    '6s', '6h', '6d', '6c',
    '7s', '7h', '7d', '7c',
    '8s', '8h', '8d', '8c',
    '9s', '9h', '9d', '9c',
    'Ts', 'Th', 'Td', 'Tc',
    'Js', 'Jh', 'Jd', 'Jc',
    'Qs', 'Qh', 'Qd', 'Qc',
    'Ks', 'Kh', 'Kd', 'Kc',
    'As', 'Ah', 'Ad', 'Ac',
])
SIMPLE_CARDS.setflags(write=False)

PRETTY_CARDS = np.array([
    '', 
    '2♠', '2♥', '2♦', '2♣',
    '3♠', '3♥', '3♦', '3♣',
    '4♠', '4♥', '4♦', '4♣',
    '5♠', '5♥', '5♦', '5♣',
    '6♠', '6♥', '6♦', '6♣',
    '7♠', '7♥', '7♦', '7♣',
    '8♠', '8♥', '8♦', '8♣',
    '9♠', '9♥', '9♦', '9♣',
    'T♠', 'T♥', 'T♦', 'T♣',
    'J♠', 'J♥', 'J♦', 'J♣',
    'Q♠', 'Q♥', 'Q♦', 'Q♣',
    'K♠', 'K♥', 'K♦', 'K♣',
    'A♠', 'A♥', 'A♦', 'A♣',
])
PRETTY_CARDS.setflags(write=False)

POSITIONS = np.array(
[   
    ['', '', '', '', '', '', '', '', '', ''],
    ['', '', '', '', '', '', '', '', '', ''],
    ['', 'D', 'BB', '', '', '', '', '', '', ''],
    ['', 'SB', 'BB', 'D', '', '', '', '', '', ''],
    ['', 'SB', 'BB', 'UTG', 'D', '', '', '', '', ''],
    ['', 'SB', 'BB', 'UTG', 'CO', 'D', '', '', '', ''],
    ['', 'SB', 'BB', 'UTG', 'MP', 'CO', 'D', '', '', ''],
    ['', 'SB', 'BB', 'UTG', 'UTG+1', 'MP', 'CO', 'D', '', ''],
    ['', 'SB', 'BB', 'UTG', 'UTG+1', 'MP', 'MP+1', 'CO', 'D', ''],
    ['', 'SB', 'BB', 'UTG', 'UTG+1', 'MP', 'LJ', 'HJ', 'CO', 'D'],
])
POSITIONS.setflags(write=False)


STREETS = np.array(['pre-flop', 'flop', 'turn', 'river'])
STREETS.setflags(write=False)


def get_rank(card_arr, return_highest=True):

    # Expect dim of cards to be (N, 5)
    if card_arr.ndim == 1:
        card_arr = card_arr.reshape(1, -1)
    
    # Tranform indexes of cards into their values and suits
    # The value of A is set to 14 instead of 1
    value_arr = (card_arr - 1) // 4 + 2
    suit_arr = (card_arr - 1) % 4 + 1
    value_arr[card_arr == 0] = 0
    suit_arr[card_arr == 0] = 0

    POWERS_OF_15 = np.power(15, range(6))
    SMALLEST_STRAIGHT_VALUES = np.array([2, 3, 4, 5, 14])


    rank_arr = np.zeros((len(card_arr), 6), dtype=card_arr.dtype)

    for i, (values, suits) in enumerate(zip(value_arr, suit_arr)):

        uniq_values, counts = np.unique(values, return_counts=True)

        if len(counts) == 5:
            
            is_normal_straight = (np.diff(uniq_values) == 1).all()
            is_smallest_straight = (uniq_values == SMALLEST_STRAIGHT_VALUES).all()
            is_straight = is_normal_straight or is_smallest_straight

            is_flush = (suits == suits[0]).all()

            if is_straight:

                if is_flush:  # straight flush
                    rank_arr[i, 5] = 8
                else:  # straight without flush
                    rank_arr[i, 5] = 4
                
                # determine the rank of straight
                if is_normal_straight:
                    rank_arr[i, 4] = uniq_values[4]
                else:  # the smallest straight
                    rank_arr[i, 4] = 5

            elif is_flush:
                rank_arr[i, 5] = 5
                rank_arr[i, 0:5] = uniq_values
            
            else:  # only high card
                rank_arr[i, 0:5] = uniq_values

        elif len(counts) == 4:  # one pair
            rank_arr[i, 5] = 1
            rank_arr[i, 4] = uniq_values[counts == 2]
            rank_arr[i, 1:4] = uniq_values[counts == 1]  # kicker

        elif len(counts) == 3:  # two pairs or set

            if 2 in counts:  # two pairs
                rank_arr[i, 5] = 2
                rank_arr[i, 3:5] = uniq_values[counts == 2]
                rank_arr[i, 2] = uniq_values[counts == 1]  # kicker

            else:  # set
                rank_arr[i, 5] = 3
                rank_arr[i, 4] = uniq_values[counts == 3]
                rank_arr[i, 2:4] = uniq_values[counts == 1]  # kicker

        else:  # full house or four of kind

            if 3 in counts:  # full house
                rank_arr[i, 5] = 6
                rank_arr[i, 4] = uniq_values[counts == 3]
                rank_arr[i, 3] = uniq_values[counts == 2]  # kicker pair

            else:  # four of kind
                rank_arr[i, 5] = 7
                rank_arr[i, 4] = uniq_values[counts == 4]
                rank_arr[i, 3] = uniq_values[counts == 1]  # kicker

    # Transform the each 1d rank arr to a single value
    ranks = (rank_arr * POWERS_OF_15).sum(axis=1)

    if return_highest:
        return ranks.max()
    else:
        return ranks


def get_bet_sizes(odd_incr=0.05, dtype=np.float_):
    odds = np.arange(start=odd_incr, step=odd_incr, stop=1, dtype=dtype)
    bet_sizes = odds / (1 - odds)
    return bet_sizes


# %%
def fn(d: dict[int, str, float]):
    print(d)

d = {1: 'a'}
l = [1, 2]
isinstance(l, list[int])







# %%
class Game():

    def __init__(self, max_n_player=6, bet_sizes=get_bet_sizes(0.05), dtype=np.float_):

        self.__DTYPE = dtype
        
        self.__MAX_N_PLAYER = max_n_player
        
        self.__observed_board_keys = [
            'value',
            'is_spade',
            'is_heart',
            'is_diamond', 
            'is_club',
        ]
        self.__observed_player_i_keys = [
            'card_0_value',
            'card_1_value',
            'card_0_is_spade',
            'card_1_is_spade',
            'card_0_is_heart',
            'card_1_is_heart',
            'card_0_is_diamond',
            'card_1_is_diamond',
            'card_0_is_club',
            'card_1_is_club',
            'position_index',
        ]
        self.__observed_player_j_keys = [
            'eff_pot',
            'current_bet',
            'total_bet',
            'chip',
            'line',
            'folded',
        ]
        self.__N_OBS = len(self.__observed_board_keys) * 5 + len(self.__observed_player_i_keys) + len(self.__observed_player_j_keys) * self.__MAX_N_PLAYER

        self.__N_DEFAULT_ACT = 3
        self.__N_ACT = self.__N_DEFAULT_ACT + len(bet_sizes)
        self.__ACTS = np.arange(self.__N_ACT, dtype=np.int_)

        self.__BET_SIZES = bet_sizes.astype(self.__DTYPE)
        self.__CHIP_UNIT = np.array(0.1, self.__DTYPE)  # 0.1BB
        
        self.__n_game = 0



    def reset(
            self, 
            n_player: int = 2, 
            street_index: int = 0,
            SB_index: int = None,
            hole_cards: tuple | list | np.ndarray | dict | ArrayDict = None,
            pre_dealt_board: tuple | list | np.ndarray | dict | ArrayDict = None,
        ):

        self.__n_game += 1

        self.__n_player = n_player if n_player else 2

        self.__players = ArrayDict({
            'index': np.arange(self.__MAX_N_PLAYER),
            'playing': np.arange(self.__MAX_N_PLAYER) < self.__n_player
        })

        self.__street_index = street_index if street_index else 0
        self.__street_name = STREETS[self.__street_index]
        self.__showdown = False
        self.__over = False

        self.__SB_index = SB_index if SB_index else (self.__n_game - 1) % self.__n_player
        self.__BB_index = (self.__SB_index + 1) % self.__n_player
        self.__whose_turn = (self.__SB_index + 2) % self.__n_player

        if hole_cards:

            if isinstance(hole_cards, (tuple, list)):
                hole_cards = np.array(hole_cards)
                if hole_cards.shape != (self.__n_player, 2):
                    raise ValueError(f'Expected shape of "player_hole_cards" {(self.__n_player, 2)}, got {hole_cards.shape}')
                hc = np.zeros((self.__MAX_N_PLAYER, 2), dtype=np.int_)
                hc[0:self.__n_player] = hole_cards
                hole_cards = hc
                
            elif isinstance(hole_cards, (dict, ArrayDict)):
                hc = np.zeros((self.__MAX_N_PLAYER, 2), dtype=np.int_)
                for k, v in hole_cards.items():
                    k = int(k)
                    v = np.array(v)
                    if k in range(self.__n_player) and v.shape == (2,):
                        hc[k] = v
                    else:
                        raise ValueError('Invalid "player_hole_cards"')
                hole_cards = hc

        if pre_dealt_board:

            if isinstance(pre_dealt_board, (tuple, list)):
                pre_dealt_board = np.array(pre_dealt_board)
                if pre_dealt_board.shape != (5,):
                    raise ValueError(f'Expected shape of "pre_dealt_board" {(5,)}, got {pre_dealt_board.shape}')
                
            elif isinstance(pre_dealt_board, (dict, ArrayDict)):
                pdb = np.zeros(5, dtype=np.int_)
                if all([type(k) == int or (type(k) == str and k.isdigit()) for k in pre_dealt_board.keys()]):
                    for k, v in pre_dealt_board.items():
                        k = int(k)
                        if k in range(5):
                            pdb[k] = v
                        else:
                            raise ValueError('Invalid "player_hole_cards"')
                elif all([k in STREETS for k in pre_dealt_board.keys()]):
                    for k, v in pre_dealt_board.items():
                        if k == 'flop':
                            pdb[0:3] = v
                        elif k == 'turn':
                            pdb[3] = v
                        elif k == 'river':
                            pdb[4] = v
                pre_dealt_board = pdb

            
            
        else:
            deck = np.random.permutation(52) + 1  # reserve 0 for no card
            hole_cards = np.zeros((self.__MAX_N_PLAYER, 2), dtype=np.int_)
            hole_cards[0:self.__n_player] = np.sort(deck[0 : self.__n_player * 2].reshape(self.__n_player, 2), axis=1)
            hole_card_values = ((hole_cards - 1) // 4 + 2) * (hole_cards > 0)
            hole_card_suits = ((hole_cards - 1) % 4 + 1) * (hole_cards > 0)

            j = self.__n_player * 2
            pre_dealt_board = np.concatenate([np.sort(deck[j:j+3]), deck[j+3:j+5]])
            pre_dealt_board_values = (pre_dealt_board - 1) // 4 + 2
            pre_dealt_board_suits = (pre_dealt_board - 1) % 4 + 1

        print(hole_cards, pre_dealt_board)

        self.__board = pd.DataFrame(
            {
                'index_': np.zeros(5, dtype=np.int_),
                'value': np.zeros(5, dtype=np.int_),
                'is_spade': np.zeros(5, dtype=np.bool_),
                'is_heart': np.zeros(5, dtype=np.bool_),
                'is_diamond': np.zeros(5, dtype=np.bool_),
                'is_club': np.zeros(5, dtype=np.bool_),
                'str': np.zeros(5, dtype=np.str_),
                'pretty_str': np.zeros(5, dtype=np.str_),
                'pre_dealt_index': pre_dealt_board,
                'pre_dealt_value': pre_dealt_board_values,
                'pre_dealt_is_spade': pre_dealt_board_suits == 1,
                'pre_dealt_is_heart': pre_dealt_board_suits == 2,
                'pre_dealt_is_diamond': pre_dealt_board_suits == 3,
                'pre_dealt_is_club': pre_dealt_board_suits == 4,
                'pre_dealt_str': SIMPLE_CARDS[pre_dealt_board],
                'pre_dealt_pretty_str': PRETTY_CARDS[pre_dealt_board],
            },
            index=pd.RangeIndex(5, name='board_index')
        )

        initial_chips = 100 * self.__players.loc[:, 'playing'].astype(self.__DTYPE)
        initial_chips_CU = np.round(initial_chips / self.__CHIP_UNIT).astype(np.int_)

        blind_bets = np.zeros(self.__MAX_N_PLAYER, dtype=self.__DTYPE)
        blind_bets[[self.__SB_index, self.__BB_index]] = 0.5, 1
        blind_bets_CU = np.round(blind_bets / self.__CHIP_UNIT).astype(np.int_)

        self.__last_bet_CU = np.round(1 / self.__CHIP_UNIT).astype(np.int_)
        self.__min_raise_CU = np.round(2 / self.__CHIP_UNIT).astype(np.int_)

        eff_pots = np.repeat(blind_bets.sum(), self.__MAX_N_PLAYER) * self.__players.loc[:, 'playing']
        eff_pots_CU = np.round(eff_pots / self.__CHIP_UNIT).astype(np.int_)

        chips = initial_chips - blind_bets
        chips_CU = initial_chips_CU - blind_bets_CU

        position_indices = self.__assign_position()

        self.__players = self.__players.assign(
            **{ 
                # obs
                'card_0_value': hole_card_values[:, 0],
                'card_1_value': hole_card_values[:, 1],
                'card_0_is_spade': hole_card_suits[:, 0] == 1,
                'card_1_is_spade': hole_card_suits[:, 1] == 1,
                'card_0_is_heart': hole_card_suits[:, 0] == 2,
                'card_1_is_heart': hole_card_suits[:, 1] == 2,
                'card_1_is_diamond': hole_card_suits[:, 1] == 3,
                'card_0_is_diamond': hole_card_suits[:, 0] == 3,
                'card_0_is_club': hole_card_suits[:, 0] == 4,
                'card_1_is_club': hole_card_suits[:, 1] == 4,

                'position_index': position_indices,
                'eff_pot': eff_pots,
                'current_bet': blind_bets,
                'total_bet': blind_bets,
                'chip': chips,

                'line': self.__players.loc[:, 'playing'].astype(np.int_),
                'folded': np.zeros(self.__MAX_N_PLAYER, dtype=np.bool_),

                # view
                'card_0_index': hole_cards[:, 0],
                'card_1_index': hole_cards[:, 1],
                'card_0_str': SIMPLE_CARDS[hole_cards[:, 0]],
                'card_1_str': SIMPLE_CARDS[hole_cards[:, 1]],
                'card_0_pretty_str': PRETTY_CARDS[hole_cards[:, 0]],
                'card_1_pretty_str': PRETTY_CARDS[hole_cards[:, 1]],

                'position': POSITIONS[self.__n_player, position_indices],
                
                'initial_chip': initial_chips,

                'eff_pot_CU': eff_pots_CU,
                'current_bet_CU': blind_bets_CU,
                'total_bet_CU': blind_bets_CU,
                'chip_CU': chips_CU,
                'initial_chip_CU': initial_chips_CU,

                'all_ined': np.zeros(self.__MAX_N_PLAYER, dtype=np.bool_),
                'called': np.zeros(self.__MAX_N_PLAYER, dtype=np.bool_),
                'raised': np.zeros(self.__MAX_N_PLAYER, dtype=np.bool_),
                'can_raise': self.__players.loc[:, 'playing'],
                
                'rank': np.zeros(self.__MAX_N_PLAYER, dtype=np.int_),

                'reward': np.zeros((self.__MAX_N_PLAYER, 1), dtype=self.__DTYPE),
                'done': np.zeros((self.__MAX_N_PLAYER, 1), dtype=np.bool_),
                'step': np.zeros((self.__MAX_N_PLAYER, 1), dtype=np.int_),
            }
        )

        self.__record = pd.DataFrame(
            {
                'street': np.array(['pre-flop', 'pre-flop']),
                'player': self.__players.index.to_numpy()[[self.__SB_index, self.__BB_index]],
                'position': POSITIONS[self.__n_player, position_indices[[self.__SB_index, self.__BB_index]]],
                'eff_pot': np.array([0, 0.5], dtype=self.__DTYPE),
                'act': np.zeros(2, dtype=np.str_),
                'add': np.array([0.5, 1], dtype=self.__DTYPE),
                'bet': np.array([0.5, 1], dtype=self.__DTYPE),
            },
            index=pd.RangeIndex(2, name='state')
        )

    
    def receive_act(self, act):

        if self.__over:
            self.reset()

        valid_act_filter = self.check_valid_act(self.__whose_turn)
        act = 1 - valid_act_filter[0] if valid_act_filter[act] == False else act
                
        if act == 0:
            self.__fold()
        elif act == 1:
            self.__call()
        elif act == 2:
            self.__all_in()
        else:
            self.__raise(self.__BET_SIZES[act - self.__N_DEFAULT_ACT])

        self.__check_game_state()

        return act


    def to_player_view(self, i=None):
        i = self.__whose_turn if i is None else i

        observed_board_data = self.__board.loc[:, self.__observed_board_keys].to_numpy().T.reshape(-1).astype(self.__DTYPE)
        observed_player_i_data = self.__players.loc[i, self.__observed_player_i_keys].to_numpy().reshape(-1).astype(self.__DTYPE)
        observed_player_j_data = self.__players.loc[:, self.__observed_player_j_keys].to_numpy().T.reshape(-1).astype(self.__DTYPE)
        obs = np.concatenate((observed_board_data, observed_player_i_data, observed_player_j_data)).reshape(1, -1)
        reward = self.__players.loc[i, 'reward'].reshape(1, 1).copy()
        done = self.__players.loc[i, 'done'].reshape(1, 1).copy()
        step = self.__players.loc[i, 'step'].reshape(1, 1).copy()

        valid_act_filter = self.check_valid_act(i).reshape(1, -1)

        return step, obs, reward, done, valid_act_filter


    def check_valid_act(self, i):

        if i != self.__whose_turn or self.__over:
            return np.zeros(self.__N_ACT, np.bool_)
        
        my_eff_pot_CU = self.__players.loc[i, 'eff_pot_CU']
        my_last_bet_CU = self.__players.loc[i, 'current_bet_CU']
        my_chip_CU = self.__players.loc[i, 'chip_CU']
        max_opp_chip_CU = self.__players.loc[self.__players.loc[:, 'chip_CU'] != my_chip_CU, 'chip_CU'].max()

        my_add_to_call_CU = self.__last_bet_CU - my_last_bet_CU
        my_add_to_min_raise_CU = self.__min_raise_CU - my_last_bet_CU
        my_possible_bets_CU = np.round(my_eff_pot_CU * self.__BET_SIZES).astype(np.int_)

        can_fold = self.__last_bet_CU > 0
        can_call = my_chip_CU > my_add_to_call_CU
        can_all_in = True

        if self.__players.loc[i, 'can_raise']:
            valid_bet_filter = (my_add_to_min_raise_CU <= my_possible_bets_CU) & (my_possible_bets_CU < min(my_chip_CU, max_opp_chip_CU))
        else:
            valid_bet_filter = np.zeros_like(my_possible_bets_CU, dtype=np.bool_)
            can_all_in = False

        valid_act_filter = np.concatenate(((can_fold, can_call, can_all_in), valid_bet_filter))
        return valid_act_filter
    
    def __assign_position(self):
        positions = np.zeros(self.__MAX_N_PLAYER, np.int_)  # reserve 0 for absence
        for p in range(self.__n_player):
            i = (self.__SB_index + p) % self.__n_player
            positions[i] = p + 1
        return positions


    def __fold(self):
        i = self.__whose_turn

        self.__update_record(act_name='fold', add=0, bet=0)

        self.__players.loc[i, 'folded'] = True
        self.__players.loc[i, 'reward'] = (self.__players.loc[i, 'chip_CU'] - self.__players.loc[i, 'initial_chip_CU']) * self.__CHIP_UNIT
        self.__players.loc[i, 'done'] = True
        self.__players.loc[i, 'step'] += 1
        self.__players.loc[i, 'eff_pot_CU'] = 0
        self.__players.loc[i, 'eff_pot'] = 0


    def __call(self):
        i = self.__whose_turn

        my_add_to_call_CU = self.__last_bet_CU - self.__players.loc[i, 'current_bet_CU']

        act_name = 'call' if self.__last_bet_CU > 0 else 'check'
        
        self.__players.loc[i, 'current_bet_CU'] += my_add_to_call_CU
        self.__players.loc[i, 'total_bet_CU'] += my_add_to_call_CU
        self.__players.loc[i, 'chip_CU'] -= my_add_to_call_CU
        self.__players.loc[i, 'line'] *= 2
        self.__players.loc[i, 'called'] = True

        self.__players.loc[i, 'current_bet'] = self.__players.loc[i, 'current_bet_CU'] * self.__CHIP_UNIT
        self.__players.loc[i, 'total_bet'] = self.__players.loc[i, 'total_bet_CU'] * self.__CHIP_UNIT
        self.__players.loc[i, 'chip'] = self.__players.loc[i, 'chip_CU'] * self.__CHIP_UNIT

        self.__players.loc[i, 'step'] += 1

        self.__update_record(act_name=act_name, add=my_add_to_call_CU * self.__CHIP_UNIT, bet=self.__players.loc[i, 'current_bet'])

        self.__count_eff_pot()


    def __raise(self, bet_size):
        i = self.__whose_turn
        my_eff_pot_CU = self.__players.loc[i, 'eff_pot_CU']
        my_raise_CU = np.round(my_eff_pot_CU * bet_size).astype(np.int_)

        self.__players.loc[i, 'current_bet_CU'] += my_raise_CU
        self.__players.loc[i, 'total_bet_CU'] += my_raise_CU
        self.__players.loc[i, 'chip_CU'] -= my_raise_CU

        self.__players.loc[i, 'current_bet'] = self.__players.loc[i, 'current_bet_CU'] * self.__CHIP_UNIT
        self.__players.loc[i, 'total_bet'] = self.__players.loc[i, 'total_bet_CU'] * self.__CHIP_UNIT
        self.__players.loc[i, 'chip'] = self.__players.loc[i, 'chip_CU'] * self.__CHIP_UNIT

        self.__update_record(act_name=f'{np.round(bet_size * 100).astype(np.int_)}%', add=my_raise_CU * self.__CHIP_UNIT, bet=self.__players.loc[i, 'current_bet'])

        self.__count_eff_pot()

        my_current_bet = self.__players.loc[i, 'current_bet_CU']
        self.__min_raise_CU = 2 * my_current_bet - self.__last_bet_CU
        self.__last_bet_CU = my_current_bet.copy()

        self.__players.loc[i, 'line'] = self.__players.loc[i, 'line'] * 2 + 1
        self.__players.loc[:, 'called'] = np.zeros(self.__MAX_N_PLAYER, dtype=np.bool_)
        self.__players.loc[:, 'raised'] = np.arange(self.__MAX_N_PLAYER) == i
        self.__players.loc[:, 'can_raise'] = self.__players.loc[:, 'playing'].copy()

        self.__players.loc[i, 'step'] += 1

    def __all_in(self):
        i = self.__whose_turn
        my_chip_CU = self.__players.loc[i, 'chip_CU']
        my_last_bet_CU = self.__players.loc[i, 'current_bet_CU']
        my_add_to_call_CU = self.__last_bet_CU - my_last_bet_CU
        my_add_to_min_raise_CU = self.__min_raise_CU - my_last_bet_CU

        self.__players.loc[i, 'current_bet_CU'] += my_chip_CU
        self.__players.loc[i, 'total_bet_CU'] += my_chip_CU
        self.__players.loc[i, 'chip_CU'] = 0
        self.__players.loc[i, 'all_ined'] = True

        self.__players.loc[i, 'current_bet'] = self.__players.loc[i, 'current_bet_CU'] * self.__CHIP_UNIT
        self.__players.loc[i, 'total_bet'] = self.__players.loc[i, 'total_bet_CU'] * self.__CHIP_UNIT
        self.__players.loc[i, 'chip'] = self.__players.loc[i, 'chip_CU'] * self.__CHIP_UNIT

        self.__update_record(act_name=f'all-in', add=my_chip_CU * self.__CHIP_UNIT, bet=self.__players.loc[i, 'current_bet'])

        self.__count_eff_pot()
        
        my_current_bet_CU = self.__players.loc[i, 'current_bet_CU']
        if my_chip_CU <= my_add_to_call_CU:  # all-in to call
            self.__players.loc[i, 'line'] *= 2
            self.__players.loc[i, 'called'] = True

        else:  # all-in to raise
            if my_chip_CU >= my_add_to_min_raise_CU:  # real raise
                self.__min_raise_CU = 2 * my_current_bet_CU - self.__last_bet_CU
                self.__players.loc[:, 'can_raise'] = self.__players.loc[:, 'playing'].copy()
            else:  # fake raise
                self.__players.loc[self.__players.loc[:, 'raised']|self.__players.loc[:, 'called'], 'can_raise'] = False

            self.__last_bet_CU = my_current_bet_CU.copy()
            self.__players.loc[i, 'line'] = self.__players.loc[i, 'line'] * 2 + 1
            self.__players.loc[:, 'called'] = np.zeros(self.__MAX_N_PLAYER, dtype=np.bool_)
            self.__players.loc[:, 'raised'] = np.arange(self.__MAX_N_PLAYER) == i

        self.__players.loc[i, 'step'] += 1

    def __update_record(self, act_name='fold', add=0, bet=0):
        i = self.__whose_turn

        self.__record.loc[len(self.__record)] = {
            'street': self.__street_name,
            'player': i,
            'position': POSITIONS[self.__n_player, self.__players.loc[i, 'position_index']],
            'eff_pot': self.__players.loc[i, 'eff_pot'],
            'act': act_name,
            'add': add,
            'bet': bet,
        }


    def __check_game_state(self):

        waiting = self.__players.loc[:, 'folded'] | self.__players.loc[:, 'all_ined']

        all_folded = self.__players.loc[:, 'folded'].sum() == (self.__n_player - 1)
        if all_folded:
            self.__check_winner()
            return None
        
        all_all_ined = waiting.sum() == self.__n_player
        if all_all_ined:
            self.__showdown = True
            self.__check_winner()
            return None
        
        all_called = (self.__players.loc[:, 'raised'] | self.__players.loc[:, 'called'] | waiting).sum() == self.__n_player
        if all_called:
            self.__players.loc[:, 'current_bet_CU'] = np.zeros(self.__MAX_N_PLAYER, dtype=np.int_)
            self.__players.loc[:, 'current_bet'] = np.zeros(self.__MAX_N_PLAYER, dtype=self.__DTYPE)
            self.__last_bet_CU = 0
            self.__min_raise_CU = 1

            self.__players.loc[:, 'called'] = np.zeros(self.__MAX_N_PLAYER, dtype=np.bool_)
            self.__players.loc[:, 'raised'] = np.zeros(self.__MAX_N_PLAYER, dtype=np.bool_)
            self.__players.loc[:, 'can_raise'] = self.__players.loc[:, 'playing'].copy()

            if self.__street_index == 3:
                self.__showdown = True
                self.__check_winner()
                return None
            
            else:    
                self.__street_index += 1
                self.__street_name = STREETS[self.__street_index]
                self.__board.loc[0:1+self.__street_index, 'index_'] = self.__board.loc[0:1+self.__street_index, 'pre_dealt_index']
                dealt = np.arange(5) < (self.__street_index + 2)
                self.__board.loc[:, 'value'] = ((self.__board.loc[:, 'index_'] - 1) // 4 + 2) * dealt
                board_suits = ((self.__board.loc[:, 'index_'] - 1) % 4 + 1) * dealt
                self.__board.loc[:, 'is_spade'] = board_suits == 1
                self.__board.loc[:, 'is_heart'] = board_suits == 2
                self.__board.loc[:, 'is_diamond'] = board_suits == 3
                self.__board.loc[:, 'is_club'] = board_suits == 4
                self.__board.loc[:, 'str'] = SIMPLE_CARDS[self.__board.loc[:, 'index_']]
                self.__board.loc[:, 'pretty_str'] = PRETTY_CARDS[self.__board.loc[:, 'index_']]
                if self.__n_player == 2:
                    i = self.__BB_index
                else:
                    i = self.__SB_index
                
        else:
            i = (self.__whose_turn + 1) % self.__n_player

        for j in range(0, self.__n_player):
            k = (i + j) % self.__n_player
            if ~waiting[k]:
                self.__whose_turn = k
                break


    def __check_winner(self):

        unfolded = ~self.__players.loc[:, 'folded'] & self.__players.loc[:, 'playing']
        player_ranks = np.zeros(self.__MAX_N_PLAYER, dtype=np.int_)

        if self.__showdown:  # if all players all-in or it's post-river, showdown is true
            
            for i, hole_cards in enumerate(self.__players[['card_0_index', 'card_1_index']].to_numpy()):
                if unfolded[i]:
                    card_combos = np.array(list(combinations(np.concatenate([hole_cards, self.__board.loc[:, 'pre_dealt_index']]), r=5)))
                    player_ranks[i] = get_rank(card_combos, return_highest=True)

            self.__players.loc[:, 'rank'] = player_ranks
            
            sorted_ranks = -np.sort(-np.unique(player_ranks))

            for winner_rank in sorted_ranks:

                winner_indices = self.__players.index[(player_ranks==winner_rank) & unfolded]

                winner_total_bets_CU = np.sort(self.__players.loc[winner_indices, 'total_bet_CU'])
                winner_sub_indices = np.argsort(self.__players.loc[winner_indices, 'total_bet_CU'])

                n = len(winner_sub_indices)

                for i in winner_sub_indices:
                    gain_CU = 0
                    for j in self.__players.index[self.__players.loc[:, 'playing']]:
                        if winner_total_bets_CU[i] <= self.__players.loc[:, 'total_bet_CU'][j]: # covered by player_j
                            gain_from_player_j_CU = winner_total_bets_CU[i] // (n - i)
                        else:
                            gain_from_player_j_CU = self.__players.loc[:, 'total_bet_CU'][j] // (n - i)
                        self.__players.loc[j, 'total_bet_CU'] -= gain_from_player_j_CU
                        gain_CU += gain_from_player_j_CU

                    self.__players.loc[winner_indices[i], 'chip_CU'] += gain_CU
                
                if not self.__players.loc[:, 'total_bet_CU'].any():
                    break

        else: # if all fold to one
            winner_index = self.__players.index[unfolded]
            self.__players.loc[winner_index, 'chip_CU'] += self.__players.loc[:, 'total_bet_CU'].sum()
            self.__players.loc[:, 'total_bet_CU'] = np.zeros(self.__MAX_N_PLAYER, dtype=np.int_)

        self.__players.assign(rank=player_ranks)

        self.__players.loc[:, 'current_bet_CU'] = np.zeros(self.__MAX_N_PLAYER, dtype=np.int_)

        self.__players.loc[:, 'chip'] = self.__players.loc[:, 'chip_CU'] * self.__CHIP_UNIT
        self.__players.loc[:, 'total_bet'] = np.zeros(self.__MAX_N_PLAYER, dtype=self.__DTYPE)
        self.__players.loc[:, 'current_bet'] = np.zeros(self.__MAX_N_PLAYER, dtype=self.__DTYPE)

        self.__players.loc[:, 'reward'] = (self.__players.loc[:, 'chip_CU'] - self.__players.loc[:, 'initial_chip_CU']) * self.__CHIP_UNIT
        self.__players.loc[:, 'done'] = np.ones(self.__MAX_N_PLAYER, dtype=np.bool_)

        self.__over = True

    def __count_eff_pot(self):
        unfolded = ~self.__players.loc[:, 'folded'] & self.__players.loc[:, 'playing']
        for i in range(self.__n_player):
            if unfolded[i]:
                my_initial_chips_CU = self.__players.loc[i, 'chip_CU']
                covering_me = self.__players.loc[:, 'total_bet_CU'] >= my_initial_chips_CU
                my_eff_pot_CU = covering_me.sum() * my_initial_chips_CU + self.__players.loc[:, 'total_bet_CU'][~covering_me].sum()
                self.__players.loc[i, 'eff_pot_CU'] = my_eff_pot_CU
            # else:
            #     self.__players.loc[i, 'eff_pot_CU'] = 0
        self.__players.loc[:, 'eff_pot'] = self.__players.loc[:, 'eff_pot_CU'] * self.__CHIP_UNIT

    @property
    def board(self):
        return self.__board.copy()
    
    @property
    def players(self):
        return self.__players.copy()
    
    @property
    def record(self):
        return self.__record.copy()
    
    @property
    def over(self):
        return self.__over
    
    @property
    def whose_turn(self):
        return self.__whose_turn
    
    @property
    def dtype(self):
        return self.__DTYPE
    
    @property
    def n_obs(self):
        return self.__N_OBS
    
    @property
    def n_act(self):
        return self.__N_ACT
    
    @property
    def acts(self):
        return self.__ACTS
    
    @property
    def n_player(self):
        return self.__n_player
    
    @property
    def max_n_player(self):
        return self.__MAX_N_PLAYER



game = Game(max_n_player=6)

for _ in range(1):

    print('\n============================================================================')
    
    game.reset(hole_cards=[[1, 2], [3, 4]], pre_dealt_board=[5, 6, 7, 8, 9])

    # while not game.over:
    #     step, obs, reward, done, valid_act_filter = game.to_player_view()

    #     # print(step, obs, reward, done, valid_act_filter)

    #     valid_act_filter[0, 2] = 0

    #     n = 22
    #     act = np.random.choice(np.arange(n)[valid_act_filter[0, 0:n]])
        
    #     game.receive_act(act)

    # print(game.record, '\n')
    # print(game.board[['pre_dealt_pretty_str']].T, '\n')
    # print(game.players[['position', 'card_0_pretty_str', 'card_1_pretty_str', 'reward']].T)


# %%
