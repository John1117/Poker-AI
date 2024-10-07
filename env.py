# %%
import numpy as np
import pandas as pd
from itertools import combinations



# %%
CARD_INDICES_TO_NAMES = np.array([
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
CARD_NAMES_TO_INDICES = pd.DataFrame(
    {k: [i] for i, k in enumerate(CARD_INDICES_TO_NAMES)}
)

# %%
CARD_INDICES_TO_PRETTY_NAMES = np.array([
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

POSITION_INDICES_TO_NAMES = np.array(
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


STREET_INDICES_TO_NAMES = np.array(['pre-flop', 'flop', 'turn', 'river'])


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


def get_bet_sizes_of_equal_spacing_odds(odd_incr=0.05, dtype=np.float_):
    odds = np.arange(start=odd_incr, step=odd_incr, stop=1, dtype=dtype)
    bet_sizes = odds / (1 - odds)
    return bet_sizes


# %%
class Game():

    def __init__(
        self, 
        max_n_player: int = 6, 
        n_player: int = 6, 
        bet_sizes: np.ndarray = get_bet_sizes_of_equal_spacing_odds(0.05),
        chip_unit: float = 0.1,
        dtype: np.dtype = np.float_
    ):

        self.dtype = dtype
        
        self.n_player = n_player
        self.max_n_player = max_n_player
        

        self.observed_board_column_names = [
            'value',
            'is_spade',
            'is_heart',
            'is_diamond', 
            'is_club',
        ]
        self.observed_player_i_column_names = [
            'card_1_value',
            'card_2_value',
            'card_1_is_spade',
            'card_2_is_spade',
            'card_1_is_heart',
            'card_2_is_heart',
            'card_1_is_diamond',
            'card_2_is_diamond',
            'card_1_is_club',
            'card_2_is_club',
            'position_index',
        ]
        self.observed_player_j_column_names = [
            'effective_pot',
            'bet',
            'total_bet',
            'chip',
            'line',
            'folded',
        ]
        self.n_observation = len(self.observed_board_column_names) * 5 + len(self.observed_player_i_column_names) + len(self.observed_player_j_column_names) * self.max_n_player


        self.n_default_action = 3  # fold, call, all-in
        self.n_action = self.n_default_action + len(bet_sizes)
        self.actions = np.arange(self.n_action, dtype=np.int_)
        self.bet_sizes = bet_sizes.astype(self.dtype)

        if chip_unit is None:
            chip_unit = 0.1
        self.chip_unit = np.array(chip_unit, self.dtype)  # 0.1BB
        

        self.n_game = 0
        self.small_blind_index = self.n_player - 1

        self.players = pd.DataFrame(
            data={'playing': np.arange(self.max_n_player) < self.n_player},
            index=pd.RangeIndex(self.max_n_player, name='player_index')
        )


    def reset(
            self,
            small_blind_index: int =None,
            hole_cards: list[list[str]]=None,
            board: list[str]=None,
            initial_chips: np.ndarray=None
        ):

        self.n_game += 1
        self.street_index = 0
        self.street_name = STREET_INDICES_TO_NAMES[self.street_index]
        self.showdown = False
        self.over = False


        if small_blind_index is None:
            self.small_blind_index = (self.small_blind_index + 1) % self.n_player
        else:
            self.small_blind_index = self.small_blind_index
        self.big_blind_index = (self.small_blind_index + 1) % self.n_player
        self.whose_turn = (self.small_blind_index + 2) % self.n_player


        deck = np.random.permutation(52) + 1  # reserve 0 for no card
        if hole_cards is None:
            hole_cards_1d = np.zeros(self.max_n_player * 2, dtype=np.int_)
        else:
            hole_cards_1d = np.ravel(hole_cards)
            hole_cards_1d = CARD_NAMES_TO_INDICES[hole_cards_1d].to_numpy()[0]
        undealt_mask = hole_cards_1d == 0
        dealt_hole_cards = hole_cards_1d[~undealt_mask]
        deck = np.setdiff1d(deck, dealt_hole_cards, assume_unique=True)
        n_undealt = undealt_mask.sum()
        hole_cards_1d[undealt_mask], deck = deck[:n_undealt], deck[n_undealt:]
        hole_cards = hole_cards_1d.reshape((self.max_n_player, 2))
        hole_cards = np.sort(hole_cards, axis=1)

        if board is None:
            board = np.zeros(5, dtype=np.int_)
        else:
            board = CARD_NAMES_TO_INDICES[board].to_numpy()[0]
        undealt_mask = board == 0
        dealt_board = board[~undealt_mask]
        deck = np.setdiff1d(deck, dealt_board, assume_unique=True)
        n_undealt = undealt_mask.sum()
        board[undealt_mask], deck = deck[:n_undealt], deck[n_undealt:]
        dealt_board = np.concatenate((np.sort(board[:3]), board[3:]))

        hole_card_values = (hole_cards - 1) // 4 + 2
        hole_card_suits = (hole_cards - 1) % 4 + 1

        dealt_board_values = (dealt_board - 1) // 4 + 2
        dealt_board_suits = (dealt_board - 1) % 4 + 1


        self.board = pd.DataFrame(
            {
                'index_': np.zeros(5, dtype=np.int_),
                'value': np.zeros(5, dtype=np.int_),
                'is_spade': np.zeros(5, dtype=np.bool_),
                'is_heart': np.zeros(5, dtype=np.bool_),
                'is_diamond': np.zeros(5, dtype=np.bool_),
                'is_club': np.zeros(5, dtype=np.bool_),
                'name': np.zeros(5, dtype=np.str_),
                'pretty_name': np.zeros(5, dtype=np.str_),
                'dealt_index': dealt_board,
                'dealt_value': dealt_board_values,
                'dealt_is_spade': dealt_board_suits == 1,
                'dealt_is_heart': dealt_board_suits == 2,
                'dealt_is_diamond': dealt_board_suits == 3,
                'dealt_is_club': dealt_board_suits == 4,
                'dealt_name': CARD_INDICES_TO_NAMES[dealt_board],
                'dealt_pretty_name': CARD_INDICES_TO_PRETTY_NAMES[dealt_board],
            },
            index=pd.RangeIndex(5, name='board_index')
        )

        if initial_chips is None:
            initial_chips = 100 * self.players.loc[:, 'playing'].astype(self.dtype)
        initial_chips_cu = np.round(initial_chips / self.chip_unit).astype(np.int_)

        blind_bets = np.zeros(self.max_n_player, dtype=self.dtype)
        blind_bets[[self.small_blind_index, self.big_blind_index]] = 0.5, 1
        blind_bets_cu = np.round(blind_bets / self.chip_unit).astype(np.int_)

        self.last_bet_cu = np.round(1 / self.chip_unit).astype(np.int_)
        self.min_raise_cu = np.round(2 / self.chip_unit).astype(np.int_)

        effective_pots = np.repeat(blind_bets.sum(), self.max_n_player) * self.players.loc[:, 'playing']
        effective_pots_cu = np.round(effective_pots / self.chip_unit).astype(np.int_)

        chips = initial_chips - blind_bets
        chips_cu = initial_chips_cu - blind_bets_cu

        position_indices = self.assign_position()

        self.players = self.players.assign(
            **{ 
                # observation
                'card_1_value': hole_card_values[:, 0],
                'card_2_value': hole_card_values[:, 1],
                'card_1_is_spade': hole_card_suits[:, 0] == 1,
                'card_2_is_spade': hole_card_suits[:, 1] == 1,
                'card_1_is_heart': hole_card_suits[:, 0] == 2,
                'card_2_is_heart': hole_card_suits[:, 1] == 2,
                'card_2_is_diamond': hole_card_suits[:, 1] == 3,
                'card_1_is_diamond': hole_card_suits[:, 0] == 3,
                'card_1_is_club': hole_card_suits[:, 0] == 4,
                'card_2_is_club': hole_card_suits[:, 1] == 4,

                'position_index': position_indices,
                'effective_pot': effective_pots,
                'bet': blind_bets,
                'total_bet': blind_bets,
                'chip': chips,

                'line': self.players.loc[:, 'playing'].astype(np.int_),
                'folded': np.zeros(self.max_n_player, dtype=np.bool_),

                # view
                'card_1_index': hole_cards[:, 0],
                'card_2_index': hole_cards[:, 1],
                'card_1_name': CARD_INDICES_TO_NAMES[hole_cards[:, 0]],
                'card_2_name': CARD_INDICES_TO_NAMES[hole_cards[:, 1]],
                'card_1_pretty_name': CARD_INDICES_TO_PRETTY_NAMES[hole_cards[:, 0]],
                'card_2_pretty_name': CARD_INDICES_TO_PRETTY_NAMES[hole_cards[:, 1]],

                'position': POSITION_INDICES_TO_NAMES[self.n_player, position_indices],
                
                'initial_chip': initial_chips,

                'effective_pot_cu': effective_pots_cu,
                'bet_cu': blind_bets_cu,
                'total_bet_cu': blind_bets_cu,
                'chip_cu': chips_cu,
                'initial_chip_cu': initial_chips_cu,

                'all_ined': np.zeros(self.max_n_player, dtype=np.bool_),
                'called': np.zeros(self.max_n_player, dtype=np.bool_),
                'raised': np.zeros(self.max_n_player, dtype=np.bool_),
                'can_raise': self.players.loc[:, 'playing'],
                
                'rank': np.zeros(self.max_n_player, dtype=np.int_),

                'reward': np.zeros((self.max_n_player, 1), dtype=self.dtype),
                'done': np.zeros((self.max_n_player, 1), dtype=np.bool_),
                'step': np.zeros((self.max_n_player, 1), dtype=np.int_),
            }
        )

        self.record = pd.DataFrame(
            {
                'street': np.array(['pre-flop', 'pre-flop']),
                'player': self.players.index.to_numpy()[[self.small_blind_index, self.big_blind_index]],
                'position': POSITION_INDICES_TO_NAMES[self.n_player, position_indices[[self.small_blind_index, self.big_blind_index]]],
                'effective_pot': np.array([0, 0.5], dtype=self.dtype),
                'action': np.zeros(2, dtype=np.str_),
                'add': np.array([0.5, 1], dtype=self.dtype),
                'bet': np.array([0.5, 1], dtype=self.dtype),
            },
            index=pd.RangeIndex(2, name='record_index')
        )

    
    def receive_action(self, action: int):

        if self.over:
            self.reset()

        valid_action_filter = self.check_valid_action(self.whose_turn)
        # if action isn't valid then fold/check
        action = 1 - valid_action_filter[0] if valid_action_filter[action] == False else action
                
        if action == 0:
            self.fold()
        elif action == 1:
            self.call()
        elif action == 2:
            self.all_in()
        else:
            self.raise_(self.bet_sizes[action - self.n_default_action])

        self.check_game_state()

        return action


    def to_player_view(self, i: int=None):
        i = self.whose_turn if i is None else i

        observed_board_data = self.board.loc[:, self.observed_board_column_names].to_numpy().T.reshape(-1).astype(self.dtype)
        observed_player_i_data = self.players.loc[i, self.observed_player_i_column_names].to_numpy().reshape(-1).astype(self.dtype)
        observed_player_j_data = self.players.loc[:, self.observed_player_j_column_names].to_numpy().T.reshape(-1).astype(self.dtype)
        observation = np.concatenate((observed_board_data, observed_player_i_data, observed_player_j_data)).reshape(1, -1)
        reward = self.players.loc[i, 'reward'].reshape(1, 1).copy()
        done = self.players.loc[i, 'done'].reshape(1, 1).copy()
        step = self.players.loc[i, 'step'].reshape(1, 1).copy()

        valid_action_filter = self.check_valid_action(i).reshape(1, -1)

        return step, observation, reward, done, valid_action_filter


    def check_valid_action(self, i: int):

        if i != self.whose_turn or self.over:
            return np.zeros(self.n_action, np.bool_)
        
        my_effective_pot_cu = self.players.loc[i, 'effective_pot_cu']
        my_last_bet_cu = self.players.loc[i, 'bet_cu']
        my_chip_cu = self.players.loc[i, 'chip_cu']
        max_opp_chip_cu = self.players.loc[self.players.loc[:, 'chip_cu'] != my_chip_cu, 'chip_cu'].max()

        my_add_to_call_cu = self.last_bet_cu - my_last_bet_cu
        my_add_to_min_raise_cu = self.min_raise_cu - my_last_bet_cu
        my_possible_bets_cu = np.round(my_effective_pot_cu * self.bet_sizes).astype(np.int_)

        can_fold = self.last_bet_cu > 0
        can_call = my_chip_cu > my_add_to_call_cu
        can_all_in = True

        if self.players.loc[i, 'can_raise']:
            valid_bet_filter = (my_add_to_min_raise_cu <= my_possible_bets_cu) & (my_possible_bets_cu < min(my_chip_cu, max_opp_chip_cu))
        else:
            valid_bet_filter = np.zeros_like(my_possible_bets_cu, dtype=np.bool_)
            can_all_in = False

        valid_action_filter = np.concatenate(((can_fold, can_call, can_all_in), valid_bet_filter))
        return valid_action_filter
    
    def assign_position(self):
        positions = np.zeros(self.max_n_player, np.int_)  # reserve 0 for absence
        for p in range(self.n_player):
            i = (self.small_blind_index + p) % self.n_player
            positions[i] = p + 1
        return positions


    def fold(self):
        i = self.whose_turn

        self.update_record(act_name='fold', add=0, bet=0)

        self.players.loc[i, 'folded'] = True
        self.players.loc[i, 'reward'] = (self.players.loc[i, 'chip_cu'] - self.players.loc[i, 'initial_chip_cu']) * self.chip_unit
        self.players.loc[i, 'done'] = True
        self.players.loc[i, 'step'] += 1
        self.players.loc[i, 'effective_pot_cu'] = 0
        self.players.loc[i, 'effective_pot'] = 0


    def call(self):
        i = self.whose_turn

        my_add_to_call_cu = self.last_bet_cu - self.players.loc[i, 'bet_cu']

        act_name = 'call' if self.last_bet_cu > 0 else 'check'
        
        self.players.loc[i, 'bet_cu'] += my_add_to_call_cu
        self.players.loc[i, 'total_bet_cu'] += my_add_to_call_cu
        self.players.loc[i, 'chip_cu'] -= my_add_to_call_cu
        self.players.loc[i, 'line'] *= 2
        self.players.loc[i, 'called'] = True

        self.players.loc[i, 'bet'] = self.players.loc[i, 'bet_cu'] * self.chip_unit
        self.players.loc[i, 'total_bet'] = self.players.loc[i, 'total_bet_cu'] * self.chip_unit
        self.players.loc[i, 'chip'] = self.players.loc[i, 'chip_cu'] * self.chip_unit

        self.players.loc[i, 'step'] += 1

        self.update_record(act_name=act_name, add=my_add_to_call_cu * self.chip_unit, bet=self.players.loc[i, 'bet'])

        self.count_effective_pot()


    def raise_(self, bet_size: float):
        i = self.whose_turn
        my_effective_pot_cu = self.players.loc[i, 'effective_pot_cu']
        my_raise_cu = np.round(my_effective_pot_cu * bet_size).astype(np.int_)

        self.players.loc[i, 'bet_cu'] += my_raise_cu
        self.players.loc[i, 'total_bet_cu'] += my_raise_cu
        self.players.loc[i, 'chip_cu'] -= my_raise_cu

        self.players.loc[i, 'bet'] = self.players.loc[i, 'bet_cu'] * self.chip_unit
        self.players.loc[i, 'total_bet'] = self.players.loc[i, 'total_bet_cu'] * self.chip_unit
        self.players.loc[i, 'chip'] = self.players.loc[i, 'chip_cu'] * self.chip_unit

        self.update_record(act_name=f'bet {np.round(bet_size * 100).astype(np.int_)}%', add=my_raise_cu * self.chip_unit, bet=self.players.loc[i, 'bet'])

        self.count_effective_pot()

        my_bet = self.players.loc[i, 'bet_cu']
        self.min_raise_cu = 2 * my_bet - self.last_bet_cu
        self.last_bet_cu = my_bet.copy()

        self.players.loc[i, 'line'] = self.players.loc[i, 'line'] * 2 + 1
        self.players.loc[:, 'called'] = np.zeros(self.max_n_player, dtype=np.bool_)
        self.players.loc[:, 'raised'] = np.arange(self.max_n_player) == i
        self.players.loc[:, 'can_raise'] = self.players.loc[:, 'playing'].copy()

        self.players.loc[i, 'step'] += 1

    def all_in(self):
        i = self.whose_turn
        my_chip_cu = self.players.loc[i, 'chip_cu']
        my_last_bet_cu = self.players.loc[i, 'bet_cu']
        my_add_to_call_cu = self.last_bet_cu - my_last_bet_cu
        my_add_to_min_raise_cu = self.min_raise_cu - my_last_bet_cu

        self.players.loc[i, 'bet_cu'] += my_chip_cu
        self.players.loc[i, 'total_bet_cu'] += my_chip_cu
        self.players.loc[i, 'chip_cu'] = 0
        self.players.loc[i, 'all_ined'] = True

        self.players.loc[i, 'bet'] = self.players.loc[i, 'bet_cu'] * self.chip_unit
        self.players.loc[i, 'total_bet'] = self.players.loc[i, 'total_bet_cu'] * self.chip_unit
        self.players.loc[i, 'chip'] = self.players.loc[i, 'chip_cu'] * self.chip_unit

        self.update_record(act_name=f'all-in', add=my_chip_cu * self.chip_unit, bet=self.players.loc[i, 'bet'])

        self.count_effective_pot()
        
        my_bet_cu = self.players.loc[i, 'bet_cu']
        if my_chip_cu <= my_add_to_call_cu:  # all-in to call
            self.players.loc[i, 'line'] *= 2
            self.players.loc[i, 'called'] = True

        else:  # all-in to raise
            if my_chip_cu >= my_add_to_min_raise_cu:  # real raise
                self.min_raise_cu = 2 * my_bet_cu - self.last_bet_cu
                self.players.loc[:, 'can_raise'] = self.players.loc[:, 'playing'].copy()
            else:  # fake raise
                self.players.loc[self.players.loc[:, 'raised']|self.players.loc[:, 'called'], 'can_raise'] = False

            self.last_bet_cu = my_bet_cu.copy()
            self.players.loc[i, 'line'] = self.players.loc[i, 'line'] * 2 + 1
            self.players.loc[:, 'called'] = np.zeros(self.max_n_player, dtype=np.bool_)
            self.players.loc[:, 'raised'] = np.arange(self.max_n_player) == i

        self.players.loc[i, 'step'] += 1

    def update_record(self, act_name: str = 'fold', add: float = 0, bet: float = 0):
        i = self.whose_turn

        self.record.loc[len(self.record)] = {
            'street': self.street_name,
            'player': i,
            'position': POSITION_INDICES_TO_NAMES[self.n_player, self.players.loc[i, 'position_index']],
            'effective_pot': self.players.loc[i, 'effective_pot'],
            'action': act_name,
            'add': add,
            'bet': bet,
        }


    def check_game_state(self):

        waiting = self.players.loc[:, 'folded'] | self.players.loc[:, 'all_ined']

        all_folded = self.players.loc[:, 'folded'].sum() == (self.n_player - 1)
        if all_folded:
            self.check_winner()
            return None
        
        all_all_ined = waiting.sum() == self.n_player
        if all_all_ined:
            self.showdown = True
            self.check_winner()
            return None
        
        all_called = (self.players.loc[:, 'raised'] | self.players.loc[:, 'called'] | waiting).sum() == self.n_player
        if all_called:
            self.players.loc[:, 'bet_cu'] = np.zeros(self.max_n_player, dtype=np.int_)
            self.players.loc[:, 'bet'] = np.zeros(self.max_n_player, dtype=self.dtype)
            self.last_bet_cu = 0
            self.min_raise_cu = 1

            self.players.loc[:, 'called'] = np.zeros(self.max_n_player, dtype=np.bool_)
            self.players.loc[:, 'raised'] = np.zeros(self.max_n_player, dtype=np.bool_)
            self.players.loc[:, 'can_raise'] = self.players.loc[:, 'playing'].copy()

            if self.street_index == 3:
                self.showdown = True
                self.check_winner()
                return None
            
            else:    
                self.street_index += 1
                self.street_name = STREET_INDICES_TO_NAMES[self.street_index]
                self.board.loc[0:1+self.street_index, 'index_'] = self.board.loc[0:1+self.street_index, 'dealt_index']
                dealt = np.arange(5) < (self.street_index + 2)
                self.board.loc[:, 'value'] = ((self.board.loc[:, 'index_'] - 1) // 4 + 2) * dealt
                board_suits = ((self.board.loc[:, 'index_'] - 1) % 4 + 1) * dealt
                self.board.loc[:, 'is_spade'] = board_suits == 1
                self.board.loc[:, 'is_heart'] = board_suits == 2
                self.board.loc[:, 'is_diamond'] = board_suits == 3
                self.board.loc[:, 'is_club'] = board_suits == 4
                self.board.loc[:, 'name'] = CARD_INDICES_TO_NAMES[self.board.loc[:, 'index_']]
                self.board.loc[:, 'pretty_name'] = CARD_INDICES_TO_PRETTY_NAMES[self.board.loc[:, 'index_']]
                if self.n_player == 2:
                    i = self.big_blind_index
                else:
                    i = self.small_blind_index
                
        else:
            i = (self.whose_turn + 1) % self.n_player

        for j in range(0, self.n_player):
            k = (i + j) % self.n_player
            if not waiting[k]:
                self.whose_turn = k
                break


    def check_winner(self):

        unfolded = ~self.players.loc[:, 'folded'] & self.players.loc[:, 'playing']
        player_ranks = np.zeros(self.max_n_player, dtype=np.int_)

        if self.showdown:  # if all players all-in or it's post-river, showdown is true
            
            for i, hole_cards in enumerate(self.players[['card_1_index', 'card_2_index']].to_numpy()):
                if unfolded[i]:
                    card_combos = np.array(list(combinations(np.concatenate([hole_cards, self.board.loc[:, 'dealt_index']]), r=5)))
                    player_ranks[i] = get_rank(card_combos, return_highest=True)

            self.players.loc[:, 'rank'] = player_ranks
            
            sorted_ranks = -np.sort(-np.unique(player_ranks))

            for winner_rank in sorted_ranks:

                winner_indices = self.players.index[(player_ranks==winner_rank) & unfolded]

                winner_total_bets_cu = np.sort(self.players.loc[winner_indices, 'total_bet_cu'])
                winner_sub_indices = np.argsort(self.players.loc[winner_indices, 'total_bet_cu'])

                n = len(winner_sub_indices)

                for i in winner_sub_indices:
                    gain_cu = 0
                    for j in self.players.index[self.players.loc[:, 'playing']]:
                        if winner_total_bets_cu[i] <= self.players.loc[:, 'total_bet_cu'][j]: # covered by player_j
                            gain_from_player_j_cu = winner_total_bets_cu[i] // (n - i)
                        else:
                            gain_from_player_j_cu = self.players.loc[:, 'total_bet_cu'][j] // (n - i)
                        self.players.loc[j, 'total_bet_cu'] -= gain_from_player_j_cu
                        gain_cu += gain_from_player_j_cu

                    self.players.loc[winner_indices[i], 'chip_cu'] += gain_cu
                
                if not self.players.loc[:, 'total_bet_cu'].any():
                    break

        else: # if all fold to one
            winner_index = self.players.index[unfolded]
            self.players.loc[winner_index, 'chip_cu'] += self.players.loc[:, 'total_bet_cu'].sum()
            self.players.loc[:, 'total_bet_cu'] = np.zeros(self.max_n_player, dtype=np.int_)

        self.players.assign(rank=player_ranks)

        self.players.loc[:, 'bet_cu'] = np.zeros(self.max_n_player, dtype=np.int_)

        self.players.loc[:, 'chip'] = self.players.loc[:, 'chip_cu'] * self.chip_unit
        self.players.loc[:, 'total_bet'] = np.zeros(self.max_n_player, dtype=self.dtype)
        self.players.loc[:, 'bet'] = np.zeros(self.max_n_player, dtype=self.dtype)

        self.players.loc[:, 'reward'] = (self.players.loc[:, 'chip_cu'] - self.players.loc[:, 'initial_chip_cu']) * self.chip_unit
        self.players.loc[:, 'done'] = np.ones(self.max_n_player, dtype=np.bool_)

        self.over = True

    def count_effective_pot(self):
        unfolded = ~self.players.loc[:, 'folded'] & self.players.loc[:, 'playing']
        for i in range(self.n_player):
            if unfolded[i]:
                my_initial_chips_cu = self.players.loc[i, 'initial_chip_cu']
                covering_me = self.players.loc[:, 'total_bet_cu'] >= my_initial_chips_cu
                my_effective_pot_cu = covering_me.sum() * my_initial_chips_cu + self.players.loc[:, 'total_bet_cu'][~covering_me].sum()
                self.players.loc[i, 'effective_pot_cu'] = my_effective_pot_cu
            # else:
            #     self.__players.loc[i, 'effective_pot_cu'] = 0
        self.players.loc[:, 'effective_pot'] = self.players.loc[:, 'effective_pot_cu'] * self.chip_unit




game = Game(n_player=6, max_n_player=6)

for _ in range(1):

    print('\n============================================================================')
    hole_cards = [
        ['4s', '5s'],
        ['6h', '7h'],
        ['As', ''],
        ['', ''],
        ['', ''],
        ['', ''],
    ]
    game.reset(board=['2s', '3s', '', '', ''], hole_cards=hole_cards)

    while not game.over:
        step, observation, reward, done, valid_action_filter = game.to_player_view()

        valid_action_filter[0, 2] = 0

        n = 22
        action = np.random.choice(np.arange(n)[valid_action_filter[0, 0:n]])
        
        game.receive_action(action)

    print(game.record, '\n')
    print(game.board[['dealt_pretty_name']].T, '\n')
    print(game.players[['position', 'card_1_pretty_name', 'card_2_pretty_name', 'reward']].T)

# %%
