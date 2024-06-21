import logging, time
from copy import deepcopy

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand, TransferCommand
from dicewars.ai.kb.move_selection import get_transfer_from_endangered, get_transfer_to_border
from dicewars.ai.dt import stei
from ..utils import *

from .nnmodel import NeuralNetwork
import torch


class AI:
    """Agent combining STEi with aggressive transfers.

    First, it tries to transfer forces towards borders, then it proceeds
    with attacks following the dt.stei AI.
    """
    def __init__(self, player_name, board, players_order, max_transfers):
        self.player_name = player_name
        self.logger = logging.getLogger('AI')
        self.logger.setLevel(logging.DEBUG)
        self.max_transfers = max_transfers
        self.players_order = players_order

        self.stei = stei.AI(player_name, board, players_order, max_transfers)
        self.reserved_evacs = 0

        nb_players = board.nb_players_alive()
        self.logger.info('Setting up for {}-player game'.format(nb_players))
        if nb_players == 2:
            self.treshold = 0.2
            self.score_weight = 3
        else:
            self.treshold = 0.4
            self.score_weight = 2


        inputWidth = 7
        input_model_filename = 'dicewars/ai/xtodo00/cpu_model_xtodo00_2000_epochs_20.pt'
        self.model = NeuralNetwork(inputWidth)
        self.model.load_state_dict(torch.load(input_model_filename))

        #time measuring
        self.last_maxN_search_time = 0

        self.stage = 'attack'

    def ai_turn(self, board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):
        if nb_transfers_this_turn + self.reserved_evacs < self.max_transfers:
            transfer = get_transfer_to_border(board, self.player_name)
            if transfer:
                return TransferCommand(transfer[0], transfer[1])
        else:
            self.logger.debug(f'Already did {nb_transfers_this_turn}/{self.max_transfers} transfers, reserving {self.reserved_evacs} for evac, skipping further aggresive ones')

        if self.stage == 'attack':
            self.board = board
            self.logger.debug("Looking for possible attacks.")
            self.get_largest_region()

            # stei_move = self.stei.ai_turn(board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left)
            # if turns:
            #     turn = turns[0]
            #     self.logger.debug("Possible turn: {}".format(turn))
            #     hold_prob = turn[3]
            #     self.logger.debug("{0}->{1} attack and hold probabiliy {2}".format(turn[0], turn[1], hold_prob))

            #     return BattleCommand(turn[0], turn[1])
            # else:
            #     self.stage = 'evac'

            self.logger.debug(f"{time_left=}")

            if time_left > 1:
                # depth = board.nb_players_alive()
                if time_left > 10:
                    depth = 8
                elif time_left > 5:
                    depth = 4
                else:
                    depth = board.nb_players_alive()
                
                self.logger.debug(f"Starting a maxN search (depth={depth})")
                T1 = time.time()
                attack, scores = self.maxN(board, depth, self.player_name, self.players_order, time_left)
                T2 = time.time()
                self.last_maxN_search_time = T2 - T1
                self.logger.debug(f"MaxN to depth {depth} took {self.last_maxN_search_time} seconds")
            else:
                turns = self.possible_turns()
                if turns:
                    attack =  turns[0]
                else:
                    attack = None

            if attack:
                if self.score_board(self.simulate_successful_attack(board, attack[0], attack[1]), self.player_name, time_left) > self.score_board(board, self.player_name, time_left):
                    self.logger.debug(f"Performing attack")
                    return BattleCommand(attack[0], attack[1])
                else:
                    self.stage = 'evac'
            else:
                self.stage = 'evac'


        if self.stage == 'evac':
            self.logger.debug(f"Moving to evac")
            if nb_transfers_this_turn < self.max_transfers:
                transfer = get_transfer_from_endangered(board, self.player_name)
                if transfer:
                    return TransferCommand(transfer[0], transfer[1])
            else:
                self.logger.debug(f'Already did {nb_transfers_this_turn}/{self.max_transfers} transfers, skipping further')

        self.stage = 'attack'
        self.logger.debug(f"Ending Turn")
        return EndTurnCommand()

    def possible_turns(self):
        """Find possible turns with hold higher hold probability than treshold

        This method returns list of all moves with probability of holding the area
        higher than the treshold or areas with 8 dice. In addition, it includes
        the preference of these moves. The list is sorted in descending order with
        respect to preference * hold probability
        """
        turns = []
        for source, target in possible_attacks(self.board, self.player_name):
            atk_power = source.get_dice()
            atk_prob = probability_of_successful_attack(self.board, source.get_name(), target.get_name())
            hold_prob = atk_prob * probability_of_holding_area(self.board, target.get_name(), atk_power - 1, self.player_name)
            if hold_prob >= self.treshold or atk_power == 8:
                preference = hold_prob
                if source.get_name() in self.largest_region:
                    preference *= self.score_weight
                turns.append([source.get_name(), target.get_name(), preference, hold_prob])

        return sorted(turns, key=lambda turn: turn[2], reverse=True)

    def reasonable_turns(self, board, player_name):
        turns = []
        for source, target in possible_attacks(board, player_name):
            atk_power = source.get_dice()
            atk_prob = probability_of_successful_attack(board, source.get_name(), target.get_name())
            hold_prob = atk_prob * probability_of_holding_area(board, target.get_name(), atk_power - 1, self.player_name)
            if hold_prob >= self.treshold or atk_power == 8:
                preference = hold_prob
                if source.get_name() in self.get_largest_region_for_player(board, player_name):
                    preference *= self.score_weight
                turns.append([source.get_name(), target.get_name(), preference, hold_prob])

        return sorted(turns, key=lambda turn: turn[2], reverse=True)

    def get_largest_region(self):
        """Get size of the largest region, including the areas within

        Attributes
        ----------
        largest_region : list of int
            Names of areas in the largest region

        Returns
        -------
        int
            Number of areas in the largest region
        """
        self.largest_region = []

        players_regions = self.board.get_players_regions(self.player_name)
        max_region_size = max(len(region) for region in players_regions)
        max_sized_regions = [region for region in players_regions if len(region) == max_region_size]

        self.largest_region = max_sized_regions[0]
        return max_region_size

    def get_largest_region_for_player(self, board, player_name):
        players_regions = board.get_players_regions(player_name)
        max_region_size = max(len(region) for region in players_regions)
        max_sized_regions = [region for region in players_regions if len(region) == max_region_size]

        return max_sized_regions[0]

    def score_board(self, board, player_name, time_left):
        return float(len(board.get_player_areas(player_name)))

    def score_heuristic(self, board, player_name, time_left):
        with torch.no_grad():
            features = parse_board(player_name, board)
            score = self.model(torch.FloatTensor([features]))
        return score
        # return float(len(board.get_player_areas(player_name)))


    def maxN(self, board, depth, player_to_move, players_order, time_left):
        if depth == 0:
            return (None, [self.score_heuristic(board, player, time_left) for player in players_order])
        
        optimal_index = -1
        optimal_move = None
        optimal_scores = tuple([float('-inf') for _ in players_order])

        moves = self.reasonable_turns(board, player_to_move)
        # self.logger.debug(f"MaxN: {player_to_move=}, {depth=}, moves to eval={len(moves)}")
        for i, move in enumerate(moves):
            win_prob = attack_succcess_probability(board.get_area(move[0]).get_dice(), board.get_area(move[1]).get_dice())
            loss_prob = 1 - win_prob

            # board_copy = deepcopy(board)

            scores = map(sum, zip(
                map(lambda x : win_prob * x, self.maxN(self.simulate_successful_attack(board, move[0], move[1]), depth-1, self.nextPlayer(player_to_move, players_order), players_order, time_left)[1]),
                map(lambda x : loss_prob * x, self.maxN(self.simulate_unsuccessful_attack(board, move[0], move[1]), depth-1, self.nextPlayer(player_to_move, players_order), players_order, time_left)[1])
            ))
            scores = tuple(scores)

            if scores[players_order.index(player_to_move)] > optimal_scores[players_order.index(player_to_move)] or optimal_move == None:
                optimal_index = i
                optimal_move = move
                optimal_scores = scores

            #Consider at most four best attacks
            #Seems the search barelly ever choses the other ones
            if i == 3:
                break

        # if depth != 0:
        #     self.logger.debug(f"Chose move number {optimal_index} : ({optimal_move} | {optimal_scores})")
        return (optimal_move, optimal_scores)

    def simulate_successful_attack(self, board, source_name, target_name, in_place=False):
        """Modifies the board as if the attack was successful

        Args:
            board (Board): Board to simulate the attack on
            source_name (int): Name of the source of the attack
            target_name (int): Name of the target of the attack
        """
        if not in_place:
            board = deepcopy(board)

        # Finds the copy of the board areas on the given board based on their name
        source_copy = board.get_area(source_name)
        target_copy = board.get_area(target_name)

        # Simulates the attack
        target_copy.set_dice(source_copy.get_dice() - 1)
        target_copy.set_owner(source_copy.get_owner_name())
        source_copy.set_dice(1)
        return board

    def simulate_unsuccessful_attack(self, board, source_name, target_name, in_place=False):
        """Modifies the board as if the attack was unsuccessful

        Args:
            board (Board): Board to simulate the attack on
            source_name (int): Name of the source of the attack
            target_name (int): Name of the target of the attack
        """
        if not in_place:
            board = deepcopy(board)

        # Finds the copy of the board areas on the given board based on their name
        source_copy = board.get_area(source_name)
        target_copy = board.get_area(target_name)

        # Simulates the attack
        att_dice = source_copy.get_dice()
        if att_dice == 8:
            target_copy.set_dice(target_copy.get_dice()-2 if target_copy.get_dice()-2 > 0 else 1)
        elif att_dice >= 4:
            target_copy.set_dice(target_copy.get_dice()-1 if target_copy.get_dice()-1 > 0 else 1)
        #target_copy.set_owner(source_copy.get_owner_name())
        source_copy.set_dice(1)
        return board

    def nextPlayer(self, player_to_move, players_order):
        """Return the next player to move. This might be unnecesary, I'm not sure how exactly the player order is implemented"""
        return players_order[players_order.index(player_to_move)+1] if players_order.index(player_to_move)+1 < len(players_order) else players_order[0]

def parse_board(player_name, board):
    """Converts current board into a csv formatted line
    Args:
        player_name (int): Name of the player whose state to consider
        board (Board): Current board
    """

    """Ideas for factors to take into consideration
        a) ++ Number of player areas in the main territory
        b1) -- Number of border areas
        b2) -- Number of different enemy players bordering on border area
        b3) -- Number of different enemy areas bordering on border area
            ?c1) +- Number of border areas that have probability of holding until next turn less than x?
        c2) +- Worst strength difference between a border area and an enemy border area (+ is good, - is bad)
        d1) ++ Number of dice
        d2) ++ Number of dice on border areas
            ?d3) +- Ratio of dice on borders compared to total number of dice (related to d1 and d2))
            ?d4) -- Number of enemy dice bordering on border areas (might be related to c)
            ?d5) ++ Number/Ratio of borders with 7+ dice vs number of borders
    """
    
    # Factors d1, d2, d5
    d1 = 0
    d2 = 0
    #d5 = 0

    all_player_areas = board.get_player_areas(player_name)
    all_player_areas_names = []
    for area in all_player_areas:
        all_player_areas_names.append(area.get_name())
        d1 += area.get_dice()
    border = board.get_player_border(player_name)
    border_names = []
    for area in border:
        border_names.append(area.get_name())
        dice = area.get_dice()
        d2 += dice
        #if dice > 6:
        #    d5 += 1

    # Factors a, b1
    a = 0
    main_region = None
    for region in board.get_players_regions(player_name):
        if len(region) > a:
            a = len(region)
            main_region = region
    b1 = len(border) if len(border) > 0 else 1

    # Use ratio in d5 instead of absolute number?
    #d5 = round((float)(d5 / b1), 3)

    # Factors b2, b3, c1, d3, TODO c2? d4? 
    b2 = 0
    b3 = 0
    c2 = 8
    bordering_players = []
    bordering_enemy_areas = []
    for area in border:
        for neighbour in area.get_adjacent_areas_names():
            if neighbour not in all_player_areas_names:
                # This line is a bit expensive, remove along with b2 if b2 does not have a big impact on winrate
                bordering_players.append(board.get_area(neighbour).get_owner_name())
                bordering_enemy_areas.append(neighbour)
                dice_dif = area.get_dice() - board.get_area(neighbour).get_dice()
                if dice_dif < c2:
                    c2 = dice_dif
    b2 = len(set(bordering_players)) # Removes duplicates and calculates total number of players bordering
    b3 = len(set(bordering_enemy_areas))
    
    """c1 = 0
    for area in border:
        if aiutils.probability_of_holding_area(board, area.get_name(), area.get_dice(), player_name) < 0.3:
            c1 += 1"""

    #d3 = (float)(d2/d1)
    
    features = a, b1, b2, b3, c2, d1, d2
    return features
