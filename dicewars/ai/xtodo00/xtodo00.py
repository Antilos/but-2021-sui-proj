import dicewars
import random
import logging, time
from copy import deepcopy
from itertools import starmap
from typing import Tuple
from statistics import mean

import dicewars.ai.utils as aiutils

# TODO remove later
from dicewars.client.game.board import Board
from dicewars.client.game.area import Area

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand, TransferCommand

from nnmodel import NeuralNetwork
import torch

# For logging
import os.path


class AI:
    def __init__(self, player_name, board, players_order, max_transfers):
        self.player_name = player_name
        self.logger = logging.getLogger('AI')
        self.max_transfers = max_transfers
        self.players_order = players_order
        # Attack strategy flag
        # simple - choose a move with the highest probability that also improves board presence (blame Craszh)
        # simple+ - same as simple but simulates one best attack for every other opponent (blame Craszh and complain how slow it is)
        # maxN - use maxN to choose the best move (blame Antilos)
        self.strategy = 'maxN'
        self.use_nn = True # Use NN to evaluate board
        self.attack_times = []
        self.slowest_attack_time = 0
        self.attacks_this_turn = 0

        #NN
        inputWidth = 5
        input_model_filename = 'dicewars/ai/xtodo00/cpu_model_epochs_30_1.pt'
        self.model = NeuralNetwork(inputWidth)
        self.model.load_state_dict(torch.load(input_model_filename))

        # Logging
        self.create_logs = True
        self.csv_lines = []
        self.log_file_name = "xtodo00.csv"

        if self.strategy != "maxN":
            # Reorders the list so that this AI is the first item
            while self.players_order[0] != player_name:
                player = self.players_order[0]
                self.players_order.append(player)
                del self.players_order[0]

    def ai_turn(self, board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):
        """AI turn:
        TODO: Create heuristic for evaluating current board state (board score) based on different factors
        """

        if self.create_logs:
            features = parse_current_state(self.player_name, board)
            self.csv_lines.append(f'{",".join(map(str, features))}\n')
            pass
        
        # Transfers dice towards borders whenever transfers are available
        if nb_transfers_this_turn < self.max_transfers:
            transfer = self.find_best_transfer(board, self.player_name, nb_turns_this_game)
            if transfer is not None:
                return TransferCommand(transfer[0], transfer[1])
        current_board_score = self.evaluate_board(board, self.player_name)
        #choose strategy
        if self.strategy == 'simple' or self.strategy == 'simple+':
            # Gets the list of all possible attack moves
            attacks = list(aiutils.possible_attacks(board, self.player_name))

            # If not empty
            if attacks:
                # List of possible attacks with probability of success > 0.470
                # Evaluated attack == (source_area_name, target_area_name, probability * board_score_diff)
                evaluated_attacks = []

                # Evaluates and appends every probable attack
                for attack in attacks:
                    prob = aiutils.attack_succcess_probability(attack[0].get_dice(), attack[1].get_dice())
                    if prob < 0.470:
                        continue

                    # TODO If not enough time, skips evaluation and calculates differently?
                    
                    # Creates a copy of the board and simulates the attack as well as turns of all other opponents
                    board_copy = deepcopy(board)
                    self.simulate_successful_attack(board_copy, attack[0].get_name(), attack[1].get_name())
                    if self.strategy == "simple+":
                        for opponent in self.players_order[1:]:
                            self.simulate_opponent_turn(board_copy, opponent)

                    # Calculates the board score difference before the attack and after the attack + all players' turns
                    board_score_diff = self.evaluate_board(board_copy, self.player_name) - current_board_score
                    evaluated_attacks.append((attack[0].get_name(), attack[1].get_name(), board_score_diff * prob)) #TODO weight of probability?

                # No probable attacks available
                if len(evaluated_attacks) == 0:
                    return EndTurnCommand()

                # Sorts attacks based on their total score and executes the first attack in the list
                evaluated_attacks.sort(key = lambda x: x[2], reverse = True)

                # If not even the best attack is positive, ends turn
                if evaluated_attacks[0][2] <= 0:
                    # If no attacks or all moves have been done this round, does the best possible negative attack in order not to deadlock
                    if self.attacks_this_turn == 0 and nb_transfers_this_turn < self.max_transfers:
                        self.attacks_this_turn += 1
                        return BattleCommand(evaluated_attacks[0][0], evaluated_attacks[0][1])
                    self.attacks_this_turn = 0
                    return EndTurnCommand()

                self.attacks_this_turn += 1
                return BattleCommand(evaluated_attacks[0][0], evaluated_attacks[0][1])
            else:
                self.logger.debug("No more possible turns.")
                self.attacks_this_turn = 0
                return EndTurnCommand()

        elif self.strategy == 'maxN':
            # Transfers dice towards borders whenever transfers are available
            if nb_transfers_this_turn < self.max_transfers:
                transfer = self.find_best_transfer(board, self.player_name, nb_turns_this_game)
                if transfer is not None:
                    return TransferCommand(transfer[0], transfer[1])

            a_time = time.time()
            attack, scores = self.maxN(board, 4, self.player_name, self.players_order)
            if attack:
                board_copy = deepcopy(board)
                self.simulate_successful_attack(board_copy, attack[0].get_name(), attack[1].get_name())
                if self.evaluate_board(board_copy, self.player_name) - current_board_score > 0:
                    self.logger.debug("WoW, making maxN attack!")
                    a_time = time.time() - a_time
                    self.slowest_attack_time = a_time if a_time > self.slowest_attack_time else self.slowest_attack_time
                    self.attack_times.append(a_time)
                    self.logger.debug(f"Average attack time so far: {mean(self.attack_times)}")
                    self.logger.debug(f"Slowest attack time so far: {self.slowest_attack_time}")
                    return BattleCommand(attack[0].get_name(), attack[1].get_name())
                else:
                    return EndTurnCommand()
            else:
                #What? but it did happen once
                self.logger.debug('maxN didn\'t find a valid attack, ending turn.')
                return EndTurnCommand()
        else:
            #Just fuck me up
            return EndTurnCommand()


    def evaluate_board(self, board, player_name):
        """Evaluates the board TODO
        TODO optimize once the final heuristic has been decided
        Args:
            board (Board): Board to evaluate
            player_name (int): Name of the player whose board to evaluate
        """

        """Ideas for factors to take into consideration - not sure how complex it should be considering the time limit
        -- probably better to consider all of these only for the largest united territory
        -- split territories seem like a really big disadvantage with the way the game works tbh
        -- might add a big penalty for having split territory so that it always tries to unite them?
        a) ++ Number of player areas in the main territory
        b1) -- Number of border areas
        b2) -- Number of different enemy players bordering on border area / average or total for all border areas
        b3) -- Number of different enemy areas bordering on border area / average or total for all border areas
        c1) +- Number of border areas that have probability of holding until next turn less than x?
        c2) +- Strength difference between border areas and enemy border areas (might be related to c1 but this can also be used offensively)
        d1) ++ Number of dice
        d2) ++ Number of dice on border areas
        d3) +- Ratio of dice on borders compared to total number of dice (related to d1 and d2))
        d4) -- Number of enemy dice bordering on border areas (might be related to c)
        e) ++ Number of strong(ly positioned) border areas. Combining b123c12d24
        f) -- Number of weak border areas
        g) -- Number of weak border areas with weak areas behind them - very weak points
        z) +- ML model (NN?) trained to evaluate board based on these heuristics (as features?)
        """

        if self.use_nn:
            features = parse_current_state(player_name, board)
            logits = self.model(torch.FloatTensor([features]))
            win_prob = torch.nn.Softmax(dim=1)(logits)[0][1]
            return win_prob
        else:
            all_player_areas = board.get_player_areas(player_name)
            all_player_areas_names = []
            for area in all_player_areas:
                all_player_areas_names.append(area.get_name())
            border = board.get_player_border(player_name)
            border_names = []
            for area in border:
                border_names.append(area.get_name())

            # TODO Rename these as needed, for now they correspond to the docstring above
            a = 0
            main_region = None
            for region in board.get_players_regions(player_name):
                if len(region) > a:
                    a = len(region)
                    main_region = region
            b1 = len(border)

            # maxN did not seem to work well with the new heuristic, leaving the old one for now
            if self.strategy == "maxN":
                return a - 0.3 * b1

            # Evaluates every border area to decide whether it's a strong or weak border
            e = 0
            f = 0
            g = 0
            for area in border:
                # If area is not in the main region => not strong
                if area.get_name() not in main_region:
                    f += 1
                    continue
                # Area score = 
                # + number of dice
                # + number of dice in friendly surrounding non-border areas * 0.2
                # - number of enemy areas - 1
                # - number of enemy players surrounding area - 1
                # - number of dice of the strongest enemy area
                current_dice = area.get_dice()
                friendly_dice_behind = 0
                weakest_area_behind = 8
                enemy_areas = 0
                enemy_players = []
                strongest_enemy = 1

                # Checks neighbouring areas
                for neighbour_name in area.get_adjacent_areas_names():
                    neighbour = board.get_area(neighbour_name)

                    if neighbour_name in all_player_areas_names:
                        if neighbour_name not in border_names:
                            # Friendly non-border area
                            neighbour_dice = neighbour.get_dice()
                            friendly_dice_behind += neighbour_dice
                            if neighbour_dice < weakest_area_behind:
                                weakest_area_behind = neighbour_dice
                    else:
                        # Enemy border area
                        enemy_areas += 1
                        enemy_players.append(neighbour.get_owner_name())
                        enemy_dice = neighbour.get_dice()
                        if enemy_dice > strongest_enemy:
                            strongest_enemy = enemy_dice

                area_score = current_dice + 0.2 * friendly_dice_behind - 0.5 * (enemy_areas - 1) - (len(set(enemy_players)) - 1) - strongest_enemy
                if area_score < 0:
                    # Weak point
                    if weakest_area_behind < current_dice:
                        # Weaker points behind -- very weak point
                        g += 1
                    else:
                        f += 1
                else:
                    e += 1

            # Temporary solution, score = number of areas owned in main territory - (0.3 * length of border)
            # score = a - 0.3 * b1      ---- USE THIS IF ANYTHING BREAKS, THIS ONE WORKED ALWAYS
            """areas_score = 1.4 * a
            strong_score = 0.8 * e
            weak_score = 0 - 0.5 * f
            very_weak_score = 0 - g
            border_score = 0 - 0.5 * b1
            print("STATS:")
            print(areas_score, strong_score, weak_score, very_weak_score, border_score)"""
            # score = (1.4 * number of areas in main territory) - (0.5 * length of border) + (0.8 * strong borders) - (0.5 * weak borders) - very weak borders
            score = 1.4 * a - 0.5 * b1 + 0.8 * e  - 0.5 * f - g
            # Win condition
            if e == 0 and f == 0 and g == 0:
                score += 50
            # No weak points exist
            #if f == 0 and g == 0:
            #    score += 4
            #    print("+4")
            # Very weak points exist
            if g > 0:
                score -= 1
            return score

            """ Unused factors
            b2 = 0
            bordering_players = []
            b3 = 0
            for area in border:
                for neighbour in area.get_adjacent_areas_names():
                    if neighbour not in all_player_areas_names:
                        # This line is a bit expensive, remove along with b2 if b2 does not have a big impact on winrate
                        bordering_players.append(board.get_area(neighbour).get_owner_name())
                        b3 += 1
            # b3 = (float)(b3 / len(border)) # -- average instead of total
            b2 = len(set(bordering_players)) # Removes duplicates and calculates total number of players bordering
            
            c1 = 0
            for area in border:
                if aiutils.probability_of_holding_area(board, area.get_name(), area.get_dice(), player_name) < 0.3:
                    c1 += 1
            score = 1.4 * a - 0.4 * b1 - 4 * b2 - 0.8 * c1
            """  
            
    def evaluate_board_for_each_player(self, board:Board, players_order) -> Tuple:
        """Evaluates the board TODO PRIORITY 1

        Args:
            board (Board): Board to evaluate
            player_name (int): Order in which players take turns
        """

        # Temporary solution, score = number of areas owned - (0.3 * length of border)
        return tuple([self.evaluate_board(board, player) for player in players_order])
    
    def simulate_successful_attack(self, board, source_name, target_name):
        """Modifies the board as if the attack was successful

        Args:
            board (Board): Board to simulate the attack on
            source_name (int): Name of the source of the attack
            target_name (int): Name of the target of the attack
        """
        # Finds the copy of the board areas on the given board based on their name
        source_copy = board.get_area(source_name)
        target_copy = board.get_area(target_name)

        # Simulates the attack
        target_copy.set_dice(source_copy.get_dice() - 1)
        target_copy.set_owner(source_copy.get_owner_name())
        source_copy.set_dice(1)
        return board

    def simulate_unsuccessful_attack(self, board, source_name, target_name):
        """Modifies the board as if the attack was unsuccessful

        Args:
            board (Board): Board to simulate the attack on
            source_name (int): Name of the source of the attack
            target_name (int): Name of the target of the attack
        """
        # Finds the copy of the board areas on the given board based on their name
        source_copy = board.get_area(source_name)
        target_copy = board.get_area(target_name)

        # Simulates the attack
        att_dice = source_copy.get_dice()
        if att_dice == 8:
            target_copy.set_dice(target_copy.set_dice()-2)
        elif att_dice >= 4:
            target_copy.set_dice(target_copy.set_dice()-1)
        #target_copy.set_owner(source_copy.get_owner_name())
        source_copy.set_dice(1)
        return board

    def simulate_transfer(self, board, source_name, target_name):
        """Modifies the board to simulate a transfer (assumes source and target have the same owner)

        Args:
            board (Board): Board to simulate the transfer on
            source_name (int): Name of the source of the transfer
            target_name (int): Name of the target of the transfer
        """
        # Finds the copy of the board areas on the given board based on their name
        source_copy = board.get_area(source_name)
        target_copy = board.get_area(target_name)

        # Simulates the transfer
        dice_to_transfer = min(8 - target_copy.get_dice(), source_copy.get_dice() - 1)
        target_copy.set_dice(target_copy.get_dice() + dice_to_transfer)
        source_copy.set_dice(source_copy.get_dice() - dice_to_transfer)
        return board

    def find_best_transfer(self, board, player_name, turns):
        """TODO Finds the best possible transfer to strengthen borders (max depth 3)

        Args:
            board (Board): Board on which to find the transfer
            player_name (int): Name of the player transfering
            turns (int): Number of turns that have passed this game
        """
        # Gets the border (depth == 1)
        border = board.get_player_border(player_name)
        border_names = [area.name for area in border]
        all_player_areas_names = [area.name for area in board.get_player_areas(player_name)]

        # Gets names of areas closest to border (depth == 2)
        depth2_areas_names = []
        for area in border:
            for neighbour in area.get_adjacent_areas_names():
                if neighbour in all_player_areas_names and neighbour not in border_names:
                    depth2_areas_names.append(neighbour)
        # Removes duplicates and gets the actual areas based on the set of names
        depth2_areas_names = set(depth2_areas_names)
        depth2_areas = []
        for area_name in depth2_areas_names:
            depth2_areas.append(board.get_area(area_name))

        # For every depth2 area checks best possible transfers to depth1 areas (border)
        for area in depth2_areas:
            # TODO transfer only 2 or 3 or more dice per move at later stages of the game???
            # TODO do the most efficient transfers first???
            if area.get_dice() > 1:
                for neighbour in area.get_adjacent_areas_names():
                    if neighbour in border_names and board.get_area(neighbour).get_dice() < 8:
                        return (area.get_name(), neighbour)

        # Basically the same for depth 3 areas with only minor differences
        # Gets names of areas closest to depth2 areas (depth == 3)
        depth3_areas_names = []
        for area in depth2_areas:
            for neighbour in area.get_adjacent_areas_names():
                if neighbour not in border_names and neighbour not in depth2_areas_names:
                    depth3_areas_names.append(neighbour)
        # Removes duplicates and gets the actual areas based on the set of names
        depth3_areas_names = set(depth3_areas_names)
        depth3_areas = []
        for area_name in depth3_areas_names:
            depth3_areas.append(board.get_area(area_name))

        # For every depth3 area checks best possible transfers to depth2 areas
        for area in depth3_areas:
            # TODO transfer only 2 or 3 or more dice per move at later stages of the game???
            # TODO do the most efficient transfers first???
            # TODO simulate transfers first and check heuristic to find the best possible transfer?
            if area.get_dice() > 1:
                for neighbour in area.get_adjacent_areas_names():
                    if neighbour in depth2_areas_names and board.get_area(neighbour).get_dice() < 8:
                        return (area.get_name(), neighbour)

        return None

    def find_best_transfer_heuristic(self, board, player_name, turns):
        """TODO Finds the best possible transfer according to heuristic
        TODO Remove if unused

        Args:
            board (Board): Board on which to find the transfer
            player_name (int): Name of the player transfering
            turns (int): Number of turns that have passed this game
        """
        # (area_name, area_name)
        possible_transfers = []
        # (area_name, area_name, score)
        evaluated_transfers = []

        # Gets the border (depth == 1)
        border = board.get_player_border(player_name)
        border_names = [area.name for area in border]
        all_player_areas_names = [area.name for area in board.get_player_areas(player_name)]

        # Gets names of areas closest to border (depth == 2)
        depth2_areas_names = []
        for area in border:
            for neighbour in area.get_adjacent_areas_names():
                if neighbour in all_player_areas_names and neighbour not in border_names:
                    depth2_areas_names.append(neighbour)
        # Removes duplicates and gets the actual areas based on the set of names
        depth2_areas_names = set(depth2_areas_names)
        depth2_areas = []
        for area_name in depth2_areas_names:
            depth2_areas.append(board.get_area(area_name))

        # For every depth2 area checks best possible transfers to depth1 areas (border)
        for area in depth2_areas:
            # TODO transfer only 2 or 3 or more dice per move at later stages of the game???
            # TODO do the most efficient transfers first???
            if area.get_dice() > 1:
                for neighbour in area.get_adjacent_areas_names():
                    if neighbour in border_names and board.get_area(neighbour).get_dice() < 8:
                        possible_transfers.append((area.get_name(), neighbour))

        # Basically the same for depth 3 areas with only minor differences
        # Gets names of areas closest to depth2 areas (depth == 3)
        depth3_areas_names = []
        for area in depth2_areas:
            for neighbour in area.get_adjacent_areas_names():
                if neighbour not in border_names and neighbour not in depth2_areas_names:
                    depth3_areas_names.append(neighbour)
        # Removes duplicates and gets the actual areas based on the set of names
        depth3_areas_names = set(depth3_areas_names)
        depth3_areas = []
        for area_name in depth3_areas_names:
            depth3_areas.append(board.get_area(area_name))

        # For every depth3 area checks best possible transfers to depth2 areas
        for area in depth3_areas:
            # TODO transfer only 2 or 3 or more dice per move at later stages of the game???
            # TODO do the most efficient transfers first???
            # TODO simulate transfers first and check heuristic to find the best possible transfer?
            if area.get_dice() > 1:
                for neighbour in area.get_adjacent_areas_names():
                    if neighbour in depth2_areas_names and board.get_area(neighbour).get_dice() < 8:
                        possible_transfers.append((area.get_name(), neighbour))

        current_board_score = self.evaluate_board(board, self.player_name)

        # Evaluates transfers
        for transfer in possible_transfers:
            board_copy = deepcopy(board)
            self.simulate_transfer(board_copy, transfer[0], transfer[1])
            board_score_diff = self.evaluate_board(board_copy, self.player_name) - current_board_score
            evaluated_transfers.append((transfer[0], transfer[1], board_score_diff))

        if len(possible_transfers) == 0:
            return None

        evaluated_transfers.sort(key = lambda x: x[2], reverse = True)

        if evaluated_transfers[0][2] <= 0:
            return None

        return (evaluated_transfers[0][0], evaluated_transfers[0][1])

    def simulate_opponent_turn(self, board, player_name):
        """Simulates the turn of a given opponent
        For now opponents only do one best attack that is assumed to be successful and end turn
        TODO - remove if unused

        Args:
            board (Board): Board to simulate on
            player_name (int): Name of the player whose turn to simulate
        """
        # Gets the list of all possible attack moves
        attacks = list(aiutils.possible_attacks(board, player_name))
        current_board_score = self.evaluate_board(board, self.player_name)

        # If not empty
        if attacks:
            # List of possible attacks with probability of success > 0.470
            # Evaluated attack == (source_area_name, target_area_name, probability * board_score_diff)
            evaluated_attacks = []

            # Evaluates and appends every probable attack
            for attack in attacks:
                prob = aiutils.attack_succcess_probability(attack[0].get_dice(), attack[1].get_dice())
                if prob < 0.470:
                    continue

                # Creates a copy of the board and simulates the attack to calculate difference in board score
                board_copy = deepcopy(board)
                self.simulate_successful_attack(board_copy, attack[0].get_name(), attack[1].get_name())
                
                # Calculates the board score difference before the attack and after the attack + all players' turns
                board_score_diff = self.evaluate_board(board_copy, self.player_name) - current_board_score
                evaluated_attacks.append((attack[0].get_name(), attack[1].get_name(), board_score_diff * prob)) #TODO weight of probability?

            # No optimal attacks to simulate
            if len(evaluated_attacks) == 0:
                return

            # Sorts attacks based on their total score and executes the first attack in the list
            evaluated_attacks.sort(key = lambda x: x[2], reverse = True)

            # If not even the best attack is positive, ends turn
            if evaluated_attacks[0][2] <= 0:
                return

            # Simulates the best attack
            self.simulate_successful_attack(board, evaluated_attacks[0][0], evaluated_attacks[0][1])
            return
        else:
            # No move to simulate
            return

    def reasonable_attacks(self, board, player_name):
        """Return only moves that are reasonable
        
        Filters out moves that are not worth considering
            - Cuts moves with fewer dice than defender
        """
        f = lambda m : m[0].get_dice() >= m[1].get_dice() and aiutils.attack_succcess_probability(m[0].get_dice(), m[1].get_dice()) > 0.5 #Only consider moves with more dice

        return (move for move in aiutils.possible_attacks(board, player_name) if \
            f(move))

    def log_game_end(self, winner_name):
        """Appends winner to all collected csv lines and saves them to a file

        Args:
            winner_name (int): Name of the winner
        """
        if self.create_logs:
            save_csv_logs(self.csv_lines, winner_name == self.player_name, self.log_file_name)
        return


    def maxN(self, board:Board, depth:int, player_to_move, players_order) -> Tuple[Tuple[Area, Area], Tuple]:
        """Searches the stochastic game tree
        
        Evaluates all moves for each player up to a certain depth. The random event is loss X win.

        This can somehow be pruned using alphabeta, provided the scores are bounded
        See https://www.cc.gatech.edu/%7Ethad/6601-gradAI-fall2015/Korf_Multi-player-Alpha-beta-Pruning.pdf

        Ought to be optimized somehow, for example not considering moves agains bigger field, unclear on whether it should be part of this function

        Args:
            board (Board): Board to simulate the attack on
            depth (int): Maximum depth to search to (consider making this a multiple of number of players to ensure all players are taken into account)
            player_to_move: Name of the player that is currently on the move
            players_order: Order in which players take turns

        Return:
            The current players best move and it's scores. Should maybe also return the entire path?
        """

        if depth == 0:
            result = (None, self.evaluate_board_for_each_player(board, players_order))
            return result
        
        optimal_move = None
        optimal_scores = tuple([float('-inf') for player in players_order])
        for move in self.reasonable_attacks(board, player_to_move):
        # for move in aiutils.possible_attacks(board, player_to_move):
            winProb = aiutils.attack_succcess_probability(move[0].get_dice(), move[1].get_dice())
            lossProb = 1 - winProb

            board_copy = deepcopy(board) #!!!will run out of memory?!!! also probably should be part of the simulation function
            #Maybe just pass it by reference dumbass?

            # scores of the move averaged over win and loss
            scores = \
                map(sum, \
                    zip(\
                        map(lambda x : winProb * x,\
                            self.maxN(self.simulate_successful_attack(board_copy, move[0].get_name(), move[1].get_name()), depth-1, nextPlayer(player_to_move, players_order), players_order)[1]\
                        ),\
                        map(lambda x : lossProb * x,\
                            self.maxN(self.simulate_unsuccessful_attack(board_copy, move[0].get_name(), move[1].get_name()), depth-1, nextPlayer(player_to_move, players_order), players_order)[1]\
                        )\
                    )\
                )
            scores = tuple(scores)

            #only consider at most two best attacks
            # if i 

            # Does the move improve the current player's position?
            if scores[players_order.index(player_to_move)] > optimal_scores[players_order.index(player_to_move)] or optimal_move == None:
                optimal_scores = scores
                optimal_move = move

        return (optimal_move, optimal_scores)

def nextPlayer(player_to_move, players_order):
    """Return the next player to move. This might be unnecesary, I'm not sure how exactly the player order is implemented"""
    return players_order[players_order.index(player_to_move)+1] if players_order.index(player_to_move)+1 < len(players_order) else players_order[0]


def parse_current_state(player_name, board):
    """Converts current board into a csv formatted line

    Args:
        player_name (int): Name of the player whose state to consider
        board (Board): Current board
    """

    """Ideas for factors to take into consideration
        a) ++ Number of player areas in the main territory
        b1) -- Number of border areas
            ?b2) -- Number of different enemy players bordering on border area / average or total for all border areas
            ?b3) -- Number of different enemy areas bordering on border area / average or total for all border areas
            ?c1) +- Number of border areas that have probability of holding until next turn less than x?
            ?c2) +- Strength difference between border areas and enemy border areas (might be related to c1 but this can also be used offensively)
        d1) ++ Number of dice
        d2) ++ Number of dice on border areas
            ?d3) +- Ratio of dice on borders compared to total number of dice (related to d1 and d2))
            ?d4) -- Number of enemy dice bordering on border areas (might be related to c)
        d5) ++ Number/Ratio of borders with 7+ dice vs number of borders
    """
    
    # Factors d1, d2, d5
    d1 = 0
    d2 = 0
    d5 = 0

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
        if dice > 6:
            d5 += 1

    # Factors a, b1
    a = 0
    main_region = None
    for region in board.get_players_regions(player_name):
        if len(region) > a:
            a = len(region)
            main_region = region
    b1 = len(border) if len(border) > 0 else 1

    # Use ratio in d5 instead of absolute number?
    d5 = round((float)(d5 / b1), 3)

    """ Unused factors b2, b3, c1, d3, TODO c2? d4? 
    b2 = 0
    bordering_players = []
    b3 = 0
    for area in border:
        for neighbour in area.get_adjacent_areas_names():
            if neighbour not in all_player_areas_names:
                # This line is a bit expensive, remove along with b2 if b2 does not have a big impact on winrate
                bordering_players.append(board.get_area(neighbour).get_owner_name())
                b3 += 1
    # b3 = (float)(b3 / len(border)) # -- average instead of total
    b2 = len(set(bordering_players)) # Removes duplicates and calculates total number of players bordering
    
    c1 = 0
    for area in border:
        if aiutils.probability_of_holding_area(board, area.get_name(), area.get_dice(), player_name) < 0.3:
            c1 += 1

    d3 = (float)(d2/d1)
    """
    
    # Currently uses: a b1 d1 d2 d5
    #csv_line = a + "," + b1 + "," + d1 + "," + d2 + "," + d5 + "\n"
    # csv_line = f'{a},{b1},{d1},{d2},{d5}\n'
    features = a, b1, d1, d2, d5
    return features

def save_csv_logs(csv_lines, winner, file_name):
    """Appends win/loss to every csv line and saves to a file

    Args:
        csv_lines (string[]): Csv lines logging individual game states
        winner (bool): True if win, False if loss
        file_name (string): Name of the output file
    """
    new_file = True
    if os.path.isfile(file_name):
        new_file = False

    with open(file_name, "a") as f:
        if new_file:
            f.write("win,a,b1,d1,d2,d5\n")

        if winner:
            for line in csv_lines:
                f.write("1," + line)
        else:
            for line in csv_lines:
                f.write("0," + line)
    return