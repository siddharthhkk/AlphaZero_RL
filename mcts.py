"""
1. Standard MCTS:
   Iteratively builds a search tree across four phases: Selection, Expansion, 
   Simulation (rollout), and Backpropagation. During Selection, child nodes 
   are chosen by maximizing the Upper Confidence Bound (UCB1):
   
       UCB1 = (w / n) + c * sqrt(ln(N) / n)

2. AlphaZero MCTS:
   Replaces random rollouts with a neural network f(s) = (p, v) that evaluates 
   leaf states directly:
     - p (policy): Prior probability distribution over valid actions, guiding exploration.
     - v (value): Scalar in [-1, 1] estimating the expected game outcome from state s.

   During Selection, nodes are chosen using the PUCT (Predictor + UCT) formula:
   
       PUCT = (w / n) + c * p * (sqrt(N) / (1 + n))

Variables:
   w : Total action value accumulated by the child node
   n : Visit count of the child node
   N : Visit count of the parent node
   c : Exploration hyperparameter
   p : Prior probability of selecting the child action (from policy network)
"""



# for every game we want to store the data to training set for the next iteration
# add (state, action(mcts distribution), reward) to the training set

"""
Training
    sample = s, π, z                        (take sample from training data)
    f(s) = (p, v)                           (get output from the neural network)
    loss = (z - v)^2 - π^T * log(p) + c     (minimize the loss function through backpropagation)

"""

import numpy as np


class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken

        self.children = []
        self.expandable_moves = game.get_valid_moves(state)

        self.visit_count = 0
        self.value_sum = 0

    def is_fully_expanded(self):
        return np.sum(self.expandable_moves) == 0 and len(self.children) > 0

    def select(self):
        best_child = None
        best_ucb1 = -(np.inf)
        for child in self.children:
            ucb1 = self.get_ucb(child)
            if ucb1 > best_ucb1:
                best_ucb1 = ucb1
                best_child = child
        return best_child

    def get_ucb(self, child): # Q(s, a) + c * sqrt(ln(n) / n)
        q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2   # opponent in bad pos so 1 - Q(s, a)
        return q_value + self.args['c'] * np.sqrt(np.log(self.visit_count) / child.visit_count)

    def expand(self):
        action = np.random.choice(np.where(self.expandable_moves == 1)[0])
        self.expandable_moves[action] = 0
        child_state = self.game.get_next_state(self.state.copy(), action, 1) # Assuming player 1 is the current player
        child_state = self.game.change_game_perspective(child_state, player=-1) 

        child = Node(self.game, self.args, child_state, self, action)
        self.children.append(child)
        return child

    def simulate(self):
        value, is_terminal = self.game.get_value_and_terminated(self.state, self.action_taken)
        value = self.game.get_opponent_value(value)

        if is_terminal:
            return value
        rollout_state = self.state.copy()
        rollout_player = 1
        while True:
            valid_moves = self.game.get_valid_moves(rollout_state)
            action = np.random.choice(np.where(valid_moves == 1)[0])
            rollout_state = self.game.get_next_state(rollout_state, action, rollout_player)
            value, is_terminal = self.game.get_value_and_terminated(rollout_state, action)
            if is_terminal:
                if rollout_player == -1:
                    value = self.game.get_opponent_value(value)
                return value
            rollout_player = self.game.get_opponent(rollout_player)

    def backprop(self, value):
        self.value_sum += value
        self.visit_count += 1
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backprop(value)


class MCTS:
    def __init__(self, game, args):
        self.game = game
        self.args = args


    def search(self, state):
        root = Node(self.game, self.args, state)
        for search in range(self.args['num_searches']):
            node = root
            # Selection
            while node.is_fully_expanded():
                node = node.select()
            # Expansion
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
            if not is_terminal:
                node = node.expand()
            # Simulation
                value = node.simulate()
            # Backpropagation
            node.backprop(value)

        # return visit counts for each action
        action_probabilities = np.zeros(self.game.action_size)
        for c in root.children:
            action_probabilities[c.action_taken] = c.visit_count
        action_probabilities /= np.sum(action_probabilities)
        return action_probabilities