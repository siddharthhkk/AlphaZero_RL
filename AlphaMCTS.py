import numpy as np
import torch

class AlphaNode:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = []
        # self.expandable_moves = game.get_valid_moves(state)    with policy we expand in all directions

        self.visit_count = visit_count
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.children) > 0 #np.sum(self.expandable_moves) == 0 and len(self.children) > 0

    def select(self):
        best_child = None
        best_ucb1 = -(np.inf)
        for child in self.children:
            ucb1 = self.get_ucb(child)
            if ucb1 > best_ucb1:
                best_ucb1 = ucb1
                best_child = child
        return best_child

    def get_ucb(self, child): # PUCT = (w / n) + c * p * (sqrt(N) / (1 + n))
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2   # opponent in bad pos so 1 - Q(s, a)
        return q_value + self.args['c'] * child.prior * (np.sqrt(self.visit_count) / (child.visit_count + 1)) 

    def expand(self, policy):
        #action = np.random.choice(np.where(self.expandable_moves == 1)[0])
        #self.expandable_moves[action] = 0
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.game.get_next_state(self.state.copy(), action, 1) # Assuming player 1 is the current player
                child_state = self.game.change_game_perspective(child_state, player=-1) 

                child = AlphaNode(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)
        return child

    def backprop(self, value):
        self.value_sum += value
        self.visit_count += 1
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backprop(value)





class AlphaMCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad() #as we just want to get the prediction
    def search(self, state):
        root = AlphaNode(self.game, self.args, state, visit_count=1)

        #adding noise
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
        
        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)
        
        for search in range(self.args['num_searches']):
            node = root
            # Selection
            while node.is_fully_expanded():
                node = node.select()
            # Expansion
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
            if not is_terminal:
                # for prediction from the trained model!
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)

                value = value.item()

                node.expand(policy)
            # Simulation
                #value = node.simulate() # no need in AlphaMCTS
            # Backpropagation
            node.backprop(value)

        # return visit counts for each action
        action_probabilities = np.zeros(self.game.action_size)
        for c in root.children:
            action_probabilities[c.action_taken] = c.visit_count
        action_probabilities /= np.sum(action_probabilities)
        return action_probabilities