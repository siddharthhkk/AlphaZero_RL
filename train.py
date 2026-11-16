import torch
import os
import numpy as np
import random
from tqdm import trange
from AlphaMCTS import AlphaMCTS
from TicTacToe import TicTacToe
from model import ResNet


class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = AlphaMCTS(game, args, model)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(base_dir, "checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)

    def selfPlay(self):
        memory = []
        player = 1
        state = self.game.get_initial_state()

        while True:
            neutral_state = self.game.change_game_perspective(state, player)
            action_probs = self.mcts.search(neutral_state)

            memory.append((neutral_state, action_probs, player))

            action = np.random.choice(self.game.action_size, p=action_probs)
            state = self.game.get_next_state(state, action, player)

            value, is_terminal = self.game.get_value_and_terminated(state, action)

            if is_terminal:
                returnMemory = []

                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs, hist_outcome
                    ))
                return returnMemory

            player = self.game.get_opponent(player)

    def train(self, memory):
        pass


    def learn(self):
        for iter in range(self.args['num_iterations']):
            memory = []

            self.model.eval()
            for selfPlay_iter in trange(self.args['num_selfPlay_iterations']):  #trange for progress bar
                memory += self.selfPlay()

            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)

            torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"model_{iter}.pt"))
            torch.save(self.optimizer.state_dict(), os.path.join(self.save_dir, f"optimizer_{iter}.pt"))



game = TicTacToe()

model = ResNet(game, 4, 64)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
args = {
    'c': 2,
    'num_searches': 60,
    'num_iterations': 3,
    'num_selfPlay_iterations': 10,
    'num_epochs': 4
}

alphaZero = AlphaZero(model, optimizer=opt, game=game, args=args)
alphaZero.learn()