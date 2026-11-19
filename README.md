# AlphaZero RL

An implementation of AlphaZero-style reinforcement learning for **Tic-Tac-Toe** and **Connect Four**, maybe will add Chess later.

The project combines Monte Carlo Tree Search (MCTS), a residual neural network, Dirichlet exploration, and checkpointed training.


## Project Structure

- `AlphaMCTS.py` — AlphaZero MCTS implementation
- `mcts.py` — basic MCTS implementation
- `model.py` — residual policy/value network
- `TicTacToe.py` — Tic-Tac-Toe environment
- `ConnectFour.py` — ConnectFour environment
- `train.py` — sequential self-play training
- `trainParallel.py` — parallel self-play training
- `Alphamain.py` — human vs. MCTS/AlphaZero play
- `checkpoints/` — saved model and optimizer checkpoints


Create an isolated environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```


## Play a Game

```bash
python Alphamain.py
```

Set `game_name` in `main()` to one of:

- `TicTacToe`
- `ConnectFour`
- `TicTacToe_parallel`
- `ConnectFour_parallel`

For Tic-Tac-Toe, enter a column-free position from `0` to `8`. For Connect Four, enter a column from `0` to `6`.

Always instantiate the model with the same game and network dimensions used during training.
