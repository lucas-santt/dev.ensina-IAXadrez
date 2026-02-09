import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import trange
import random
from mcts import MCTS
from xadrez import Xadrez
from rede_neural import ResNet

xadrez = Xadrez()
model = ResNet(Xadrez(), num_resBlocks=4, num_hidden=64)

model.load_state_dict(torch.load("model_0.pt", map_location="cpu"))

args = {
    # MCTS
    "num_searches": 1000,
    "C": 1.5,
}

mcts = MCTS(xadrez, args, model)

state = xadrez.get_initial_state()

player = 1

while True:
    print(state)

    if player == 1:
        valid_moves = xadrez.get_valid_moves(state)

        print("valid moves: ")
        for i, m in enumerate(valid_moves):
          print(f"{i}: {state.string[m[5]]} ({m[1]}, {m[2]}) -> ({m[3]}, {m[4]})")


        action = int(input(f"Player {player}:"))

        if action+1 > len(valid_moves) or action < 0:
            print("action not valid")
            continue

        state.do_action(action, False)
        state.curr_player *= -1

    else:
      mcts.game = state
      neutral_state = state.copy()
      mcts_probs = mcts.search(state)
      action = np.argmax(mcts_probs)
      print(f"Choosed {action}")

      state = xadrez.get_next_state(state, action, -1, True)

    value, is_terminal = xadrez.get_value_and_terminated(state, action)

    if is_terminal:
        print(state)
        if value == -1:
            print(player, "won")
        elif value == 1:
            print("computer won")
        else:
            print("draw")
        break

    player = xadrez.get_opponent(player)
