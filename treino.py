import torch
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
from xadrez import Xadrez
from rede_neural import ResNet
from mcts import MCTS

class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

    def selfPlay(self):
        """
        Gera uma partida por self-play e retorna uma lista de tuplas:
          (encoded_state (13,8,8), pi (4096,), z (float))
        """
        memory = []
        self.game = self.game.get_initial_state()

        while True:
            # MCTS já trabalha com action_id (0..4095)
            pi = self.mcts.search(self.game)  # (action_size,)
            memory.append((self.game, pi, self.game.curr_player))

            # amostra uma ação seguindo pi
            action_id = int(np.random.choice(self.game.action_size, p=pi))

            # aplica a ação (seu get_next_state já decodifica action_id)
            self.game = self.game.get_next_state(self.game, action_id)

            value, is_terminal = self.game.get_value_and_terminated(self.game, action_id)
            if is_terminal:
                out = []
                for hist_state, hist_pi, hist_player in memory:
                    z = value if hist_player == 1 else self.game.get_opponent_value(value)
                    out.append((self.game.get_encoded_state(hist_state), hist_pi, z))
                    
                return out

    def train(self, memory):
        """
        Treina a rede usando:
          - policy loss: CE com soft targets (pi do MCTS)
          - value loss: MSE
        """
        random.shuffle(memory)

        bs = self.args['batch_size']
        for batchIdx in range(0, len(memory), bs):
            sample = memory[batchIdx:batchIdx + bs]
            states, policy_targets, value_targets = zip(*sample)

            states = torch.tensor(np.array(states), dtype=torch.float32)
            policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32)
            value_targets = torch.tensor(np.array(value_targets).reshape(-1, 1), dtype=torch.float32)

            out_policy_logits, out_value = self.model(states)

            # policy: soft targets (pi) -> -(pi * log_softmax).sum
            log_probs = F.log_softmax(out_policy_logits, dim=1)
            policy_loss = -(policy_targets * log_probs).sum(dim=1).mean()

            # value: MSE
            value_loss = F.mse_loss(out_value, value_targets)

            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []

            self.model.eval()
            for _ in trange(self.args['num_selfPlay_iterations'], desc=f"Self-play {iteration}"):
                memory += self.selfPlay()

            self.model.train()
            for _ in trange(self.args['num_epochs'], desc=f"Train {iteration}"):
                self.train(memory)

            torch.save(self.model.state_dict(), f"model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")

model = ResNet(Xadrez(), num_resBlocks=4, num_hidden=64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

xadrez = Xadrez()

args = {
    # MCTS
    "num_searches": 50,
    "C": 1.5,

    # treino
    "batch_size": 64,
    "num_iterations": 50,
    "num_selfPlay_iterations": 2,
    "num_epochs": 2,

    # opcionais (deixa assim por enquanto)
    "temperature": 1.0,
    "dirichlet_epsilon": 0.25,
    "dirichlet_alpha": 0.3,
}

az = AlphaZero(model, optimizer, xadrez, args)

az.learn()

#az.selfPlay()

print("Finalizado")
