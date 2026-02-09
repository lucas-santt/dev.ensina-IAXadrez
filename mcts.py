import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import trange
import random
import math

class Node:
  def __init__(self, state, args, parent=None, action_taken=None, prior=0):
    """
      Propriedades:
        game: Classe do jogo
        args: Parâmetros da árvore
        state: Estado do jogo
        parent: Pai
        action_taken: Ação tomada para chegar no estado atual

        children: Filhos
        expandable_moves: Possíveis ações que podem ser executadas
          a partir do estado

        visit_count: Número de vezes que o nó foi visitado (a partir das folhas)
        value_sum: Soma dos valores dos nós visitados
    """
    self.state = state
    self.args = args
    self.parent = parent
    self.action_taken = action_taken
    self.prior= prior

    self.children = []
    #self.expandable_moves = state.get_valid_moves(state) não precisa mais pois temos policy, ou seja, não é necessaria expandir para todas as possiveis jogadas

    self.visit_count = 0
    self.value_sum = 0

  def is_fully_expanded(self):
    """
      Retorna True se todas as ações possíveis já foram exploradas.
    """
    return len(self.children) > 0

  def select(self):
    """
      Escolhe o filho com maior valor UCB.
    """
    best_child = None
    best_ucb = -np.inf

    for child in self.children:
      ucb = self.get_ucb(child)
      if ucb > best_ucb:
        best_child = child
        best_ucb = ucb

    return best_child

  def get_ucb(self, child):
    """
      Retorna o valor UCB de um filho.

      UCB = U(s,a) = c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

      Q pode ser interpretado como uma métrica de WinRate = Q(s,a) = W(s,a) / N(s,a)

      Alteramos o Q-value para ^Q-value = 1 - Q-value pois o filho sempre
        irá estar representando um player oponente, então o valor é
        interpretado de formas diferentes para o player e para o pai

    """
    if child.visit_count == 0:
            q_value = 0
    else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
    return q_value + self.args['C'] * (math.sqrt(self.visit_count + 1) / (child.visit_count + 1)) * child.prior
  
  def expand(self, policy, game):
    """
      Expande o nó, criando um filho utilizando a política fornecida pelo modelo.

      A política é um vetor de probabilidades para cada ação possível.

      Parâmetros:
        policy: Vetor de probabilidades para cada ação possível.
        game: Classe do jogo, utilizada para obter o próximo estado a partir de uma ação.
    """
    for action, prob in enumerate(policy):
        if prob <= 0:
            continue

        child_state = self.state.copy()

        # action é o action_id (0..action_size-1)
        # use o método do game para aplicar a ação
        child_state = game.get_next_state(child_state, action)

        child = Node(
            state=child_state,
            args=self.args,
            parent=self,
            action_taken=action,
            prior=float(prob),
        )
        self.children.append(child)

# No MCTS do alphazero, a etapa de simulação é apagada(não utiliza ela, devido a policy e value)

  def backpropagate(self, value, game):
    """
      Realiza o backpropagation do valor do nó até a raiz, atualizando o valor
        ucb e o número de visitas de cada nó.
    """
    self.value_sum += value
    self.visit_count += 1

    value = game.get_opponent_value(value)
    if self.parent is not None:
        self.parent.backpropagate(value, game)

class MCTS:
    def __init__(self, game, args, model):
        """
          Parâmetros da classe MCTS
        
        game: Classe do jogo
        args: Parâmetros da árvore, com o número de iterações e o valor de C para o UCB
        model: Modelo utilizado para obter a política e o valor de um estado
        """
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state):
        """
          Realiza uma busca MCTS a partir do estado fornecido, retornando
            as probabilidades de cada ação a partir da raiz da árvore.

          Parâmetros:
            state: Estado do jogo a partir do qual a busca será realizada
          Retorna:
            action_probs: Vetor de probabilidades para cada ação possível a partir do estado fornecido
        """
        root = Node(state, self.args)

        valid_moves = state.get_valid_moves(state)

        for search in range(self.args['num_searches']):
            node = root

            # Selection
            while node.is_fully_expanded():
                node = node.select()

            # Aqui, a ação realizada é a ação realizada pelo pai do node, ou seja, pelo oponente
            # Isso vale também para o valor, por isso que temos que trocar o valor pelo get_opponent_value
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)

            if not is_terminal:
              policy, value = self.model(
                  torch.tensor(self.game.get_encoded_state(node.state)).unsqueeze(0)
              )
              policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
              mask = self.game.get_valid_moves_mask(node.state)   # (4096,)
              policy *= mask
              s = np.sum(policy)
              if s > 0:
                policy /= s
              else:
             # fallback (se por algum motivo policy zerar tudo)
                policy = mask / np.sum(mask)


              value = value.item()

              # Expansion
              node.expand(policy, self.game)

              # Simulation não precisa mais

               # BackPropagation
              node.backpropagate(value, self.game)
              
        action_probs = np.zeros(self.game.action_size, dtype=np.float32)              

        for child in root.children:
             action_probs[child.action_taken] = child.visit_count

             s = action_probs.sum()
        if s > 0:
             action_probs /= s
        else:
        # fallback: se MCTS não gerou filhos, devolve distribuição uniforme entre ações válidas
             mask = self.game.get_valid_moves_mask(state)
             action_probs = mask / mask.sum()

        return action_probs



        """
        # Assim se faz no jogo da velha, mas aqui não funciona
        # pois o action_size varia muito (é o equivalente ao len(root.children))
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
        """
