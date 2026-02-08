from abc import ABC, abstractmethod
import numpy as np
import time
import copy

def get_cor(id):
  """
    -1 se preta
    1 se branca

    parâmetros:
      id (int): id da peça
    retorna:
      cor (int): cor da peça
  """
  if id <= 16: return -1
  else: return 1

def linear_para_coord(state, id):
    """
      Transforma uma posição linear em coordenadas

      parâmetros:
        id (int): posição linear
      retorna:
        (x, y): posição em coordenadas
    """
    x = (state.pecas[id]-1) % 8
    y = (state.pecas[id]-1) // 8
    return (x, y)

def get_mat_add(state, id):
  """
    Retorna o valor de captura de uma peça
  """
  return state.valor[id]*(2*state.curr_player - 1)

def square_to_index(x, y):
    return y * 8 + x  # 0..63

def index_to_square(i):
    return i % 8, i // 8

def encode_action(fx, fy, tx, ty):
    return square_to_index(fx, fy) * 64 + square_to_index(tx, ty)

def decode_action(a):
    from_sq = a // 64
    to_sq = a % 64
    fx, fy = index_to_square(from_sq)
    tx, ty = index_to_square(to_sq)
    return fx, fy, tx, ty

def torre_casas_cobertas(x, y):
  """
    Gera todos as casas cobertas pela torre a partir de
      uma posição (x, y)

    par^amêtros:
      x (int): coluna
      y (int): linha
    retorna:
      mov (list): lista de movimentos
  """
  y_cima = range(y-1, -1, -1)
  cima = [y_cima, [x]]

  y_baixo = range(y+1, 8)
  baixo = [y_baixo, [x]]

  x_esq = range(x-1, -1, -1)
  esq = [[y], x_esq]

  x_dir = range(x+1, 8)
  dir_ = [[y], x_dir]

  return [cima, baixo, esq, dir_]

def torre_mov(state, movimentos):
  """
    Gera todos os possíveis movimentos para as torres do jogador atual.
    Consideramos a dama também pq ela anda que nem uma torre

    parâmetros:
      mov (ref lista): lista de movimentos, normalmente vazia e
        passada por referência
    Retorno:
      Lista: [
        Material a adicionar,
        from_x, from_y,
        to_x, to_y,
        id da peça, nova casa da peça]
  """
  # Torres e Dama
  torres_pretas = [1, 4, 8]
  torres_brancas = [25, 28, 32]

  torres = torres_pretas
  if state.curr_player == 1: torres = torres_brancas

  for i in torres:
    if state.pecas[i] == 0:
      continue # Já capturada

    i_cor = get_cor(i)
    x, y = linear_para_coord(state, i)

    for range_y, range_x in torre_casas_cobertas(x, y):
      # Passamos por todas as possíveis casas da torre
      # range_y = range(...) e range_x = [x] ou
      # range_y = [y]     e    range_x = range(...)

      for casa_y in range_y:
        for casa_x in range_x:
          casa_peca = state.tabuleiro[casa_y][casa_x]
          if casa_peca == 0:
            # Casa vazia - Pode avançar
            movimentos.append([0, x, y, casa_x, casa_y, i, 0])

          elif i_cor + (casa_peca > 16) == 1:
            # Capturamos a peça
            mat_add = get_mat_add(state, casa_peca)
            movimentos.append([mat_add, x, y, casa_x, casa_y, i, casa_peca])
            break

          else:
            break

        else:
          continue

        break


def bispo_casas_cobertas(x, y):
  y_cima = range(y-1, -1, -1)
  y_baixo = range(y+1, 8)
  y_lista = [y_cima, y_baixo]

  x_esq = range(x-1, -1, -1)
  x_dir = range(x+1, 8)
  x_lista = [x_esq, x_dir]

  return [y_lista, x_lista]


def bispo_mov(state, movimentos):
  # Consideramos a dama como bispo
  bispos_pretos = [3, 6, 4]
  bispos_brancos = [27, 30, 28]

  bispos = bispos_pretos
  if state.curr_player == 1: bispos = bispos_brancos

  for i in bispos:
    if state.pecas[i] == 0:
      continue

    i_cor = get_cor(i)
    x, y = linear_para_coord(state, i)

    casas_cobertas = bispo_casas_cobertas(x, y)

    for range_y in casas_cobertas[0]:
      for range_x in casas_cobertas[1]:
        eixo_minimo = min(len(range_y), len(range_x))
        passos = range(eixo_minimo)
        for p in passos:
          casa_y = range_y[p]
          casa_x = range_x[p]

          casa_peca = state.tabuleiro[casa_y][casa_x]

          if casa_peca == 0:
            movimentos.append([0, x, y, casa_x, casa_y, i, 0])

          elif i_cor + (casa_peca > 16) == 1:
            mat_add = get_mat_add(state, casa_peca)
            movimentos.append([mat_add, x, y, casa_x, casa_y, i, casa_peca])
            break

          else:
            break

def cavalo_mov(state, movimentos):
  cavalos_pretos = [2, 7]
  cavalos_brancos = [26, 31]

  cavalos = cavalos_pretos
  if state.curr_player == 1: cavalos = cavalos_brancos

  for i in cavalos:
    if state.pecas[i] == 0:
      continue

    i_cor = get_cor(i)
    x, y = linear_para_coord(state, i)

    for dist_y in [1, 2]:
      for dir_y in [-1, 1]:
        casa_y = y + dist_y * dir_y

        if casa_y < 0 or casa_y > 7:
          continue

        for dir_x in [-1, 1]:
          casa_x = x + (3 - dist_y) * dir_x

          if casa_x < 0 or casa_x > 7:
            continue

          casa_peca = state.tabuleiro[casa_y][casa_x]

          if casa_peca == 0 or ((i_cor) + (casa_peca > 16) == 1):
            mat_add = get_mat_add(state, casa_peca)
            movimentos.append([mat_add, x, y, casa_x, casa_y, i, casa_peca])

def peao_mov(state, movimentos):
  dir_ = -state.curr_player

  peoes_pretos = range(9, 17)
  peoes_brancos = range(17, 25)

  peoes = peoes_pretos
  if state.curr_player == 1: peoes = peoes_brancos

  for i in peoes:
    if state.pecas[i] == 0:
      continue

    i_cor = get_cor(i)
    x, y = linear_para_coord(state, i)

    if not (0 <= y+dir_ < 8):
      continue

    if x > 0 and state.tabuleiro[y+dir_][x-1] != 0 and (i_cor + (state.tabuleiro[y+dir_][x-1] > 16)) == 1:
      mat_add = get_mat_add(state, state.tabuleiro[y+dir_][x-1])
      movimentos.append([mat_add, x, y, x-1, y+dir_, i, state.tabuleiro[y+dir_][x-1]])

    if x < 7 and state.tabuleiro[y+dir_][x+1] != 0 and (i_cor + (state.tabuleiro[y+dir_][x+1] > 16)) == 1:
      mat_add = get_mat_add(state, state.tabuleiro[y+dir_][x+1])
      movimentos.append([mat_add, x, y, x+1, y+dir_, i, state.tabuleiro[y+dir_][x+1]])

    if state.tabuleiro[y+dir_][x] == 0:
      movimentos.append([0, x, y, x, y+dir_, i, 0])

      start_row = 6 if state.curr_player == 1 else 1

      if y == start_row and state.tabuleiro[y + dir_*2][x] == 0:
        movimentos.append([0, x, y, x, y+dir_*2, i, 0])

def rei_casas_cobertas(x, y):
  cima = max(0, y-1)
  baixo = min(8, y+2)
  y_lista = range(cima, baixo)

  esq = max(0, x-1)
  dir_ = min(8, x+2)
  x_lista = range(esq, dir_)


  return [y_lista, x_lista]

def rei_mov(state, movimentos):
  if state.curr_player == -1: rei = 5
  else: rei = 29

  i_cor = get_cor(rei)
  x, y = linear_para_coord(state, rei)

  casas_cobertas = rei_casas_cobertas(x, y)

  for casa_y in casas_cobertas[0]:
    for casa_x in casas_cobertas[1]:
      casa_peca = state.tabuleiro[casa_y][casa_x]
      if casa_peca == 0 or (i_cor + (casa_peca > 16) == 1):
        mat_add = get_mat_add(state, casa_peca)
        movimentos.append([mat_add, x, y, casa_x, casa_y, rei, casa_peca])

class Game(ABC):
  def __init__(self):
    pass

  @abstractmethod
  def get_next_state(self, state, action, player):
    pass

  @abstractmethod
  def get_valid_moves(self, state):
    pass

  @abstractmethod
  def get_value_and_terminated(self, value):
    pass

  @abstractmethod
  def get_opponent(self, player):
    pass

  @abstractmethod
  def get_opponent_value(self, value):
    pass

class Xadrez(Game):
  def __init__(self):
    self.initial_state()

    self.row_count = 8
    self.column_count = 8
    self.action_size = 4096
    self.input_channels = 13 # Added for ResNet model input

  def initial_state(self):
    """
      Inicia o Estado inicial do jogo
    """
    self.tabuleiro = [[i for i in range(1,9)],
      [i for i in range(9,17)],
      [0]*8,
      [0]*8,
      [0]*8,
      [0]*8,
      [i for i in range(17,25)],
      [i for i in range(25,33)]]

    self.pecas = [0, *[i for i in range(1,17)], *[i for i in range(49,65)]]
    self.posicoes = {tuple(self.pecas):1}

    self.valor = [0, 5, 3, 3, 9, float("inf"), 3, 3, 5,
              1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1,
              5, 3, 3, 9, float("inf"), 3, 3, 5]

    self.string = {0:"OO", 1:"BR", 2:"BN", 3:"BB", 4:"BQ", 5:"BK", 6:"BB", 7:"BN", 8:"BR",
                    9:"BP", 10:"BP", 11:"BP", 12:"BP", 13:"BP", 14:"BP", 15:"BP", 16:"BP",
                    17:"WP", 18:"WP", 19:"WP", 20:"WP", 21:"WP", 22:"WP", 23:"WP", 24:"WP",
                    25:"WR", 26:"WN", 27:"WB", 28:"WQ", 29:"WK", 30:"WB", 31:"WN", 32:"WR"}

    self.curr_player = 1
    self.material = 0

  def do_action(self, action, verbose):
    """
      Anda o próprio jogo em após a
        ação do player atual

      parâmetros:
        action: Ação a ser executada em texto. Ex: "e2e4"
        player: Jogador atual
      retorno:
        True se o jogo ainda continua
        False se o jogo acabou
    """
    valid_moves = self.get_valid_moves(self)

    move = valid_moves[action]
    from_x, from_y, to_x, to_y = move[1:5]

    if verbose:
      print(f"Moving {self.string[move[5]]} from ({from_x}, {from_y}) to ({to_x}, {to_y})")

    # Descobre as pecas
    peca = self.tabuleiro[from_y][from_x]
    capturada = self.tabuleiro[to_y][to_x]
    mat_add = -self.valor[capturada]

    # Atualiza o tabuleiro
    self.tabuleiro[from_y][from_x] = 0
    self.tabuleiro[to_y][to_x] = peca
    self.material += mat_add

    # Atualiza a lista de peças
    self.pecas[peca] = to_y * 8 + to_x + 1
    if capturada != 0:
        self.pecas[capturada] = 0

    # Checa empate
    if tuple(self.pecas) in self.posicoes:
        self.posicoes[tuple(self.pecas)] += 1
        if self.posicoes[tuple(self.pecas)] == 3:
            # Tie
            return False
    else:
        self.posicoes[tuple(self.pecas)] = 1

    # Verifica fim de jogo
    # WARNING: o valor de infinito é setado especificamente
    #   para o minmax, pode quebrar se usar outro
    if self.material == float("inf"):
        #print("Brancas vencem!")
        return False
    elif self.material == -float("inf"):
        #print("Pretas vencem!")
        return False

    return True

  def change_perspective(self, state, player):
    new = state.copy()
    new.curr_player = state.curr_player * player
    return new

  def get_initial_state(self):
    x = self.initial_state()
    new = copy.deepcopy(self)
    return new

  def get_valid_moves(self, state):
    """
      Retorna as ações possíveis para o estado atual.

      Cada movimento é uma array:
      []

      parâmetros:
        state: Estado atual
      retorno:
        Lista com todos os estados possíveis
    """
    moves = []
    torre_mov(state, moves)
    bispo_mov(state, moves)
    cavalo_mov(state, moves)
    peao_mov(state, moves)
    rei_mov(state, moves)
    return moves

  def get_valid_moves_mask(self, state):
    mask = np.zeros(self.action_size, dtype=np.float32)
    moves = self.get_valid_moves(state)  # sua lista (n,7)

    for m in moves:

        fx, fy = int(m[1]), int(m[2])
        tx, ty = int(m[3]), int(m[4])

        a = encode_action(fx, fy, tx, ty)
        mask[a] = 1.0

    return mask


  def get_next_state(self, state, action_id, player=1, verbose=False):
    new_state = copy.deepcopy(state)

    # action_id (0..4095) -> (fx,fy,tx,ty)
    fx, fy, tx, ty = decode_action(action_id)

    valid_moves = new_state.get_valid_moves(new_state)

    chosen_index = None
    for i, m in enumerate(valid_moves):
        if int(m[1]) == fx and int(m[2]) == fy and int(m[3]) == tx and int(m[4]) == ty:
            chosen_index = i
            break

    # Se não achou, é ação inválida para esse estado (bug ou mismatch)
    if chosen_index is None:
        return new_state  # fallback seguro (não altera)

    # Reusa sua lógica atual (que espera índice da lista)
    new_state.do_action(chosen_index, verbose)

    # troca jogador (mantém seu padrão +1/-1)
    new_state.curr_player = new_state.curr_player * -1
    return new_state


  def get_value_and_terminated(self, state, action):
    """
    Retorna:
      value: resultado do ponto de vista de quem jogou a action
      is_terminal: se o jogo terminou

    """
    if action == None:
      return 0, False

    # Pretas ganharam
    if state.material == -float("inf"):
        return -1, True

    # Brancas ganharam
    if state.material == float("inf"):
        return 1, True

    # Empate
    if tuple(state.pecas) in state.posicoes and state.posicoes[tuple(state.pecas)] >= 3:
        return 0, True

    # Jogo ainda não acabou
    return 0, False

  def get_opponent(self, player):
    return -player

  def get_opponent_value(self, value):
    return -value

  def copy(self):
    new = copy.deepcopy(self)
    return new

  def __str__(self):
    string = "/\n"
    for i, x in enumerate(self.tabuleiro):
        line = f"{i}  "
        for peca in x:
            line += self.string[peca] + " "
        line += "\n"
        string += line
    string += "\\  0  1  2  3  4  5  6  7   /"
    return string

  def get_encoded_state(self, state):
    """
    Baseado no seu state:
      - state.tabuleiro: 8x8 com ints (0 = vazio)
      - state.string: dict que converte int -> "BR","BP","WQ",...
      - state.curr_player: 1 ou -1 (no seu fluxo)

    Saída:
      np.array float32 shape (13, 8, 8)
        canais 0..5  : peças brancas  P N B R Q K
        canais 6..11 : peças pretas   P N B R Q K
        canal 12     : side-to-move (1 se curr_player==1, senão 0)
    """
    enc = np.zeros((13, 8, 8), dtype=np.float32)

    # ordem: P N B R Q K
    piece_to_idx = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5}

    for y in range(8):
        for x in range(8):
            pid = state.tabuleiro[y][x]
            if pid == 0:
                continue

            # No seu código, normalmente pid está em state.string.
            # Mas se você mudar IDs (ex: 49..64), isso mantém robustez.
            base_id = pid
            s = state.string.get(pid)
            if s is None:
                base_id = ((pid - 1) % 12) + 1
                s = state.string.get(base_id)
            if s is None:
                continue

            # peça desconhecida

            color = s[0]          # 'B' ou 'W'
            piece = s[1].upper()  # 'P','N','B','R','Q','K'

            if piece not in piece_to_idx:
                continue

            base = 0 if color == "W" else 6
            ch = base + piece_to_idx[piece]
            enc[ch, y, x] = 1.0

    # side-to-move: 1 se curr_player == 1 (como no seu código), senão 0
    enc[12, :, :] = 1.0 if state.curr_player == 1 else 0.0
    return enc
