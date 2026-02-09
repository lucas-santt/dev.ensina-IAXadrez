import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import trange
import random
import math

torch.manual_seed(0)

def square_to_index(x, y):
    """
        Converte as coordenadas (x,y) em um índice único de 0 a 63

        Parâmetros:
            x: Coordenada x (coluna)
            y: Coordenada y (linha)
        Retorna:
            Índice único correspondentee à posição (x,y)
    """
    return y * 8 + x  # 0..63

def index_to_square(i):
    """
        Converte um índice único de 0 a 63 em coordenadas (x,y)

        Parâmetros:
            i: Índice único (0..63)
        Retorna:
            x: Coordenada x (coluna)
            y: Coordenada y (linha)
    """
    return i % 8, i // 8

def encode_action(fx, fy, tx, ty):
    """
        Converte as coordenadas (fx, fy) e (tx, ty) em um índice único de 0 a 4095
        4095 são o número de ações possíveis em um tabuleiro de xadrez (64 quadrados de origeem * 64 quadrados de destino)

        Prâmetros:
            fx: Coordenada x de origem
            fy: Coordenada y de origem
            tx: Coordenada x de destino
            ty: Coordenada y de destino
        Retorna:
            Índice único correspondente à ação de mover de (fx, fy) para (tx, ty)
    """
    return square_to_index(fx, fy) * 64 + square_to_index(tx, ty)

def decode_action(a):
    """
        Converte um índice único de 0 a 4095 em coordenadas (fx, fy) e (tx, ty)

        Parâmetros:
            a: Índice único da ação (0..4095)
        Retorna:
            fx: Coordenada x de origem
            fy: Coordenada y de origem
            tx: Coordenada x de destino
            ty: Coordenada y de destino    
    """
    from_sq = a // 64
    to_sq = a % 64
    fx, fy = index_to_square(from_sq)
    tx, ty = index_to_square(to_sq)
    return fx, fy, tx, ty

class ResNet(nn.Module):
    """
        Classe que define a rede neural responsável por estimar
            a política e valor de um estado de jogo.

        A rede é composta por um bloco inicial de convolução,
            seguido por uma série de blocos residuais, e finalmente
            duas "cabeças" (saídas) separadas para a política e o valor.
    """
    def __init__(self, game, num_resBlocks, num_hidden):
        """
            Inicializa a rede neural seguindo a arquitetura da ResNet
        """
        super().__init__()

        self.startBlock = nn.Sequential(
            nn.Conv2d(game.input_channels, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )

        self.backBone = nn.ModuleList([ResBlock(num_hidden) for _ in range(num_resBlocks)])

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.column_count, game.action_size)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.row_count * game.column_count, 1),
            nn.Tanh()
        )

    def forward(self, x):
        """
            Define a passagem de dados pela rede,
                aplicando o bloco incial
                            bloco residuais
                            cabeça de política e valor

            Parâmetros:
                x: Tensor de entrada representando o estado do jogo
            Retorna:
                policy: Tensor de saída representando as probabilidades de cada ação
                value: Tensor de saída representando o valor do estado
        """
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value

class ResBlock(nn.Module):
    """
        Bloco residual, composto por duas camadas de convolução e uma skip connection
            onde seu resultado é somado à saída do bloco (antes de aplicar a função ReLU) 
    """
    def __init__(self, num_hidden):
        """
            Inicializa o bloco residual com as camadas de convolução e batch normalization
        """
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        """
            Aplica x nas camadas de convolução e batch normalization,
                somando o resultado à entrada original (skip connection) e aplicando Relu
        """
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)