import matplotlib.pyplot as plt
import torch
from TicTacToe import TicTacToe
from ResNet import ResNet
from engine import engine

tictactoe = TicTacToe()
model = ResNet(tictactoe, 4, 64)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

args = {
    'C': 2,
    'num_searches': 60,
    'num_iterations': 3,
    'num_selfPlay_iterations': 500,
    'num_epochs': 10,
    'batch_size': 64
}

mctn = engine(model, optimizer, tictactoe, args)
mctn.learn()