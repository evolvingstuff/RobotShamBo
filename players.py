import random
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from config import *


class Player(ABC):
    @abstractmethod
    def move(self, last_opponent_action: int) -> int:
        pass


class StaticOpponent(Player):
    def __init__(self):
        self.x = None
        self.round_num = 0

    def get_dim(self):
        return total_rounds * 3

    def set_parameters(self, x: torch.Tensor):
        self.x = x.reshape(-1, 3)

    def move(self, last_opponent_action):
        output = self.x[self.round_num]
        # softmax_output = F.softmax(output)
        # choice = torch.multinomial(softmax_output, 1)
        choice = output.argmax()  # deterministic
        assert 0 <= choice <= 2, f'invalid choice {choice}'
        self.round_num += 1
        return choice.item()


class RNN(Player):
    def __init__(self):
        self.x = None
        self.round_num = 0
        self.input_dim = 6
        self.hidden_dim = 10
        self.output_dim = 3
        self.batch_size = 1
        self.rnn = torch.nn.LSTMCell(self.input_dim, self.hidden_dim)
        self.readout = torch.nn.Linear(self.hidden_dim, self.output_dim)
        self.h = torch.zeros(self.batch_size, self.hidden_dim)
        self.c = torch.zeros(self.batch_size, self.hidden_dim)
        self.my_last_action = torch.zeros(3)
        self.rnn.eval()
        self.readout.eval()

    def get_dim(self):
        p1 = sum(p.numel() for p in self.rnn.parameters())
        p2 = sum(p.numel() for p in self.readout.parameters())
        return p1 + p2

    def set_parameters(self, x: torch.Tensor):
        pointer = 0
        for param in self.rnn.parameters():
            num_elements = param.numel()
            param_vector = x[pointer:pointer + num_elements]
            param.data = param_vector.view(param.shape)
            pointer += num_elements
        for param in self.readout.parameters():
            num_elements = param.numel()
            param_vector = x[pointer:pointer + num_elements]
            param.data = param_vector.view(param.shape)
            pointer += num_elements

    def move(self, last_opponent_action: int):
        with torch.no_grad():
            op = torch.zeros(3)
            if last_opponent_action is not None:
                op[last_opponent_action] = 1.0
            x = torch.hstack((op, self.my_last_action)).reshape((self.batch_size, -1))
            self.h, self.c = self.rnn(x, (self.h, self.c))
            output = self.readout(self.h)
            if allow_model_rng_access:
                softmax_output = F.softmax(output)
                choice = torch.multinomial(softmax_output, 1).item()
            else:
                choice = output.argmax().item()
            self.my_last_action = torch.zeros(3)
            self.my_last_action[choice] = 1.0
            assert 0 <= choice <= 2, f'invalid choice {choice}'
            return choice
