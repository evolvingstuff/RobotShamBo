import random
from abc import ABC, abstractmethod
import math
import torch
import torch.nn.functional as F
from config import *


class Player(ABC):
    @abstractmethod
    def move(self, last_opponent_action: int) -> int:
        pass


class RandomPlayer(Player):
    """
    The classic Nash equilibrium strategy that cannot be exploited
    if we assume symmetrical rewards.
    """
    def move(self, last_opponent_action):
        return random.randint(0, 2)


class RockBiasedRandomPlayer(Player):
    """
    Similar to the RandomPlayer, but can learn to favor Rock more
    or less than the other possible actions.
    """
    def __init__(self):
        self.bias = 0

    def get_dim(self):
        return 1

    def set_parameters(self, x: torch.Tensor):
        self.bias = x.item()

    def move(self, last_opponent_action):
        def sig(x):
            return 1 / (1 + math.e**(-x))
        p = sig(self.bias)
        if random.random() < p:
            return ROCK
        else:
            return random.randint(1, 2)


class Rnn(Player):
    """
    A very simple one layer LSTM with a (configurable) argmax or softmax
    readout function to make the choice.
    """
    def __init__(self):
        self.x = None
        self.input_dim = 6
        self.hidden_dim = 25
        self.batch_size = 1
        self.rnn = torch.nn.LSTMCell(self.input_dim, self.hidden_dim)
        self.readout = torch.nn.Linear(self.hidden_dim, 3)
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


class RnnMeta(Player):
    """
    This extends the simpler Rnn model with the ability to specify "meta" levels,
    akin to what the Iocaine Powder strategy does.

    So, for example, level 1 of a "meta" shift would be to have picked rock, but then
    instead choose the action that beats rock, namely paper.
    """
    def __init__(self):
        self.x = None
        self.input_dim = 9
        self.hidden_dim = 25
        self.batch_size = 1
        self.rnn = torch.nn.LSTMCell(self.input_dim, self.hidden_dim)
        self.readout = torch.nn.Linear(self.hidden_dim, 3)
        self.meta_shift = torch.nn.Linear(self.hidden_dim, 3)
        self.h = torch.zeros(self.batch_size, self.hidden_dim)
        self.c = torch.zeros(self.batch_size, self.hidden_dim)
        self.my_last_action = torch.zeros(6)
        self.rnn.eval()
        self.readout.eval()

    def get_dim(self):
        p1 = sum(p.numel() for p in self.rnn.parameters())
        p2 = sum(p.numel() for p in self.readout.parameters())
        p3 = sum(p.numel() for p in self.meta_shift.parameters())
        return p1 + p2 + p3

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
        for param in self.meta_shift.parameters():
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
            output2 = self.meta_shift(self.h)
            if allow_model_rng_access:
                softmax_output = F.softmax(output)
                choice = torch.multinomial(softmax_output, 1).item()
                softmax_output2 = F.softmax(output2)
                meta_shift = torch.multinomial(softmax_output2, 1).item()
            else:
                choice = output.argmax().item()
                meta_shift = output2.argmax().item()
            self.my_last_action = torch.zeros(6)
            self.my_last_action[choice] = 1.0
            self.my_last_action[meta_shift + 3] = 1.0  # want to capture both our original choice and the meta shift
            assert 0 <= choice <= 2, f'invalid choice {choice}'
            assert 0 <= meta_shift <= 2, f'invalid choice {meta_shift}'
            meta_shifted_choice = (choice + meta_shift) % 3
            assert 0 <= meta_shifted_choice <= 2, f'invalid shifted_choice {meta_shifted_choice}'
            return meta_shifted_choice
