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
    def __init__(self, x: torch.Tensor):
        raise NotImplementedError

    def get_dim(self):
        raise NotImplementedError

    def set_parameters(self, x: torch.Tensor):
        raise NotImplementedError

    def move(self, last_opponent_action):
        raise NotImplementedError
