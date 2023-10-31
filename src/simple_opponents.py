import random
from config.config import ROCK, PAPER, SCISSORS


class ConstantOpponent():
    """
    Always play same move
    """
    def __init__(self, constant_choice):
        self.constant_choice = constant_choice

    def move(self, last_opponent_action):
        return self.constant_choice


class RandomStrategyChangeOpponent():
    """
    Switches strategy with probability p
    """
    def __init__(self, p=0.1):
        self.p = p
        self.choice = random.randint(0, 2)

    def move(self, last_opponent_action):
        if random.random() < self.p:
            self.choice = random.randint(0, 2)
        return self.choice


class RoundRobinOpponent():
    """
    Rock, paper, scissors, rock, paper, scissors, ...
    """
    def __init__(self):
        self.choice = random.randint(0, 2)
        self.update_rule = random.choice([-1, 1])

    def move(self, last_opponent_action):
        self.choice += self.update_rule  # be able to cycle both directions
        self.choice = self.choice % 3
        return self.choice


class ReverseRoundRobinDoubleTapOpponent():
    """
    Rock, rock, scissors, scissors, paper, paper, ...
    """
    def __init__(self):
        self.random_offset = random.randint(0, 5)
        self.pattern = [0, 0, 2, 2, 1, 1]
        self.round_num = 0

    def move(self, last_opponent_action):
        choice = self.pattern[(self.round_num + self.random_offset)% len(self.pattern)]
        self.round_num += 1
        return choice


class RandomOpponent():
    """
    The classic Nash equilibrium strategy that cannot be exploited
    if we assume symmetrical rewards.
    """
    def move(self, last_opponent_action):
        return random.randint(0, 2)


class PreviousLossOrDrawOpponent():
    """
    Play whichever option lost or got a draw last time (so click the red or gray square)
    """
    def __init__(self):
        self.last_move = None

    def move(self, last_opponent_move):
        if self.last_move is None:
            choice = random.randint(0, 2)
        else:
            if last_opponent_move is None:
                choice = random.randint(0, 2)
            elif last_opponent_move == ROCK:
                if self.last_move == ROCK:
                    choice = ROCK
                elif self.last_move == PAPER:
                    choice = ROCK
                elif self.last_move == SCISSORS:
                    choice = SCISSORS
            elif last_opponent_move == PAPER:
                if self.last_move == ROCK:
                    choice = ROCK
                elif self.last_move == PAPER:
                    choice = PAPER
                elif self.last_move == SCISSORS:
                    choice = PAPER
            elif last_opponent_move == SCISSORS:
                if self.last_move == ROCK:
                    choice = SCISSORS
                elif self.last_move == PAPER:
                    choice = PAPER
                elif self.last_move == SCISSORS:
                    choice = SCISSORS
        self.last_move = choice
        return choice
