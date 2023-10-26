from src.players import *
from src.iocaine import Iocaine

# assumes only player1 is being evolved
player1_class = Rnn

opponent_classes = [
    Iocaine,
    # RoundRobinPlayer
    # RandomPlayer,
    # RandomStrategyChangePlayer
]