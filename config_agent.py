from players import *
from iocaine import Iocaine

# assumes only player1 is being evolved
player1_class = Rnn

opponents = [
    Iocaine(),
    RandomPlayer(),
    RotatingPlayer(),
    RandomStrategyChangePlayer()
]