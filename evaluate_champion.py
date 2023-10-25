from iocaine import Iocaine
from utils import *
from players import *
import statistics


def main():
    print('Evaluate current champion vs Iocaine symmetric rewards')
    scores = []
    rounds = 1000
    for round in range(rounds):
        player1 = load_champion()
        player2 = Iocaine()
        score = evaluate(player1, player2, balanced_weights)
        scores.append(score)
        print(f'round {round+1}/{rounds}: mean score = {statistics.mean(scores):.4f} | median score = {statistics.median(scores):.4f}')


if __name__ == '__main__':
    main()