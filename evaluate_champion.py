from iocaine import Iocaine
from utils import *
from players import *
import statistics


def main():
    print('Evaluate current champion vs Iocaine using symmetric rewards')
    scores = []
    for game in range(evaluation_games):
        player1 = load_champion()
        player2 = Iocaine()
        weights = balanced_weights  # instead of asymmetric
        score = evaluate(player1, player2, weights, total_rounds)
        scores.append(score)
        print(f'game {game+1}/{evaluation_games}: mean score = {statistics.mean(scores):.4f} | median score = {statistics.median(scores):.4f}')


if __name__ == '__main__':
    main()