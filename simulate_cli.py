import pickle
import glob
from config_agent import *


def main():
    path = sorted(glob.glob('data/*.pickle'))[-1]
    with open(path, 'rb') as f:
        solution = pickle.load(f)
    x = solution['center']
    dim = player1_class().get_dim()
    assert len(x) == dim, 'dimension mismatch'
    player1 = player1_class()
    player1.set_parameters(x)
    print('Playing Rock-Paper-Scissors')
    print('---------------------------')
    print('"r" = rock')
    print('"p" = paper')
    print('"s" = scissors')
    print('')
    print('"q" to quit')
    round = 1
    wins, losses, draws = 0, 0, 0
    player2_action = None
    while True:
        print('===============================')
        player1_action = player1.move(player2_action)
        inp = input(f'round {round}: ')
        if inp == 'q':
            break
        elif inp == 'r':
            player2_action = ROCK
        elif inp == 'p':
            player2_action = PAPER
        elif inp == 's':
            player2_action = SCISSORS
        else:
            raise NotImplementedError
        if player1_action == ROCK:
            print('ROCK')
            if player2_action == ROCK:
                print('draw')
                draws += 1
            elif player2_action == PAPER:
                print('PAPER wins!')
                wins += 1
            elif player2_action == SCISSORS:
                print('SCISSORS lose :(')
                losses += 1
        elif player1_action == PAPER:
            print('PAPER')
            if player2_action == ROCK:
                print('ROCK loses :(')
                losses += 1
            elif player2_action == PAPER:
                print('draw')
                draws += 1
            elif player2_action == SCISSORS:
                print('SCISSORS win!')
                wins += 1
        elif player1_action == SCISSORS:
            print('SCISSORS')
            if player2_action == ROCK:
                print('ROCK wins!')
                wins += 1
            elif player2_action == PAPER:
                print('PAPER loses :(')
                losses += 1
            elif player2_action == SCISSORS:
                print('draw')
                draws += 1
        print(f'{wins} wins | {losses} losses | {draws} draws')
        round += 1


if __name__ == '__main__':
    main()
