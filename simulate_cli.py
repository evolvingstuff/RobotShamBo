from utils import load_champion
from config import *


def main():
    """
    Play against the current champion in the CLI
    """
    ai = load_champion()
    print('Playing Rock-Paper-Scissors')
    print('---------------------------')
    print('"r" = rock')
    print('"p" = paper')
    print('"s" = scissors')
    print('')
    print('"q" to quit')
    round = 1
    wins, losses, draws = 0, 0, 0
    human_action = None
    while True:
        print('===============================')
        ai_action = ai.move(human_action)
        inp = input(f'round {round}: ')
        if inp == 'q':
            break
        elif inp == 'r':
            human_action = ROCK
        elif inp == 'p':
            human_action = PAPER
        elif inp == 's':
            human_action = SCISSORS
        else:
            raise NotImplementedError
        if ai_action == ROCK:
            print('ROCK')
            if human_action == ROCK:
                print('draw')
                draws += 1
            elif human_action == PAPER:
                print('PAPER wins!')
                wins += 1
            elif human_action == SCISSORS:
                print('SCISSORS lose :(')
                losses += 1
        elif ai_action == PAPER:
            print('PAPER')
            if human_action == ROCK:
                print('ROCK loses :(')
                losses += 1
            elif human_action == PAPER:
                print('draw')
                draws += 1
            elif human_action == SCISSORS:
                print('SCISSORS win!')
                wins += 1
        elif ai_action == SCISSORS:
            print('SCISSORS')
            if human_action == ROCK:
                print('ROCK wins!')
                wins += 1
            elif human_action == PAPER:
                print('PAPER loses :(')
                losses += 1
            elif human_action == SCISSORS:
                print('draw')
                draws += 1
        print(f'{wins} wins | {losses} losses | {draws} draws')
        round += 1


if __name__ == '__main__':
    main()
