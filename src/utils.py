import glob
import pickle
from config.config_agents import player1_class
from config.config import champion_type, warmup_rounds, ROCK, PAPER, SCISSORS


def load_champion():
    saved = sorted(glob.glob('data/*.pickle'))
    if len(saved) == 0:
        path = 'pretrained/sample_rnn_agent.pickle'
    else:
        path = saved[-1]
    with open(path, 'rb') as f:
        solution = pickle.load(f)
    x = solution[champion_type]
    dim = player1_class().get_dim()
    assert len(x) == dim, 'dimension mismatch'
    champion = player1_class()
    champion.set_parameters(x)
    return champion


def evaluate(player1, player2, weights, rounds):
    player1_score = 0
    player2_score = 0
    draw_count = 0
    last_player1_action = None
    last_player2_action = None
    for round_num in range(rounds):
        player1_action = player1.move(last_player2_action)
        player2_action = player2.move(last_player1_action)
        last_player1_action = player1_action
        last_player2_action = player2_action
        if round_num >= warmup_rounds:
            if player1_action == player2_action:
                draw_count += 1
            elif player1_action == ROCK:
                if player2_action == SCISSORS:
                    player1_score += weights['rock']
                elif player2_action == PAPER:
                    player2_score += weights['paper']
            elif player1_action == PAPER:
                if player2_action == ROCK:
                    player1_score += weights['paper']
                elif player2_action == SCISSORS:
                    player2_score += weights['scissors']
            elif player1_action == SCISSORS:
                if player2_action == PAPER:
                    player1_score += weights['scissors']
                elif player2_action == ROCK:
                    player2_score += weights['rock']
    return player1_score - player2_score