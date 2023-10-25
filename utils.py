import glob
import pickle
from config_agent import player1_class
from config import champion_type


def load_champion():
    saved = sorted(glob.glob('data/*.pickle'))
    if len(saved) == 0:
        path = 'sample_agent.pickle'
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