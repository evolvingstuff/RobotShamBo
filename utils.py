import glob
import pickle
from config_agent import player1_class
from config import champion_type


def load_champion():
    path = sorted(glob.glob('data/*.pickle'))[-1]
    with open(path, 'rb') as f:
        solution = pickle.load(f)
    x = solution[champion_type]
    dim = player1_class().get_dim()
    assert len(x) == dim, 'dimension mismatch'
    champion = player1_class()
    champion.set_parameters(x)
    return champion