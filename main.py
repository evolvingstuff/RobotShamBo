from evotorch import Problem
import torch
from evotorch.algorithms import SNES, PyCMAES
from evotorch.logging import StdOutLogger, PandasLogger, PicklingLogger
import matplotlib.pyplot as plt
from config.config_agents import player1_class, opponent_classes
from config.config import *
from src.utils import evaluate


def tournament(x: torch.Tensor) -> torch.Tensor:
    """
    Compete against multiple opponents, with possibly asymmetric rewards
    """
    combined_score = 0
    for player2_class in opponent_classes:
        player2 = player2_class()
        player1 = player1_class()
        player1.set_parameters(x)
        weights = asymmetric_weights if use_asymmetric_weights_during_evolution else balanced_weights
        combined_score += evaluate(player1, player2, weights, total_rounds)
    return torch.tensor(combined_score)


def main():
    """
    Initialize evolutionary main loop
    """

    print('Robotshambo')

    # Assumes only player1 is being evolved
    dim = player1_class().get_dim()

    # Declare the objective function
    args = ['max', tournament]
    kwargs = {
        'initial_bounds': (-initial_bounds, initial_bounds),
        'solution_length': dim,
        'vectorized': False
        # device="cuda:0"  # enable this if you wish to use GPU
    }
    if distributed:
        kwargs['num_actors'] = 'max'
    problem = Problem(*args, **kwargs)

    # Initialize the SNES algorithm to solve the problem
    # searcher = SNES(problem, popsize=popsize, stdev_init=stdev_init, distributed=distributed)
    searcher = PyCMAES(problem, popsize=popsize, stdev_init=stdev_init)

    # Initialize loggers
    _ = StdOutLogger(searcher, interval=log_interval)
    _ = PicklingLogger(searcher, interval=pickle_interval, directory='data/', prefix='agent',
                       after_first_step=False, verbose=False)
    pandas_logger = PandasLogger(searcher)

    # Run SNES for the specified amount of generations
    searcher.run(generations)

    # Get the progress of the evolution into a DataFrame with the
    # help of the PandasLogger, and then plot the progress.
    pandas_frame = pandas_logger.to_dataframe()
    pandas_frame[visualization_metric].plot()
    plt.show()


if __name__ == '__main__':
    main()
