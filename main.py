from evotorch import Problem
from evotorch.algorithms import SNES
from evotorch.logging import StdOutLogger, PandasLogger
import matplotlib.pyplot as plt
from players import *
from iocaine import Iocaine


# assumes only player1 is being evolved
player1_class = StaticOpponent


def get_hall_of_champions():
    return [Iocaine()]


def tournament(x: torch.Tensor) -> torch.Tensor:
    combined_score = 0
    for player2 in get_hall_of_champions():
        if deterministic_matches:
            random.seed(seed)
            torch.manual_seed(seed)
        player1_score = 0
        player2_score = 0
        draw_count = 0
        player1 = player1_class()
        player1.set_parameters(x)
        last_player1_action = None
        last_player2_action = None
        for round_num in range(total_rounds):
            player1_action = player1.move(last_player2_action)
            player2_action = player2.move(last_player1_action)
            if round_num >= warmup_rounds:
                if player1_action == player2_action:
                    draw_count += 1
                elif player1_action == ROCK:
                    if player2_action == SCISSORS:
                        player1_score += score_weights['rock']
                    elif player2_action == PAPER:
                        player2_score += score_weights['paper']
                elif player1_action == PAPER:
                    if player2_action == ROCK:
                        player1_score += score_weights['paper']
                    elif player2_action == SCISSORS:
                        player2_score += score_weights['scissors']
                elif player1_action == SCISSORS:
                    if player2_action == PAPER:
                        player1_score += score_weights['scissors']
                    elif player2_action == ROCK:
                        player2_score += score_weights['rock']
            last_player1_action = player1_action
            last_player2_action = player2_action
        combined_score += player1_score - player2_score
        # print(f"Score: player1 = {player1_score}, player2 = {player2_score}, Draws = {draw_count}")
    return torch.tensor(combined_score)


def main():
    print('Robotshambo')
    random.seed(seed)
    torch.manual_seed(seed)

    # assumes only player1 is being evolved
    dim = player1_class().get_dim()

    # # Declare the objective function
    problem = Problem(
        "max",
        tournament,
        initial_bounds=(-initial_bounds, initial_bounds),
        solution_length=dim,
        vectorized=False,
        seed=evotorch_seed
        # device="cuda:0"  # enable this line if you wish to use GPU
    )

    # Initialize the SNES algorithm to solve the problem
    searcher = SNES(problem, popsize=popsize, stdev_init=stdev_init)

    # Initialize a standard output logger, and a pandas logger
    _ = StdOutLogger(searcher, interval=log_interval)
    pandas_logger = PandasLogger(searcher)

    # Run SNES for the specified amount of generations
    searcher.run(generations)

    # Get the progress of the evolution into a DataFrame with the
    # help of the PandasLogger, and then plot the progress.
    pandas_frame = pandas_logger.to_dataframe()
    pandas_frame["best_eval"].plot()
    plt.show()


if __name__ == '__main__':
    main()
