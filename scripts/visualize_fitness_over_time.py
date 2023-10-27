import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt


def main():
    print('Visualizing eval over time...')
    saved = sorted(glob.glob('../data/*.pickle'))
    if len(saved) == 0:
        print('No pickle files found in data/. Need to run main.py for awhile first.')
        return
    data = []
    for path in saved:
        with open(path, 'rb') as f:
            solution = pickle.load(f)
        data.append(solution['mean_eval'])
    data = np.array(data)
    fig, ax = plt.subplots()
    ax.plot(data, alpha=1.0)
    ax.set_xlabel('Generations')
    ax.set_ylabel('Fitness Value')
    ax.set_title('Fitness Values Over Time')
    plt.show()


if __name__ == '__main__':
    main()
