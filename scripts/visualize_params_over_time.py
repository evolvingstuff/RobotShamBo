import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt


def main():
    print('Visualizing params over time...')
    saved = sorted(glob.glob('../data/*.pickle'))
    if len(saved) == 0:
        print('No pickle files found in data/. Need to run main.py for awhile first.')
        return
    data = []
    for path in saved:
        with open(path, 'rb') as f:
            solution = pickle.load(f)
        x = solution['center']
        data.append(x.numpy())
    data = np.array(data)
    fig, ax = plt.subplots()
    for param_index in range(len(data[0])):
        param_data = data[:, param_index]
        ax.plot(param_data, alpha=0.1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Parameter Value')
    ax.set_title('Parameter Values Over Time')
    plt.show()


if __name__ == '__main__':
    main()
