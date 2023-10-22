# RobotShamBo

<img src="images/robotshambo.png" width="500" height="500">

## Introduction
RobotShamBo aims to explore the capabilities of neuro-evolution applied to the classic game of Rock-Paper-Scissors. This project employs various forms of neural networks to model intelligent agents and observes how they evolve to play the game more optimally.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## Installation

Clone the repository:

```
git clone https://github.com/evolvingstuff/RobotShamBo.git
```

Navigate to the project directory and install the required packages:

```
cd RobotShamBo
pip install -r requirements.txt
```

## Usage

To run the main program:

```
python main.py
```

You can customize the evolutionary parameters and neural network architecture in `config.py`.

## Features wishlist

- **Evolutionary Algorithms**: Implement evolutionary algorithms to evolve optimal decision-making strategies.
- **Reinforcement Learning**: Train agents using popular RL algorithms like Q-Learning and DQN.
- **Neural Networks**: Utilize various architectures including Feedforward, LSTM, and ConvNets.
- **Simulation**: Interactive mode to play against the trained agents.
- **Visualization**: Tools for visualizing decision-making strategies and evolution over time.


## Libraries Used

- **EvoTorch**: This project relies on the EvoTorch library for various evolutionary algorithms in Torch. 
  - **License**: Apache License 2.0. You can view the full license details in the [EvoTorch repository](https://github.com/nnaisense/evotorch/blob/master/LICENSE).
  - **Repository**: [EvoTorch GitHub](https://github.com/nnaisense/evotorch)


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Iocaine Powder Algorithm Implementation**: This project makes use of the Iocaine Powder algorithm as found [here](http://davidbau.com/downloads/rps/rps-iocaine.py), by [David Bau](http://davidbau.com/). This is, in turn, an adaptation of the original code/algorithm, written by Dan Egnor. We have made modifications to fit the requirements of the RobotShamBo project.

