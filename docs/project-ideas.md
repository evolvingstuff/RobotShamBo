# Project Ideas

## Thoughts/Questions
* Can GPT-4 play a good game of rock paper scissors?
  * could set up this experiment up through the OpenAI API 
* Can neuro-evolution lead to something that surpasses Iocaine Powder, in the sense that it is a very good general strategy?
* Can neuro-evolution attain the level of Iocaine Powder purely through self-play?
  * If so, how large/complex would it need to be?
* Is there an upper bound of complexity for RPS strategies?
  * Assuming not, would the evaluation period need to grow exponentially?
  * Also, if not, could RPS (or other generic task) act to "train" or "grow" a foundation model for time series in general? 
* Other than asymmetric rewards, are there ways of avoiding simplistic local minima?
* Is there any relationship between RPS and compression algorithms?
  * Could this be a useful medium/input? 
* If we let evolution go on for a looooong time, do we ever observe things like double descent or grokking take place with generalization?
* How well would a purely supervised learner do at this?

## Experiments
* "Hall of champions" for improved self-play and generality.
* More model types
  * Transformers
  * LSM/Echo State Networks?
* Larger models
  * More hidden layers
  * Larger hidden layers
* RL algorithms / online learning

## Possible Features
* Parallelization
  * helper function to setup Ray cluster on, say, AWS 
* Hyperparameter search
  * EvoTorch, Optuna, ...?
* Named experiments, better config handling
  * SQLite? 
* Integration and unit tests
* Web app leaderboard
* GPU support
  * (current RNN models are too small for GPUs to help much) 
* More visualizations
  * Moving average of various n-grams through time used by agents