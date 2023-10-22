seed = 1234
evotorch_seed = 1234
deterministic_matches = True
allow_model_rng_access = False

# allows the introduction of asymmetries that destabilize
#  the trivial always-random strategy
score_weights = {
    'rock': 1.0,
    'paper': 1.0,
    'scissors': 1.0
}
warmup_rounds = 0
total_rounds = 50
popsize = 250
log_interval = 1
stdev_init = 10.0
initial_bounds = 5.0
generations = 2000
ROCK = 0
PAPER = 1
SCISSORS = 2


