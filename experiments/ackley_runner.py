#!/usr/bin/env python3
from typing import Callable

import os
import sys
import torch
from botorch.acquisition.analytic import PosteriorMean
from botorch.settings import debug
from botorch.test_functions.synthetic import Ackley
from torch import Tensor

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
debug._set_state(False)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
print(script_dir[:-12])
sys.path.append(script_dir[:-12])

from src.bax.alg.evolution_strategies import EvolutionStrategies
from src.experiment_manager import experiment_manager
from src.performance_metrics import ObjValAtMaxPostMean, compute_obj_val_at_max_post_mean


# Objective function
input_dim = 5 # FIXME: is this different from n_dim?


def obj_func(X: Tensor) -> Tensor:
    ackley = Ackley(dim=input_dim)
    objective_X = -ackley.evaluate_true(65.536 * X - 32.768)
    return objective_X


# Set algorithm details
n_dim = input_dim
domain = [[0, 1]] * n_dim
init_x = [[0.0] * n_dim]

algo_params = {
    "n_generation": 50,
    "n_population": 10,
    "samp_str": "mut",
    "opt_mode": "min",
    "init_x": init_x[0],
    "domain": domain,
    "normal_scale": 0.05,
    "keep_frac": 0.3,
    "crop": False,
    "opt_mode": "max",
    #'crop': True,
}
algo = EvolutionStrategies(algo_params)

def obj_val_at_max_post_mean(
    obj_func: Callable, posterior_mean_func: PosteriorMean
) -> Tensor:
    return compute_obj_val_at_max_post_mean(obj_func, posterior_mean_func, input_dim)


# performance_metrics = {
#     "Objective value at maximizer of the posterior mean": obj_val_at_max_post_mean
# }

performance_metrics = [
    ObjValAtMaxPostMean(obj_func, input_dim),
]

# Policies
policy = "ps"

# Run experiment
if len(sys.argv) == 3:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[2])
elif len(sys.argv) == 2:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[1])
else:
    first_trial = 1
    last_trial = 1

experiment_manager(
    problem="ackley",
    obj_func=obj_func,
    algorithm=algo,
    performance_metrics=performance_metrics,
    input_dim=input_dim,
    noise_type="noiseless",
    noise_level=0.0,
    policy=policy,
    batch_size=1,
    num_init_points=2 * (input_dim + 1),
    num_iter=200,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=False,
    model_type="dkgp",
    save_data=True,
)
