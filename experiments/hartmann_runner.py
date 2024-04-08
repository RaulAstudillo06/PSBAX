#!/usr/bin/env python3
from typing import Callable

import os
import sys
import torch
import numpy as np
import argparse
from botorch.acquisition.analytic import PosteriorMean
from botorch.settings import debug
from botorch.test_functions.synthetic import Hartmann
from torch import Tensor

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
debug._set_state(False)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
print(script_dir[:-12])
sys.path.append(script_dir[:-12])

from src.bax.alg.evolution_strategies import EvolutionStrategies
from src.experiment_manager import experiment_manager
from src.performance_metrics import ObjValAtMaxPostMean, BestValue


# Objective function
input_dim = 6 # FIXME: is this different from n_dim?

# use argparse
parser = argparse.ArgumentParser()
parser.add_argument('--policy', type=str, default='ps')
parser.add_argument('--trials', type=int, default=5)
parser.add_argument('--dim', type=int, default=6)
parser.add_argument('--max_iter', type=int, default=100)
parser.add_argument('--samp_str', type=str, default='cma')
parser.add_argument('--model_type', type=str, default='gp')
parser.add_argument('--save', '-s', action='store_true', default=False)
parser.add_argument('--restart', '-r', action='store_true', default=False)
args = parser.parse_args()

first_trial = 1
last_trial = args.trials

# === To RUN === # 
# python hartmann_runner.py -s --trials 10 --policy ps

def obj_func(X: Tensor) -> Tensor:
    if isinstance(X, np.ndarray):
        X = torch.tensor(X)
    hartmann = Hartmann(dim=input_dim)
    objective_X = -hartmann.evaluate_true(X)
    return objective_X


# Set algorithm details
n_dim = input_dim
domain = [[0, 1]] * n_dim
init_x = [[0.0] * n_dim]

algo_params = {
    "n_generation": 50,
    "n_population": 10,
    "samp_str": args.samp_str,
    "init_x": init_x[0],
    "domain": domain,
    "normal_scale": 0.05,
    "keep_frac": 0.3,
    "crop": False,
    "opt_mode": "max",
    #'crop': True,
}
algo = EvolutionStrategies(algo_params)

algo_metric = algo.get_copy()

performance_metrics = [
    # ObjValAtMaxPostMean(obj_func, input_dim),
    BestValue(algo_metric, obj_func),
]

# # Run experiment
# if len(sys.argv) == 3:
#     first_trial = int(sys.argv[1])
#     last_trial = int(sys.argv[2])
# elif len(sys.argv) == 2:
#     first_trial = int(sys.argv[1])
#     last_trial = int(sys.argv[1])
# else:
#     first_trial = 1
#     last_trial = 5



experiment_manager(
    problem=f"hartmann_{args.dim}d",
    obj_func=obj_func,
    algorithm=algo,
    performance_metrics=performance_metrics,
    input_dim=input_dim,
    noise_type="noiseless",
    noise_level=0.0,
    policy=args.policy + f"_model{args.model_type}" + f"_{args.samp_str}",
    batch_size=1,
    num_init_points=2 * (input_dim + 1),
    num_iter=args.max_iter,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=False,
    model_type=args.model_type,
    save_data=args.save,
    bax_num_cand=5000,
)
