#!/usr/bin/env python3

import os
import numpy as np
import sys
import torch
import argparse
from botorch.settings import debug
from botorch.test_functions.synthetic import Rosenbrock
from torch import Tensor

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
debug._set_state(False)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
print(script_dir[:-12])
sys.path.append(script_dir[:-12])

from src.bax.alg.evolution_strategies import EvolutionStrategies
from src.bax.alg.lbfgsb import LBFGSB
from src.experiment_manager import experiment_manager
from src.performance_metrics import BestValue


# use argparse
parser = argparse.ArgumentParser()
parser.add_argument('--policy', type=str, default='random')
parser.add_argument('--samp_str', type=str, default='cma')
parser.add_argument('--model_type', type=str, default='gp')
parser.add_argument('--dim', type=int, default=5)
parser.add_argument('--trials', type=int, default=5)
parser.add_argument('--first_trial', type=int, default=1)
args = parser.parse_args()

first_trial = args.first_trial
last_trial = args.first_trial + args.trials - 1

# Objective function
def obj_func(X: Tensor) -> Tensor:
    if isinstance(X, np.ndarray):
        X = torch.tensor(X)
    f = Rosenbrock(negate=True, dim=args.dim) # negate=True is for maximization
    objective_X = f(4.0 * X - 2.0)
    return objective_X


# Algorithm
algo_id = "lbfgsb"

n_dim = args.dim

if algo_id == "cma":
    domain = [[0, 1]] * n_dim
    init_x = [[0.0] * n_dim]

    algo_params = {
        "n_generation": 100 * n_dim,
        "n_population": 10,
        "samp_str": args.samp_str,
        "init_x": init_x[0],
        "domain": domain,
        "crop": True,
        "opt_mode": "max",
    }
    algo = EvolutionStrategies(algo_params)
elif algo_id == "lbfgsb":
    num_restarts = 5 * n_dim
    raw_samples = 100 * n_dim

    algo_params = {
        "name" : "LBFGSB",
        "opt_mode" : "max",
        "n_dim" : n_dim,
        "num_restarts" : num_restarts,
        "raw_samples" : raw_samples,
    }
    algo = LBFGSB(algo_params)


# Performance metric
algo_metric = algo.get_copy()
performance_metrics = [
    BestValue(algo_metric, obj_func),
]


experiment_manager(
    problem=f"rosenbrock_{n_dim}d",
    obj_func=obj_func,
    algorithm=algo,
    performance_metrics=performance_metrics,
    input_dim=n_dim,
    noise_type="noiseless",
    noise_level=0.0,
    policy=args.policy + f"_{args.model_type}" + f"_{algo_id}",
    batch_size=1,
    num_init_points=2 * (n_dim + 1),
    num_iter=200,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=False,
    model_type=args.model_type,
    save_data=True,
    bax_num_cand=1000 * n_dim,
    exe_paths=30,
)
