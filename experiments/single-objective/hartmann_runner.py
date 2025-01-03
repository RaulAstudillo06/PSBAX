#!/usr/bin/env python3

import os
import sys
import torch
import numpy as np
import argparse
import json
from botorch.settings import debug
from botorch.test_functions.synthetic import Hartmann
from torch import Tensor

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
debug._set_state(False)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
src_dir = "/".join(script_dir.split("/")[:-2]) # src directory is two levels up
sys.path.append(src_dir)

from src.algorithms.lbfgsb import LBFGSB
from src.experiment_manager import experiment_manager
from src.performance_metrics import BestValue
from src.utils import compute_noise_std

# Use argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=6)
parser.add_argument('--policy', type=str, default='bax')
parser.add_argument('--trials', type=int, default=30)
parser.add_argument('--first_trial', type=int, default=1)
parser.add_argument('--noise', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--max_iter', type=int, default=100)
parser.add_argument('--model_type', type=str, default='gp')
parser.add_argument('--save', '-s', action='store_true', default=False)
parser.add_argument('--restart', '-r', action='store_true', default=False)
args = parser.parse_args()

# === To RUN === # 
# python hartmann_runner.py -s --trials 10 --policy ps

# Objective function
def obj_func(X: Tensor) -> Tensor:
    if isinstance(X, np.ndarray):
        X = torch.tensor(X)
    hartmann = Hartmann(dim=args.dim, negate=True)
    objective_X = hartmann(X)
    return objective_X

n_dim = args.dim

# Algorithm
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

problem = "hartmann" + f"_{n_dim}d"
if args.noise > 0:
    problem += f"_noise{args.noise}"
    noise_type = "noisy"
    bounds = torch.vstack([torch.zeros(n_dim), torch.ones(n_dim)])
    noise_levels = compute_noise_std(obj_func, args.noise, bounds=bounds)
else:
    noise_type = "noiseless"
    noise_levels = None

if args.save:
    results_dir = f"./results/{problem}"
    os.makedirs(results_dir, exist_ok=True)
    policy = args.policy
    params_dict = vars(args)
    for k,v in algo_params.items():
        if k not in params_dict and k != "ref_point":
            params_dict[k] = v

    with open(os.path.join(results_dir, f"{policy}_{args.batch_size}_params.json"), "w") as file:
        json.dump(params_dict, file)


first_trial = args.first_trial
last_trial = args.first_trial + args.trials - 1


experiment_manager(
    problem=problem,
    obj_func=obj_func,
    algorithm=algo,
    performance_metrics=performance_metrics,
    input_dim=n_dim,
    noise_type=noise_type,
    noise_level=noise_levels,
    policy=args.policy + f"_{args.model_type}",
    batch_size=args.batch_size,
    num_init_points=2 * (n_dim + 1),
    num_iter=args.max_iter,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=args.restart,
    model_type=args.model_type,
    save_data=args.save,
    bax_num_cand=1000 * n_dim,
    exe_paths=30,
)
