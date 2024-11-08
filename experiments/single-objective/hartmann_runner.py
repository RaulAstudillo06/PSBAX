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

from src.bax.alg.evolution_strategies import EvolutionStrategies
from src.bax.alg.lbfgsb import LBFGSB
from src.experiment_manager import experiment_manager
from src.performance_metrics import BestValue
from src.utils import compute_noise_std


# Use argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=6)
parser.add_argument('--policy', type=str, default='bax')
parser.add_argument('--trials', type=int, default=5)
parser.add_argument('--first_trial', type=int, default=1)
parser.add_argument('--noise', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--max_iter', type=int, default=200)
parser.add_argument('--samp_str', type=str, default='mut')
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
    policy=args.policy + f"_{args.model_type}" + f"_{algo_id}",
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
