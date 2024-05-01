#!/usr/bin/env python3
from typing import Callable

import os
import numpy as np
import sys
import torch
import json
import argparse
from botorch.acquisition.analytic import PosteriorMean
from botorch.settings import debug
from botorch.test_functions.synthetic import Rastrigin
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
from src.performance_metrics import ObjValAtMaxPostMean, BestValue
from src.utils import compute_noise_std

# use argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=10)
parser.add_argument('--policy', type=str, default='ps')
parser.add_argument('--trials', type=int, default=5)
parser.add_argument('--first_trial', type=int, default=1)
parser.add_argument('--noise', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=3)
parser.add_argument('--max_iter', type=int, default=100)
parser.add_argument('--model_type', type=str, default='gp')
parser.add_argument('--save', '-s', action='store_true', default=False)
parser.add_argument('--restart', '-r', action='store_true', default=False)
args = parser.parse_args()

# === To RUN === # 
# python lbfgsb_runner.py -s --policy ps --first_trial 1 --trials 5 --batch_size 5 --max_iter 50


bounds = [-5.12, 5.12]
def obj_func(X: Tensor) -> Tensor:
    if isinstance(X, np.ndarray):
        X = torch.tensor(X)
    
    f = Rastrigin(negate=True, dim=args.dim) # negate=True is for maximization
    X_rescaled = (bounds[1] - bounds[0]) * X + bounds[0]
    objective_X = f(X_rescaled)
    return objective_X


# Set algorithm details
n_dim = args.dim
num_restarts = 6 * n_dim
raw_samples = 180 * n_dim

algo_params = {
    "name" : "LBFGSB",
    "opt_mode" : "max",
    "n_dim" : n_dim,
    "num_restarts" : num_restarts,
    "raw_samples" : raw_samples,
}
algo = LBFGSB(algo_params)

algo_metric = algo.get_copy()
performance_metrics = [
    # ObjValAtMaxPostMean(obj_func, input_dim),
    BestValue(
        algo_metric, 
        obj_func,
        eval_mode="best_value",
        num_runs=1,
    ),
]

problem = "lbfgsb_rastrigin" + f"_{n_dim}d"
if args.noise > 0:
    problem += f"_noise{args.noise}"
    noise_type = "noisy"
    bounds = torch.vstack([torch.zeros(args.n_dim), torch.ones(args.n_dim)])
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
    policy=args.policy + f"_model{args.model_type}", 
    batch_size=args.batch_size,
    num_init_points=2 * (n_dim + 1),
    num_iter=args.max_iter,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=False,
    model_type=args.model_type,
    save_data=args.save,
    # architecture=model_architecture,
    bax_num_cand=5000,
    exe_paths=30,
)
