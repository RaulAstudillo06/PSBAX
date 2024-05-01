#!/usr/bin/env python3
from typing import Callable

import os
import sys
import torch
import numpy as np
import argparse
import json

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
from src.bax.util.domain_util import unif_random_sample_domain
from src.experiment_manager import experiment_manager
from src.performance_metrics import ObjValAtMaxPostMean, compute_obj_val_at_max_post_mean, BestValue
from src.utils import compute_noise_std

parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=10)
parser.add_argument('--policy', type=str, default='bax')
parser.add_argument('--trials', type=int, default=5)
parser.add_argument('--first_trial', type=int, default=1)
parser.add_argument('--noise', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=3)
parser.add_argument('--max_iter', type=int, default=100)
parser.add_argument('--samp_str', type=str, default='mut')
parser.add_argument('--model_type', type=str, default='gp')
parser.add_argument('--save', '-s', action='store_true', default=False)
parser.add_argument('--restart', '-r', action='store_true', default=False)
args = parser.parse_args()

# === To RUN === # 
# python ackley_runner.py -s --dim 10 --max_iter 100 --trials 10 --samp_str mut --policy ps

n_dim = args.dim
bounds = [-5, 10]
domain = [[0, 1]] * n_dim
# rescale domain to [-32.768, 32.768]
inv_standardize = lambda x: 65.536 * x - 32.768 # from [0, 1] to [-32.768, 32.768]
standardize= lambda x: (x + 32.768) / 65.536 # from [-32.768, 32.768] to [0, 1]
# domain = [[standardize(d[0]), standardize(d[1])] for d in domain]

init_x = unif_random_sample_domain(domain, n=1)

def obj_func(X: Tensor) -> Tensor:
    if isinstance(X, np.ndarray):
        X = torch.tensor(X)
    ackley = Ackley(dim=n_dim)
    objective_X = -ackley.evaluate_true((bounds[1] - bounds[0]) * X + bounds[0])
    return objective_X


algo_params = {
    "n_generation": 30,
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

# def obj_val_at_max_post_mean(
#     obj_func: Callable, posterior_mean_func: PosteriorMean
# ) -> Tensor:
#     return compute_obj_val_at_max_post_mean(obj_func, posterior_mean_func, input_dim)
# performance_metrics = {
#     "Objective value at maximizer of the posterior mean": obj_val_at_max_post_mean
# }

optimum = torch.zeros(1, n_dim) # [-32.768, 32.768]
rescaled_optimum = standardize(optimum) # [0, 1]
rescaled_optimum = (bounds[1] - bounds[0]) * rescaled_optimum + bounds[0] # [-5, 10]

performance_metrics = [
    # ObjValAtMaxPostMean(obj_func, input_dim),
    BestValue(
        algo_metric, 
        obj_func,
        optimum=rescaled_optimum,
        eval_mode="best_value",
        num_runs=5,
    ),
]

model_architecture = [32, 32, 4] # Excluding input_dim and output_dim

problem = "ackley" + f"_{n_dim}d"
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
    policy=args.policy + f"_model{args.model_type}" + f"_{args.samp_str}", 
    batch_size=args.batch_size,
    num_init_points=2 * (n_dim + 1),
    num_iter=args.max_iter,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=False,
    model_type=args.model_type,
    save_data=args.save,
    architecture=model_architecture,
    bax_num_cand=5000,
)
