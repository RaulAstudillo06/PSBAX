#!/usr/bin/env python3
from typing import Callable

import os
import sys
import torch
import numpy as np
import argparse

import gpytorch

from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
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

parser = argparse.ArgumentParser()
parser.add_argument('--policy', type=str, default='ps')
parser.add_argument('--trials', type=int, default=5)
parser.add_argument('--dim', type=int, default=5)
parser.add_argument('--max_iter', type=int, default=100)
parser.add_argument('--samp_str', type=str, default='mut')
parser.add_argument('--model_type', type=str, default='gp')
parser.add_argument('--save', '-s', action='store_true', default=False)
parser.add_argument('--restart', '-r', action='store_true', default=False)
parser.add_argument('--opt_mode', type=str, default='max')
args = parser.parse_args()

# === To RUN === # 
# python ackley_runner.py -s --dim 10 --max_iter 100 --trials 10 --samp_str mut --policy ps


first_trial = 1
last_trial = args.trials

n_dim = args.dim
domain = [[-5, 10]] * n_dim
# rescale domain to [-32.768, 32.768]
inv_standardize = lambda x: 65.536 * x - 32.768 # from [0, 1] to [-32.768, 32.768]
standardize= lambda x: (x + 32.768) / 65.536 # from [-32.768, 32.768] to [0, 1]
domain = [[standardize(d[0]), standardize(d[1])] for d in domain]

init_x = unif_random_sample_domain(domain, n=1)

def rescale(x, original_lb, original_ub, new_lb, new_ub):
    return (x - original_lb) / (original_ub - original_lb) * (new_ub - new_lb) + new_lb


def obj_func(X: Tensor) -> Tensor:
    if isinstance(X, np.ndarray):
        X = torch.tensor(X)
    ackley = Ackley(dim=n_dim)
    if args.opt_mode == 'min':
        objective_X = ackley.evaluate_true(65.536 * X - 32.768)
    else:
        objective_X = -ackley.evaluate_true(65.536 * X - 32.768)
    return objective_X


algo_params = {
    "n_generation": 50,
    "n_population": 10,
    "samp_str": args.samp_str,
    "init_x": init_x[0],
    "domain": domain,
    "normal_scale": 0.05,
    "keep_frac": 0.3,
    "crop": False,
    "opt_mode": args.opt_mode,
    #'crop': True,
}
algo = EvolutionStrategies(algo_params)

algo_metric = algo.get_copy()


optimum = torch.zeros(1, n_dim)
rescaled_optimum = standardize(optimum)

performance_metrics = [
    # ObjValAtMaxPostMean(obj_func, input_dim),
    BestValue(
        algo_metric, 
        obj_func,
        optimum=rescaled_optimum,
        eval_mode="best_value",
    ),
]




from src.fit_model import fit_model

from src.utils import (
    generate_initial_data,
    generate_random_points,
    get_obj_vals,
    seed_torch,
)


# inputs, obj_vals = generate_initial_data(
#     num_init_points=1000,
#     input_dim=n_dim,
#     obj_func=obj_func,
#     noise_type="noiseless",
#     noise_level=0.0,
# )
inputs = generate_random_points(1000, n_dim)
# scale input to domain
for i in range(n_dim):
    inputs[:, i] = inputs[:, i] * (domain[i][1] - domain[i][0]) + domain[i][0]

inputs = rescale(inputs, 0, 1, domain[0][0], domain[0][1])
obj_vals = get_obj_vals(obj_func, inputs, "noiseless", 0.0)


model = fit_model(
    inputs, 
    obj_vals, 
    model_type=args.model_type,
    # kernel_type="rbf",
)

state_dict = model.state_dict()

# check model fit rbf



experiment_manager(
    problem="ackley" + f"_{n_dim}d_fixed",
    obj_func=obj_func,
    algorithm=algo,
    performance_metrics=performance_metrics,
    input_dim=n_dim,
    # noise_type="noiseless",
    # noise_level=0.0,
    policy=args.policy + f"_model{args.model_type}" + f"_{args.samp_str}", 
    batch_size=1,
    num_init_points=1,
    num_iter=args.max_iter,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=False,
    model_type=args.model_type,
    save_data=args.save,
    bax_num_cand=1000,
    # kernel_type="rbf",
    state_dict=state_dict,
)
