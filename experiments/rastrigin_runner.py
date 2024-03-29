#!/usr/bin/env python3
from typing import Callable

import os
import numpy as np
import sys
import torch
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
from src.experiment_manager import experiment_manager
from src.performance_metrics import ObjValAtMaxPostMean, BestValue


# use argparse
parser = argparse.ArgumentParser()
parser.add_argument('--policy', type=str, default='ps')
parser.add_argument('--trials', type=int, default=5)
parser.add_argument('--dim', type=int, default=5)
parser.add_argument('--samp_str', type=str, default='cma')
parser.add_argument('--model_type', type=str, default='gp')
parser.add_argument('--save', '-s', action='store_true', default=False)
parser.add_argument('--restart', '-r', action='store_true', default=False)
args = parser.parse_args()

first_trial = 1
last_trial = args.trials

# === To RUN === # 
# python rastrigin_runner.py -s --dim 10 --trials 10 --policy bax

def obj_func(X: Tensor) -> Tensor:
    if isinstance(X, np.ndarray):
        X = torch.tensor(X)
    f = Rastrigin(negate=True, dim=args.dim) # negate=True is for maximization
    objective_X = f(X)
    return objective_X


# Set algorithm details
n_dim = args.dim
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

performance_metrics = [
    # ObjValAtMaxPostMean(obj_func, input_dim),
    BestValue(algo, obj_func),
]


experiment_manager(
    problem=f"rastrigin_{n_dim}d",
    obj_func=obj_func,
    algorithm=algo,
    performance_metrics=performance_metrics,
    input_dim=n_dim,
    noise_type="noiseless",
    noise_level=0.0,
    policy=args.policy + f"_model{args.model_type}" + f"_{args.samp_str}",
    batch_size=1,
    num_init_points=2 * (n_dim + 1),
    num_iter=100,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=False,
    model_type=args.model_type,
    save_data=args.save,
    bax_num_cand=5000,
)
