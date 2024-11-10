import os
import sys
import torch
import json
import numpy as np
import subprocess

from botorch.test_functions.synthetic import Hartmann
from botorch.settings import debug

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
debug._set_state(False)
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
src_dir = "/".join(script_dir.split("/")[:-1]) # src directory is one level up
sys.path.append(src_dir)

from src.algorithms.lbfgsb import LBFGSB
from src.experiment_manager import experiment_manager
from src.performance_metrics import BestValue
from src.utils import compute_noise_std

args = {
    "dim": 3,
    "batch_size": 1,
    "max_iter": 30,
    "noise": 0.0,
}
function = "hartmann"
n_dim = args["dim"]
batch_size = args["batch_size"]
max_iter = args["max_iter"]
noise = args["noise"]
policies = ["random", "bax", "ps"]

def obj_func(X):
    if isinstance(X, np.ndarray):
        X = torch.tensor(X)
    hartmann = Hartmann(dim=n_dim, negate=True)
    objective_X = hartmann(X)
    return objective_X

num_restarts = 5 * n_dim
raw_samples = 100 * n_dim

algo_params = {
    "name" : "LBFGSB",
    "opt_mode" : "max",
    "n_dim" : n_dim,
    "num_restarts" : num_restarts,
    "raw_samples" : raw_samples,
}

problem = "local-bo" + f"_{function}_{n_dim}d"
if noise > 0:
    problem += f"_noise{noise}"
    noise_type = "noisy"
    bounds = torch.vstack([torch.zeros(n_dim), torch.ones(n_dim)])
    noise_levels = compute_noise_std(obj_func, noise, bounds=bounds)
else:
    noise_type = "noiseless"
    noise_levels = None

results_dir = f"{script_dir}/results/{problem}"
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, f"params.json"), "w") as file:
    json.dump(args, file)


# for policy in policies:
#     algo = LBFGSB(algo_params)

#     # Performance metric
#     algo_metric = algo.get_copy()
#     performance_metrics = [
#         BestValue(algo_metric, obj_func),
#     ]

#     experiment_manager(
#         problem=problem,
#         obj_func=obj_func,
#         algorithm=algo,
#         performance_metrics=performance_metrics,
#         input_dim=n_dim,
#         noise_type=noise_type,
#         noise_level=noise_levels,
#         policy=policy,
#         batch_size=batch_size, 
#         num_init_points=2 * (n_dim + 1),
#         num_iter=max_iter,
#         first_trial=1,
#         last_trial=3,
#         restart=False,
#         save_data=True,
#         bax_num_cand=1000 * n_dim,
#         exe_paths=30,
#     )

# graph results
policies_str = ",".join(policies)
command = " ".join([
    "python",
    f"{script_dir}/graph.py",
    "-s",
    "-t",
    f"--problem {problem}",
    f"--policies {policies_str}",
])

subprocess.run(command, shell=True)