import os
import sys
import torch
import json
import numpy as np
import subprocess


from botorch.test_functions.synthetic import Levy
from botorch.settings import debug

sys.setrecursionlimit(10000) 
torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
debug._set_state(False)
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
src_dir = "/".join(script_dir.split("/")[:-1]) # src directory is one level up
sys.path.append(src_dir)

from src.algorithms.levelset import LevelSetEstimator
from src.performance_metrics import F1Score
from src.experiment_manager import experiment_manager
from src.utils import reshape_mesh, get_mesh

args = {
    "dim": 2,
    "steps": 50,
    "tau": 0.55,
    "batch_size": 1,
    "max_iter": 30,
}

function = "himmelblau"
dim = args["dim"]
steps = args["steps"]
tau = args["tau"]
batch_size = args["batch_size"]
max_iter = args["max_iter"]
bounds = [-6, 6]
policies = ["random", "ps"]

def get_threshold(f, tau, n=10000):
    x_test = torch.rand(n, dim)
    f_test = f(x_test)
    f_test_sorted, _ = torch.sort(f_test, descending=False)
    idx = int(tau * len(f_test_sorted))
    threshold = f_test_sorted[idx]
    return threshold.item()

def himmelblau(X: torch.Tensor, minimize=False) -> torch.Tensor:
    X = (bounds[1] - bounds[0]) * X + bounds[0]
    a = X[:, 0]
    b = X[:, 1]
    result = (a ** 2 + b - 11) ** 2 + (a + b ** 2 - 7) ** 2
    if not minimize:
        return -result
    return result


threshold = get_threshold(himmelblau, tau) # - 147.96
xx = get_mesh(dim, steps)
x_set = reshape_mesh(xx).numpy()
fx = himmelblau(torch.tensor(x_set))
x_to_elevation = {tuple(x): f for x, f in zip(x_set, fx)}
idx = torch.argmax(fx)
x_init = torch.tensor(x_set[idx]).reshape(1, -1)

def obj_func(X):
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X)
    y = []
    for x in X:
        y.append(x_to_elevation[tuple(x.tolist())])
    return torch.tensor(y)


problem = "level-set" + f"_{function}"
results_dir = f"{script_dir}/results/{problem}"
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, f"params.json"), "w") as file:
    json.dump(args, file)

for policy in policies:
    algo_params = {
        "name" : "SimpleLevelSet",
        "threshold" : threshold,
        "x_set" : x_set,
        "no_copy" : True,
    }
    algo = LevelSetEstimator(algo_params)

    algo_metric = algo.get_copy()
    performance_metrics = [
        F1Score(
            algo_metric, 
            obj_func,
        )
    ]

    experiment_manager(
        problem=problem,
        obj_func=obj_func,
        algorithm=algo,
        performance_metrics=performance_metrics,
        input_dim=dim,
        noise_type="noiseless",
        noise_level=0.0,
        policy=policy,
        batch_size=batch_size,
        num_init_points=2 * (dim + 1),
        num_iter=max_iter,
        first_trial=1,
        last_trial=5,
        restart=False,
        save_data=True,
        x_set=x_set,
        x_init=x_init,
        threshold=threshold,
    )

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