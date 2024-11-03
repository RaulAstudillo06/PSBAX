
#%%
import os
import sys
import math
sys.setrecursionlimit(10000) 
import torch
import json
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
# debug._set_state(False)

from botorch.test_functions.synthetic import Levy

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
src_dir = "/".join(script_dir.split("/")[:-2]) # src directory is two levels up
sys.path.append(src_dir)

from src.algorithms.levelset import LevelSetEstimator
# from src.bax.alg.levelset import LevelSetEstimator
from src.performance_metrics import F1Score
from src.acquisition_functions.lse import LSE
from src.experiment_manager import experiment_manager
from src.utils import reshape_mesh, get_mesh


parser = argparse.ArgumentParser()
parser.add_argument('--policy', type=str, default='bax')
parser.add_argument('--problem', type=str, default='volcano')
parser.add_argument('--tau', type=float, default=0.55)
parser.add_argument('--dim', type=int, default=3)
parser.add_argument('--n_init', type=int, default=0)
parser.add_argument('--first_trial', type=int, default=1)
parser.add_argument('--trials', type=int, default=5)
parser.add_argument('--max_iter', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--steps', type=int, default=50)
parser.add_argument('--save', '-s', action='store_true', default=False)
parser.add_argument('--restart', '-r', action='store_true', default=False)
args = parser.parse_args()
#%%

def get_threshold(f, tau, n=10000):
    x_test = torch.rand(n, args.dim)
    f_test = f(x_test)
    f_test_sorted, _ = torch.sort(f_test, descending=False)
    idx = int(tau * len(f_test_sorted))
    threshold = f_test_sorted[idx]
    return threshold.item()

if args.problem == "volcano":
    args.dim = 2
    # bounds = [0, 1]
    mat = np.loadtxt(f"{script_dir}/data/volcano_maungawhau.csv", delimiter=",")
    x1 = np.linspace(0, 1, mat.shape[1]) # (61, )
    x2 = np.linspace(0, 1, mat.shape[0]) # (87, )
    mat = mat.flatten()
    xx = np.meshgrid(x1, x2)
    x_set = np.hstack([xx[0].reshape(-1, 1), xx[1].reshape(-1, 1)]) # (87*61, 2)
    x_to_elevation = {tuple(x): mat[i] for i, x in enumerate(x_set)}
    threshold = 165
    idx = np.argmax(mat)
    x_init = torch.from_numpy(np.atleast_2d(x_set[idx]))
elif args.problem == "himmelblau":
    args.dim = 2
    args.steps = 50
    args.tau = 0.55
    bounds = [-6, 6]
    def himmelblau(X: torch.Tensor, minimize=False) -> torch.Tensor:
        X = (bounds[1] - bounds[0]) * X + bounds[0]
        a = X[:, 0]
        b = X[:, 1]
        result = (a ** 2 + b - 11) ** 2 + (a + b ** 2 - 7) ** 2
        if not minimize:
            return -result
        return result

    threshold = get_threshold(himmelblau, args.tau) # - 147.96
    xx = get_mesh(args.dim, args.steps)
    x_set = reshape_mesh(xx).numpy()
    fx = himmelblau(torch.tensor(x_set))
    x_to_elevation = {tuple(x): f for x, f in zip(x_set, fx)}
    idx = torch.argmax(fx)
    x_init = torch.tensor(x_set[idx]).reshape(1, -1)
elif args.problem == "griewank":
    args.dim = 3
    args.steps = 15
    bounds = [-5, 5]
    def griewank(X: torch.Tensor, minimize=False) -> torch.Tensor:
        X = (bounds[1] - bounds[0]) * X + bounds[0]
        a = X[:, 0]
        b = X[:, 1]
        result = 1 + (a ** 2 + b ** 2) / 4000 - torch.cos(a) * torch.cos(b / math.sqrt(2))
        if not minimize:
            return -result
        return result

    threshold = get_threshold(griewank, args.tau)
    print(f"Threshold: {threshold}")
    xx = get_mesh(args.dim, args.steps)
    x_set = reshape_mesh(xx).numpy()
    fx = griewank(torch.tensor(x_set))
    x_to_elevation = {tuple(x): f for x, f in zip(x_set, fx)}
    idx = torch.argmax(fx)
    x_init = torch.tensor(x_set[idx]).reshape(1, -1)
elif args.problem == "levy":
    args.dim = 3
    args.steps = 15
    args.tau = 0.8
    bounds = [-10, 10]
    f = Levy(dim=args.dim, negate=True)
    def levy(X: torch.Tensor, minimize=False) -> torch.Tensor:
        X = (bounds[1] - bounds[0]) * X + bounds[0]
        return f(X)
    threshold = get_threshold(levy, args.tau)
    print(f"Threshold: {threshold}")
    xx = get_mesh(args.dim, args.steps)
    x_set = reshape_mesh(xx).numpy()
    fx = levy(torch.tensor(x_set))
    x_to_elevation = {tuple(x): f for x, f in zip(x_set, fx)}
    idx = torch.argmax(fx)
    x_init = torch.tensor(x_set[idx]).reshape(1, -1)
elif args.problem == "alpine":
    args.dim = 3
    args.steps = 15
    args.tau = 0.9
    bounds = [-10, 10]
    def alpine(X: torch.Tensor, minimize=False) -> torch.Tensor:
        X = (bounds[1] - bounds[0]) * X + bounds[0]
        result = torch.sum(torch.abs(X * torch.sin(X) + 0.1 * X), dim=-1)
        if not minimize:
            return -result
        return result
    threshold = get_threshold(alpine, args.tau)
    print(f"Threshold: {threshold}")
    xx = get_mesh(args.dim, args.steps)
    x_set = reshape_mesh(xx).numpy()
    fx = alpine(torch.tensor(x_set))
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


algo_params = {
    "name" : "SimpleLevelSet",
    "threshold" : threshold,
    "x_set" : x_set,
    # "x_init" : x_init,
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

# if "lse" in args.policy:
#     acq_func = LSE(
#         x_set,
#         threshold,
#     )
    

if args.n_init == 0:
    args.n_init = 2 * (args.dim + 1)

problem = "levelset" + f"_{args.problem}"
if args.save:
    results_dir = f"{script_dir}/results/{problem}"
    os.makedirs(results_dir, exist_ok=True)
    policy = args.policy
    params_dict = vars(args)
    for k,v in algo_params.items():
        if k not in params_dict and k != "x_set":
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
    input_dim=args.dim,
    noise_type="noiseless",
    noise_level=0.0,
    policy=args.policy,
    batch_size=args.batch_size,
    num_init_points=args.n_init,
    num_iter=args.max_iter,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=args.restart,
    save_data=args.save,
    x_set=x_set,
    x_init=x_init,
    # acq_func=acq_func if args.policy == "lse" else None,
    threshold=threshold,
)



#%%