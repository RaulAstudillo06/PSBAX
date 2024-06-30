from typing import Callable

import os
import sys
import json
import torch
import numpy as np
import argparse
from botorch.acquisition.analytic import PosteriorMean
from botorch.settings import debug
from botorch.test_functions.synthetic import Hartmann, Rosenbrock
from torch import Tensor


torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
debug._set_state(False)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
src_dir = "/".join(script_dir.split("/")[:-2]) # src directory is two levels up
sys.path.append(src_dir)

from src.bax.alg.dijkstra import Dijkstra
from src.bax.util.domain_util import unif_random_sample_domain
from src.bax.util.graph import make_grid, edges_of_path, positions_of_path, area_of_polygons

from src.experiment_manager import experiment_manager
from src.performance_metrics import compute_obj_val_at_max_post_mean, ShortestPathCost, ShortestPathArea

# use argparse
parser = argparse.ArgumentParser()
parser.add_argument('--problem', type=str, default='multimodal')
parser.add_argument('--policy', type=str, default='ps')
parser.add_argument('--first_trial', type=int, default=1)
parser.add_argument('--trials', type=int, default=5)
parser.add_argument('--max_iter', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--save', '-s', action='store_true', default=False)
parser.add_argument('--restart', '-r', action='store_true', default=False)
args = parser.parse_args()

input_dim = 2
grid_size = 10

rescaled_x1_lims = (0, 1)
rescaled_x2_lims = (0, 1)
positions, vertices, edges = make_grid(grid_size, rescaled_x1_lims, rescaled_x2_lims)

if args.problem == "rosenbrock":
    x1_lims = (-2, 2)
    x2_lims = (-1, 4)
    start, goal = vertices[-grid_size], vertices[-1]
elif args.problem == "multimodal":
    x1_lims = (-5, 5)
    x2_lims = (-5, 5)
    start, goal = vertices[0], vertices[-1]

edge_coords = positions

# Set function
def rosenbrock(x, y, a=1, b=100):
    # NOTE rescaled to improve numerics
    # NOTE min cost path: 1.0527267184880365
    return 1e-2 * ((a - x)**2 + b * (y - x**2)**2)

def objective(X):
    if len(X.shape) > 2:
        X = X.squeeze()
    x, y = X[..., 0], X[..., 1]
    return (x**2 + y - 11)**2 + (x + y**2 -7)**2

def true_f(x_y):
    x_y = np.array(x_y).reshape(-1)
    return rosenbrock(x_y[..., 0], x_y[..., 1])

def inv_softplus(x):
    return np.log(np.exp(x) - 1)

# NOTE: this is the function we will use
def true_latent_f(x_y):
    '''
    Args:
        x_y: np.array (n_dim, )
    '''
    return inv_softplus(true_f(x_y))

def softplus(x):
    return np.log1p(np.exp(x))

rescaled = True
def obj_func(X, rescaled=rescaled, minimize=True, bounds=[x1_lims, x2_lims], latent_f=True):
    '''
    Args:
        X: (1, n_dim)
    Returns:
        torch.tensor : (1, 1) for botorch model fitting
    '''
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X).reshape(-1, 2)
    X_rescaled = X.clone()
    # Scale X to bounds
    for i in range(X_rescaled.shape[-1]):
        X_rescaled[:, i] = X_rescaled[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]

    if args.problem == "rosenbrock":
        f = Rosenbrock(dim=input_dim)
        f_X = f.evaluate_true(X_rescaled)
        
    elif args.problem == "multimodal":
        f_X = objective(X_rescaled)
    else:
        raise NotImplementedError
    if rescaled:
        f_X = 1e-3 * f_X
    if not minimize:
        f_X = -f_X
    
    def inv_softplus(x):
        return torch.log(torch.exp(x) - 1)
    
    return inv_softplus(f_X)
     
def cost_func(u, v, f, latent_f=True):
    u_pos, v_pos = u.position, v.position
    edge = (u_pos + v_pos) / 2
    edge_cost = f(edge)
    if latent_f:
        return softplus(edge_cost), [edge], [edge_cost]
    else:
        return edge_cost, [edge], [edge_cost]

# algo_params = {
#     'cost_func': lambda u, v, f: cost_func(u, v, f, latent_f=True),
#     'true_cost': lambda u, v: cost_func(u, v, true_f, latent_f=False)
# }
algo_params = {
    'cost_func': lambda u, v, f: cost_func(u, v, f, latent_f=True),
    'true_cost': lambda u, v: cost_func(u, v, obj_func, latent_f=True)
}
algo = Dijkstra(algo_params, vertices, start, goal)

algo_metric = algo.get_copy()
performance_metrics = [
    ShortestPathCost(algo_metric),
    ShortestPathArea(algo_metric, obj_func),
]

problem = "dijkstra_" + args.problem
if args.save:
    results_dir = f"{script_dir}/results/{problem}"
    os.makedirs(results_dir, exist_ok=True)
    policy = args.policy
    params_dict = vars(args)
    # for k,v in algo_params.items():
    #     if k not in params_dict:
    #         params_dict[k] = v

    with open(os.path.join(results_dir, f"{policy}_{args.batch_size}_params.json"), "w") as file:
        json.dump(params_dict, file)


first_trial = args.first_trial
last_trial = args.first_trial + args.trials - 1

experiment_manager(
    problem=problem,
    obj_func=obj_func,
    algorithm=algo,
    performance_metrics=performance_metrics,
    input_dim=input_dim,
    noise_type="noiseless",
    noise_level=0.0,
    policy=args.policy,
    batch_size=args.batch_size,
    num_init_points=2 * (input_dim + 1),
    num_iter=args.max_iter,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=args.restart,
    save_data=args.save, # TODO
    edge_coords=edge_coords,
)

# to run
# python dijkstra_runner2.py -s --problem multimodal --policy ps --max_iter 100 --trials 10