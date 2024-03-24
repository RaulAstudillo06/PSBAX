from typing import Callable

import os
import sys
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
print(script_dir[:-12])
sys.path.append(script_dir[:-12])

from src.bax.alg.dijkstra import Dijkstra
from src.bax.util.domain_util import unif_random_sample_domain
from src.bax.util.graph import make_grid, edges_of_path, positions_of_path, area_of_polygons

from src.experiment_manager import experiment_manager
from src.performance_metrics import compute_obj_val_at_max_post_mean, ShortestPathCost, ShortestPathArea

input_dim = 2
grid_size = 10
x1_lims = (-2, 2)
x2_lims = (-1, 4)

rescaled_x1_lims = (0, 1)
rescaled_x2_lims = (0, 1)
positions, vertices, edges = make_grid(grid_size, rescaled_x1_lims, rescaled_x2_lims)
start, goal = vertices[-grid_size], vertices[-1]
policy = "ps"
# if len(sys.argv) == 3:
#     first_trial = int(sys.argv[1])
#     last_trial = int(sys.argv[2])
# elif len(sys.argv) == 2:
#     first_trial = int(sys.argv[1])
#     last_trial = int(sys.argv[1])
# else:
#     first_trial = 1
#     last_trial = 5

# use argparse
parser = argparse.ArgumentParser()
parser.add_argument('--policy', type=str, default='ps')
parser.add_argument('--trials', type=int, default=5)
parser.add_argument('--save', '-s', action='store_true', default=False)

# ====== To run ======
# python dijkstra_runner.py -s --policy ps --trials 5

args = parser.parse_args()
first_trial = 1
last_trial = args.trials

# Set function
def rosenbrock(x, y, a=1, b=100):
    # NOTE rescaled to improve numerics
    # NOTE min cost path: 1.0527267184880365
    return 1e-2 * ((a - x)**2 + b * (y - x**2)**2)

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

def obj_func(X, problem="rosenbrock", rescaled=True, minimize=True, bounds=[x1_lims, x2_lims], latent_f=True):
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

    if problem == "rosenbrock":
        f = Rosenbrock(dim=input_dim)
    else:
        raise NotImplementedError
    
    f_X = f.evaluate_true(X_rescaled)
    if rescaled:
        f_X = 1e-2 * f_X
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

# def metric_cost(obj_func: Callable, posterior_mean_func: PosteriorMean):
#     algo_mf = Dijkstra(algo_params, vertices, start, goal)
#     _, output_mf = algo_mf.run_algorithm_on_f(posterior_mean_func)
#     return output_mf[0]

# def metric_area(obj_func: Callable, posterior_mean_func: PosteriorMean):
#     algo_gt = Dijkstra(algo_params, vertices, start, goal)
#     _, true_output = algo_gt.run_algorithm_on_f(obj_func)
#     algo_mf = Dijkstra(algo_params, vertices, start, goal)
#     _, output_mf = algo_mf.run_algorithm_on_f(posterior_mean_func)
#     return area_of_polygons(true_output[1], output_mf[1])

# performance_metrics = {
#     'Cost': metric_cost,
#     'Area': metric_area
# }

performance_metrics = [
    ShortestPathCost(algo),
    ShortestPathArea(algo, obj_func),
]

experiment_manager(
    problem="dijkstra",
    obj_func=obj_func,
    algorithm=algo,
    performance_metrics=performance_metrics,
    input_dim=input_dim,
    noise_type="noiseless",
    noise_level=0.0,
    policy=args.policy,
    batch_size=1,
    num_init_points=2 * (input_dim + 1),
    num_iter=100,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=False,
    save_data=args.save, # TODO
)
