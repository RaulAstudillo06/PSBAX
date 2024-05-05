
#!/usr/bin/env python3
from typing import Callable

import os
import sys
import torch
import json
import numpy as np
import argparse
from botorch.acquisition.analytic import PosteriorMean
from botorch.settings import debug
from botorch.test_functions.synthetic import Hartmann
from torch import Tensor

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
debug._set_state(False)
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
src_dir = "/".join(script_dir.split("/")[:-2]) # src directory is two levels up
sys.path.append(src_dir)

from src.bax.alg.algorithms import TopK
from src.experiment_manager import experiment_manager
from src.performance_metrics import JaccardSimilarity, NormDifference
from src.bax.util.domain_util import unif_random_sample_domain
from src.bax.util.graph import jaccard_similarity
from src.utils import seed_torch


# if len(sys.argv) == 3:
#     first_trial = int(sys.argv[1])
#     last_trial = int(sys.argv[2])
# elif len(sys.argv) == 2:
#     first_trial = int(sys.argv[1])
#     last_trial = int(sys.argv[1])
# else:
#     first_trial = 1
#     last_trial = 5

parser = argparse.ArgumentParser()
parser.add_argument('--policy', type=str, default='ps')
parser.add_argument('--function', type=str, default='himmelblau')
parser.add_argument('--dim', type=int, default=2)
parser.add_argument('--first_trial', type=int, default=1)
parser.add_argument('--trials', type=int, default=5)
parser.add_argument('--max_iter', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=3)
parser.add_argument('--len_path', type=int, default=150)
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--save', '-s', action='store_true', default=False)
parser.add_argument('--restart', '-r', action='store_true', default=False)
args = parser.parse_args()

# === To RUN === # 
# python topk_runner.py -s --trials 30 --policy ps

if args.function == 'himmelblau':
    input_dim = 2
    domain = [[-6, 6]] * input_dim # NOTE: himmelblau domain
else:
    input_dim = args.dim
    domain = [[-10, 10]] * input_dim # NOTE: original domain

rescaled_domain = [[0, 1]] * input_dim
len_path = args.len_path
k = args.k


def himmelblau(X: Tensor, minimize=False) -> Tensor:
    a = X[:, 0]
    b = X[:, 1]
    result = (a ** 2 + b - 11) ** 2 + (a + b ** 2 - 7) ** 2
    if not minimize:
        return -result
    else:
        return result


def obj_func(X, domain=domain):
    '''
    Args:
        X: (1, n_dim)
    '''
    # check if x is a torch tensor
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X)
    # rescale X from 0, 1 to domain
    X_rescaled = X.clone()
    X_rescaled = torch.mul(X, torch.tensor([domain[i][1] - domain[i][0] for i in range(len(domain))])) + torch.tensor([domain[i][0] for i in range(len(domain))])
    
    if args.function == 'original':
        f_0 = lambda x:  2 * torch.abs(x) * torch.sin(x)
        return torch.sum(torch.stack([f_0(x) for x in X_rescaled]), dim=-1)
    elif args.function == 'himmelblau':
        f_0 = himmelblau
        return f_0(X_rescaled)

seed_torch(1234) # NOTE: fix seed for generating x_path

x_path = unif_random_sample_domain(rescaled_domain, len_path) # NOTE: Action set

if args.function == 'himmelblau':
    himmelblau_opt = np.array(
        [
            [3, 2],
            [-2.805118, 3.283186],
            [-3.779310, -3.283186],
            [3.584458, -1.848126],
        ]
    )
    himmelblau_opt = (himmelblau_opt - np.array(domain)[:, 0]) / (np.array(domain)[:, 1] - np.array(domain)[:, 0])
    x_path = np.concatenate([x_path, np.array(himmelblau_opt)], axis=0)

algo = TopK({"x_path": x_path, "k": k}, verbose=False)

def output_dist_fn_norm(a, b):
    """Output dist_fn based on concatenated vector norm."""
    a_list = []
    list(map(a_list.extend, a.x))
    a_list.extend(a.y)
    a_arr = np.array(a_list)

    b_list = []
    list(map(b_list.extend, b.x))
    b_list.extend(b.y)
    b_arr = np.array(b_list)

    return np.linalg.norm(a_arr - b_arr)

def output_dist_fn_jaccard(a, b):
    """Output dist_fn based on Jaccard similarity."""
    a_x_tup = [tuple(x) for x in a.x]
    b_x_tup = [tuple(x) for x in b.x]
    jac_sim = jaccard_similarity(a_x_tup, b_x_tup)
    dist = 1 - jac_sim
    return dist

# def metric_jacc(obj_func: Callable, posterior_mean_func: PosteriorMean):
#     '''
#     TODO: metrics are sharing the same algo_mf, algo_gt?
#     '''
#     algo_gt = TopK({"x_path": x_path, "k": k, "name": "groundtruth"})
#     _, output_gt = algo_gt.run_algorithm_on_f(obj_func) # TODO: this should be saved as a common attribute
#     algo_mf = TopK({"x_path": x_path, "k": k}, verbose=False)
#     _, output_mf = algo_mf.run_algorithm_on_f(posterior_mean_func)
#     return output_dist_fn_jaccard(output_mf, output_gt)

# def metric_norm(obj_func: Callable, posterior_mean_func: PosteriorMean):
#     algo_gt = TopK({"x_path": x_path, "k": k, "name": "groundtruth"})
#     _, output_gt = algo_gt.run_algorithm_on_f(obj_func)
#     algo_mf = TopK({"x_path": x_path, "k": k}, verbose=False)
#     _, output_mf = algo_mf.run_algorithm_on_f(posterior_mean_func)
#     return output_dist_fn_norm(output_mf, output_gt)

# performance_metrics = {
#     "Jaccard": metric_jacc,
#     "Norm": metric_norm,
# }

performance_metrics = [
    JaccardSimilarity(algo, obj_func),
    NormDifference(algo, obj_func),
]

problem = f"topk_{args.function}"

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
    save_data=args.save,
)
