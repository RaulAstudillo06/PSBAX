
#!/usr/bin/env python3
from typing import Callable

import os
import sys
import torch
import json
import math
import numpy as np
import argparse
from botorch.acquisition.analytic import PosteriorMean
from botorch.settings import debug
from botorch.test_functions.synthetic import Rastrigin, Levy
from torch import Tensor

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
debug._set_state(False)
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
src_dir = "/".join(script_dir.split("/")[:-2]) # src directory is two levels up
sys.path.append(src_dir)

from src.bax.alg.algorithms import TopK
from src.experiment_manager import experiment_manager
from src.performance_metrics import JaccardSimilarity, NormDifference, SumOfObjectiveValues
from src.bax.util.domain_util import unif_random_sample_domain
from src.bax.util.graph import jaccard_similarity
from src.utils import seed_torch, generate_random_points, get_mesh, reshape_mesh

parser = argparse.ArgumentParser()
parser.add_argument('--policy', type=str, default='random')
parser.add_argument('--function', type=str, default='original')
parser.add_argument('--dim', type=int, default=3)
parser.add_argument('--first_trial', type=int, default=1)
parser.add_argument('--trials', type=int, default=5)
parser.add_argument('--max_iter', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=3)
parser.add_argument('--len_path', type=int, default=150)
parser.add_argument('--use_mesh', action='store_true', default=False)
parser.add_argument('--steps', type=int, default=15)
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--save', '-s', action='store_true', default=False)
parser.add_argument('--restart', '-r', action='store_true', default=False)
args = parser.parse_args()

# === To RUN === # 
# python topk_runner.py -s --trials 30 --policy ps


if args.function == 'himmelblau':
    input_dim = 2
    domain = [[-6, 6]] * input_dim # NOTE: himmelblau domain
elif args.function == 'original':
    input_dim = args.dim
    domain = [[-10, 10]] * input_dim # NOTE: original domain
elif args.function == 'rastrigin':
    input_dim = args.dim
    domain = [[-5.12, 5.12]] * input_dim # NOTE: original domain
elif args.function == 'levy':
    input_dim = args.dim
    domain = [[-10.0, 10.0]] * input_dim # NOTE: original domain

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

if args.function == 'himmelblau' or args.function == 'original':
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
        elif args.function == 'rastrigin':
            return f_0(X_rescaled)
elif args.function == 'rastrigin':
    rastrigin = Rastrigin(dim=input_dim, negate=True)

    def obj_func(X, domain=domain):
        return rastrigin(torch.tensor(10.24 * X - 5.12))
elif args.function == 'levy':
    levy = Levy(dim=input_dim, negate=True)

    def obj_func(X, domain=domain):
        return levy(torch.tensor((20.0 * X) - 10.0))

# seed_torch(1234) # NOTE: fix seed for generating x_path

args.use_mesh = True
if args.use_mesh:
    xx = get_mesh(input_dim, args.steps)
    x_path = reshape_mesh(xx).numpy()
    len_path = x_path.shape[0]
else: 
    # x_path = unif_random_sample_domain(rescaled_domain, len_path) # NOTE: Action set
    x_path = generate_random_points(num_points=len_path, input_dim=input_dim, seed=1234).numpy()
    # np.save(f"{script_dir}/data/{args.function[:3]}_x_np.npy", x_path)

if args.function == 'himmelblau':
#     x_path = np.load(f"{script_dir}/data/him_x_np.npy")
    himmelblau_opt = np.array(
        [
            [3.0, 2.0],
            [-2.805118, 3.131312],
            [-3.779310, -3.283186],
            [3.584458, -1.848126],
        ]
    )
    himmelblau_opt = (himmelblau_opt - np.array(domain)[:, 0]) / (np.array(domain)[:, 1] - np.array(domain)[:, 0])
    x_path = np.concatenate([x_path, np.array(himmelblau_opt)], axis=0)
# elif args.function == 'original':
#     x_path = np.load(f"{script_dir}/data/ori_x_np.npy")

x_path = [list(x) for x in x_path]
algo = TopK({"x_path": x_path, "k": k}, verbose=False)

algo_metric = algo.get_copy()
performance_metrics = [
    JaccardSimilarity(algo_metric, obj_func),
    # NormDifference(algo_metric, obj_func),
    SumOfObjectiveValues(algo_metric, obj_func),
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
    x_batch=np.array(x_path),
)
