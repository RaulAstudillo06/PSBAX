
#!/usr/bin/env python3
from typing import Callable

import os
import sys
import torch
import numpy as np
from botorch.acquisition.analytic import PosteriorMean
from botorch.settings import debug
from botorch.test_functions.synthetic import Hartmann
from torch import Tensor

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
debug._set_state(False)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
print(script_dir[:-12])
sys.path.append(script_dir[:-12])

from src.bax.alg.algorithms import TopK
from src.experiment_manager import experiment_manager
from src.performance_metrics import compute_obj_val_at_max_post_mean
from src.bax.util.domain_util import unif_random_sample_domain
from src.bax.util.graph import jaccard_similarity

input_dim = 2
domain = [[-10, 10]] * input_dim
len_path = 150
k = 10
policy = "ps"
if len(sys.argv) == 3:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[2])
elif len(sys.argv) == 2:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[1])
else:
    first_trial = 1
    last_trial = 5


def obj_func(X):
    '''
    Args:
        X: (1, n_dim)
    '''
    # check if x is a torch tensor
    if not isinstance(X, torch.Tensor):
        print(X)
        X = torch.tensor(X)
    f_0 = lambda x:  2 * torch.abs(x) * torch.sin(x)
    return torch.sum(torch.stack([f_0(x) for x in X]), dim=-1)

x_path = unif_random_sample_domain(domain, len_path)
algo = TopK({"x_path": x_path, "k": k}, verbose=False)

algo_gt = TopK({"x_path": x_path, "k": k, "name": "groundtruth"})
exepath_gt, output_gt = algo_gt.run_algorithm_on_f(obj_func)

def algo_exe(obj_func: Callable) -> Tensor:
    '''Execute TopK algorithm on obj_func.
    '''
    _, output = algo.run_algorithm_on_f(obj_func)
    print(output)
    return output.x

metric_jacc = lambda x: algo.output_dist_fn_jaccard(x, output_gt)
metric_norm = lambda x: algo.output_dist_fn_norm(x, output_gt)

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

def metric_jacc(obj_func: Callable, posterior_mean_func: PosteriorMean):
    '''
    TODO: metrics are sharing the same algo_mf, algo_gt?
    '''
    algo_gt = TopK({"x_path": x_path, "k": k, "name": "groundtruth"})
    _, output_gt = algo_gt.run_algorithm_on_f(obj_func)
    algo_mf = TopK({"x_path": x_path, "k": k}, verbose=False)
    # FIXME: posterior_mean_func cannot be directly input into algo
    _, output_mf = algo_mf.run_algorithm_on_f(posterior_mean_func)
    return output_dist_fn_jaccard(output_mf, output_gt)

def metric_norm(obj_func: Callable, posterior_mean_func: PosteriorMean):
    algo_gt = TopK({"x_path": x_path, "k": k, "name": "groundtruth"})
    _, output_gt = algo_gt.run_algorithm_on_f(obj_func)
    algo_mf = TopK({"x_path": x_path, "k": k}, verbose=False)
    # FIXME: posterior_mean_func cannot be directly input into algo
    _, output_mf = algo_mf.run_algorithm_on_f(posterior_mean_func)
    return output_dist_fn_norm(output_mf, output_gt)


performance_metrics = {
    "Jaccard": metric_jacc,
    "Norm": metric_norm,

}

experiment_manager(
    problem="topk",
    obj_func=obj_func,
    algo_exe=algo_exe,
    performance_metrics=performance_metrics,
    input_dim=input_dim,
    noise_type="noiseless",
    noise_level=0.0,
    policy=policy,
    batch_size=1,
    num_init_points=2 * (input_dim + 1),
    num_iter=100,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=False,
)
