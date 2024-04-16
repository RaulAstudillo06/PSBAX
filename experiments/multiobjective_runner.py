import os
import sys
import torch
import numpy as np
import argparse


from botorch.settings import debug
from botorch.test_functions.multi_objective import DTLZ1, ZDT1

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
debug._set_state(False)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
print(script_dir[:-12])
sys.path.append(script_dir[:-12])

from src.bax.alg.multiobjective import PymooAlgorithm
from src.experiment_manager import experiment_manager
from src.performance_metrics import PymooHypervolume

parser = argparse.ArgumentParser()
parser.add_argument('--policy', type=str, default='bax')
parser.add_argument('--trials', type=int, default=5)
parser.add_argument('--n_dim', type=int, default=5)
parser.add_argument('--n_obj', type=int, default=2)
parser.add_argument('--max_iter', type=int, default=100)
parser.add_argument('--algo_name', type=str, default='NSGA2')
parser.add_argument('--save', '-s', action='store_true', default=False)
parser.add_argument('--restart', '-r', action='store_true', default=False)
args = parser.parse_args()

def obj_func(X):
    f = ZDT1(
        dim=args.n_dim,
        num_objectives=args.n_obj,
        negate=False, # minimize
    )
    return f(X)

first_trial = 1
last_trial = args.trials

algo_params = {
    "name": args.algo_name,
    "n_dim": args.n_dim,
    "n_obj": args.n_obj,
    "n_gen": 10,
    "pop_size": 10,
    "n_offsprings": 10,
}
algo = PymooAlgorithm(algo_params)

performance_metrics = [
    PymooHypervolume(
        algo=algo.get_copy(),
        obj_func=obj_func,
        ref_point=np.array([11.0, 11.0]),
    )
]

n_dim = args.n_dim
experiment_manager(
    problem="zdt1" + f"_{n_dim}d",
    algorithm=algo,
    obj_func=obj_func,
    performance_metrics=performance_metrics,
    input_dim=n_dim,
    policy=args.policy,
    batch_size=1,
    num_init_points=10,
    num_iter=args.max_iter,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=args.restart,
    save_data=args.save,
)


