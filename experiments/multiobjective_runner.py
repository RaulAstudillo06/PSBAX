import os
import sys
import torch
import numpy as np
import argparse
import json


from botorch.settings import debug
from botorch.test_functions.multi_objective import DTLZ1, ZDT1, DTLZ2

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
parser.add_argument('--problem', type=str, default='dtlz2')
parser.add_argument('--policy', type=str, default='ps')
parser.add_argument('--trials', type=int, default=5)
parser.add_argument('--n_dim', type=int, default=10)
parser.add_argument('--n_obj', type=int, default=3)
parser.add_argument('--n_gen', type=int, default=500)
parser.add_argument('--pop_size', type=int, default=40)
parser.add_argument('--n_init', type=int, default=10)
parser.add_argument('--max_iter', type=int, default=100)
parser.add_argument('--algo_name', type=str, default='NSGA2')
parser.add_argument('--save', '-s', action='store_true', default=False)
parser.add_argument('--restart', '-r', action='store_true', default=False)
args = parser.parse_args()

# python multiobjective_runner.py -s --policy ps --problem dtlz2 --n_dim 10 --n_obj 3 --n_gen 500 --pop_size 40

if args.problem == 'dtlz1':
    f = DTLZ1(
        dim=args.n_dim,
        num_objectives=args.n_obj,
        negate=False, # minimize
    )
elif args.problem == 'dtlz2':
    f = DTLZ2(
        dim=args.n_dim,
        num_objectives=args.n_obj,
        negate=False, # minimize
    )
elif args.problem == 'zdt1':
    f = ZDT1(
        dim=args.n_dim,
        num_objectives=args.n_obj,
        negate=False, # minimize
    )

def obj_func(X):
    return f(X)

first_trial = 1
last_trial = args.trials

algo_params = {
    "name": args.algo_name,
    "n_dim": args.n_dim,
    "n_obj": args.n_obj,
    "n_gen": args.n_gen,
    "pop_size": args.pop_size,
    "n_offsprings": 10,
}
algo = PymooAlgorithm(algo_params)

ref_val = f._ref_val
# ref_points = {
#     "zdt1": np.array([1.2] * args.n_obj),
#     "dtlz1": np.array([400.0] * args.n_obj),
#     "dtlz2": np.array([1.1] * args.n_obj),
# }

performance_metrics = [
    PymooHypervolume(
        algo=algo.get_copy(),
        obj_func=obj_func,
        ref_point=np.array([ref_val] * args.n_obj),
        num_runs=5,
    )
]


n_dim = args.n_dim 
n_obj = args.n_obj
problem = args.problem + f"_{n_dim}d_{n_obj}obj" + "_test"

if args.save:
    results_dir = f"./results/{problem}"
    os.makedirs(results_dir, exist_ok=True)
    policy = args.policy
    params_dict = vars(args)
    for k,v in algo_params.items():
        if k not in params_dict:
            params_dict[k] = v

    with open(os.path.join(results_dir, f"{policy}_params.json"), "w") as file:
        json.dump(params_dict, file)

experiment_manager(
    problem=f"{problem}",
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
    bax_num_cand=10000,
)


