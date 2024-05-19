import os
import sys
import torch
import numpy as np
import argparse
import json

from pymoo.problems import get_problem
from pymoo.indicators.hv import HV
from botorch.settings import debug
from botorch.test_functions.multi_objective import DTLZ1, ZDT1, ZDT2, DTLZ2, VehicleSafety, Penicillin


torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
debug._set_state(False)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
src_dir = "/".join(script_dir.split("/")[:-2]) # src directory is two levels up
sys.path.append(src_dir)

from src.bax.alg.multiobjective import PymooAlgorithm, HypervolumeAlgorithm, ScalarizedParetoSolver
from src.experiment_manager import experiment_manager
from src.performance_metrics import PymooHypervolume
from src.utils import compute_noise_std

parser = argparse.ArgumentParser()
parser.add_argument('--problem', type=str, default='zdt1')
parser.add_argument('--opt_mode', type=str, default='maximize')
parser.add_argument('--noise', type=float, default=0.0)
parser.add_argument('--policy', type=str, default='bax')
parser.add_argument('--trials', type=int, default=5)
parser.add_argument('--first_trial', type=int, default=1)
parser.add_argument('--n_dim', type=int, default=3)
parser.add_argument('--n_obj', type=int, default=2)
parser.add_argument('--n_gen', type=int, default=500)
parser.add_argument('--pop_size', type=int, default=5)
# parser.add_argument('--n_init', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--max_iter', type=int, default=100)
parser.add_argument('--algo_name', type=str, default='NSGA2')
parser.add_argument('--save', '-s', action='store_true', default=False)
parser.add_argument('--restart', '-r', action='store_true', default=False)
args = parser.parse_args()

# python multiobjective_runner.py -s --policy ps --problem dtlz2 --n_dim 10 --n_obj 3 --n_gen 500 --pop_size 40

n_dim = args.n_dim 
n_obj = args.n_obj
problem = args.problem + f"_{n_dim}d_{n_obj}obj"

if args.opt_mode == 'maximize':
    negate = True 
if args.problem == 'dtlz1':
    f = DTLZ1(
        dim=args.n_dim,
        num_objectives=args.n_obj,
        negate=negate, 
    )
    ref_val = -f._ref_val if negate else f._ref_val
    ref_point = np.array([ref_val] * args.n_obj)
    opt_value = None
elif args.problem == 'dtlz2':
    f = DTLZ2(
        dim=args.n_dim,
        num_objectives=args.n_obj,
        negate=negate, 
    )
    ref_val = -f._ref_val if negate else f._ref_val
    ref_point = np.array([ref_val] * args.n_obj)
    pymoo_problem = get_problem(
        "dtlz2",
        n_var=n_dim,
        n_obj=n_obj,
    )
    pymoo_pf = pymoo_problem.pareto_front()
    pymoo_ref_point = np.array([1.1] * args.n_obj)
    ind_pymoo = HV(ref_point=pymoo_ref_point)
    opt_value = ind_pymoo(pymoo_pf)
elif args.problem == 'vehiclesafety':
    vehiclesafety_func = VehicleSafety(negate=True)
    ref_point = vehiclesafety_func.ref_point.numpy()

    def f(X):
        X_unscaled = 2.0 * X + 1.0
        output = vehiclesafety_func(X_unscaled)
        return output
    
elif args.problem == 'zdt1':
    f = ZDT1(
        dim=args.n_dim,
        num_objectives=args.n_obj,
        negate=negate, 
    )
    ref_point = f.ref_point.numpy()
    opt_value = None
elif args.problem == "zdt2":
    f = ZDT2(
        dim=args.n_dim,
        num_objectives=2,
        negate=negate, 
    )
    ref_point = f.ref_point.numpy()
    pymoo_problem = get_problem(
        "zdt2",
        n_var=n_dim,
    ) # minimizing
elif args.problem == "penicillin":
    penicillin = Penicillin(negate=True)
    bounds = penicillin.bounds
    ref_point = penicillin.ref_point.numpy()
    def f(X):
        X_unscaled = torch.mul(X, torch.tensor([bounds[i][1] - bounds[i][0] for i in range(len(bounds))])) + torch.tensor([bounds[i][0] for i in range(len(bounds))])
        return penicillin(X_unscaled)
    opt_value = None

def obj_func(X):
    return f(X)


# algo_params = {
#     "n_dim": args.n_dim,
#     "n_obj": args.n_obj,
#     "n_gen": args.n_gen,
#     "pop_size": args.pop_size,
#     "n_offsprings": 10,
#     "opt_mode": args.opt_mode,
#     "ref_point": ref_point,
#     "output_size": 50,
#     "num_runs": 1,
# }
# algo = HypervolumeAlgorithm(algo_params)

algo_params = {
    "n_dim": args.n_dim,
    "n_obj": args.n_obj,
    "set_size": 2 ** args.n_obj,
    "num_restarts": args.n_dim,
    "raw_samples": 50 * args.n_dim,
    "batch_limit": args.n_dim,
    "init_batch_limit": 25 * args.n_dim,
    "ref_point": ref_point,
}
algo = ScalarizedParetoSolver(algo_params)

performance_metric_algo_params = {
    "n_dim": args.n_dim,
    "n_obj": args.n_obj,
    "set_size": 5 * (2 ** args.n_obj),
    "num_restarts": 5 * args.n_dim,
    "raw_samples": 100 * args.n_dim,
    "batch_limit": 5,
    "init_batch_limit": 25 * args.n_dim,
    "ref_point": ref_point,
}
performance_metric_algo = ScalarizedParetoSolver(performance_metric_algo_params)

performance_metrics = [
    PymooHypervolume(
        algo=performance_metric_algo,
        obj_func=obj_func,
        ref_point=ref_point,
        num_runs=1,
        opt_value=opt_value,
    )
]

if args.noise > 0:
    problem += f"_noise{args.noise}"
    noise_type = "noisy"
    bounds = torch.vstack([torch.zeros(args.n_dim), torch.ones(args.n_dim)])
    noise_levels = compute_noise_std(obj_func, args.noise, bounds=bounds)
else:
    noise_type = "noiseless"
    noise_levels = None

if args.save:
    results_dir = f"./results/{problem}"
    os.makedirs(results_dir, exist_ok=True)
    policy = args.policy
    params_dict = vars(args)
    for k,v in algo_params.items():
        if k not in params_dict and k != "ref_point":
            params_dict[k] = v

    with open(os.path.join(results_dir, f"{policy}_{args.batch_size}_params.json"), "w") as file:
        json.dump(params_dict, file)

first_trial = args.first_trial
last_trial = args.first_trial + args.trials - 1

experiment_manager(
    problem=f"{problem}",
    algorithm=algo,
    obj_func=obj_func,
    performance_metrics=performance_metrics,
    input_dim=n_dim,
    policy=args.policy,
    batch_size=args.batch_size,
    num_init_points=2 * (n_dim + 1),
    num_iter=args.max_iter,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=args.restart,
    save_data=args.save,
    bax_num_cand=1000 * n_dim,
    noise_type=noise_type,
    noise_level=noise_levels,
    # exe_path=2,
)


