#%%
import numpy as np
import torch
import os
import sys
import pandas as pd
import argparse
import json

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
# debug._set_state(False)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
# cwd = os.getcwd()
# script_dir = cwd
src_dir = "/".join(script_dir.split("/")[:-2]) # src directory is two levels up
sys.path.append(src_dir)

from src.bax.alg.topk import TopKTorch
from src.experiment_manager import experiment_manager
from src.performance_metrics import JaccardSimilarity, NormDifference, SumOfObjectiveValues
from src.problems import GB1onehot

parser = argparse.ArgumentParser()
parser.add_argument('--policy', type=str, default='bax')
parser.add_argument('--first_trial', type=int, default=1)
parser.add_argument('--trials', type=int, default=5)
parser.add_argument('--max_iter', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=3)
parser.add_argument('--model_type', type=str, default='dkgp')
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--k', type=int, default=20)
parser.add_argument('--save', '-s', action='store_true', default=False)
parser.add_argument('--restart', '-r', action='store_true', default=False)
args = parser.parse_args()

# to run:
# python gb1_runner.py -s --policy ps --first_trial 1 --trials 5 --max_iter 100 --batch_size 5 --model_type dkgp --epochs 10000 --k 10

input_dim = 80

DATA_DIR = os.path.join(script_dir, 'data')

# generate random indices for testing
# test_data_size = 10000
# test_indices = np.random.choice(149361, test_data_size, replace=False)
# torch.save(torch.tensor(test_indices), os.path.join(DATA_DIR, 'gb1_test_indices.npy'))

#%%

obj_func = GB1onehot(
    DATA_DIR,
    noise_std=None,
    negate=False, # maximize fitness
    data_size=None,
)

test_indices = torch.load(os.path.join(DATA_DIR, 'gb1_test_indices.npy'))
obj_func.update_data(test_indices)
X = obj_func.X


# x_path = [list(x) for x in X.numpy()]
rescaled_domain = [[0.0, 1.0]] * input_dim
k = args.k
#%%

# df = pd.read_csv(os.path.join(DATA_DIR, 'GB1_fitness.csv'))
# X = torch.load(os.path.join(DATA_DIR, 'GB1_onehot_x_bool.pt')).to(torch.float64)
# data_size = 10000
# # choose random data_size samples
# idx = np.random.choice(len(X), data_size, replace=False)
# X = X[idx]
# Y = torch.from_numpy(df['fit'].values)
# input_dim = 80
# x_to_y = {tuple(x): y for x, y in zip(X.tolist(), Y.tolist())}
# rescaled_domain = [[0.0, 1.0]] * input_dim
# k = args.k
# # def obj_func(X):
# #     if not isinstance(X, torch.Tensor):
# #         X = torch.tensor(X)
# #     y = []
# #     for x in X:
# #         y.append(x_to_y[tuple(x.tolist())])
# #     return torch.tensor(y)

# x_path = [list(x) for x in X.numpy()]


algo = TopKTorch({"x_path": X, "k": k}, verbose=False)

algo_metric = algo.get_copy()
performance_metrics = [
    JaccardSimilarity(algo_metric, obj_func),
    # NormDifference(algo_metric, obj_func),
    SumOfObjectiveValues(algo_metric, obj_func),
]

problem = f"topk_gb1_test"

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

# NOTE: Change DKGP hyperparameters here!
model_architecture = [128, 64, 32] # Excluding input_dim and output_dim

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
    x_batch=X,
    model_type=args.model_type,
    architecture=model_architecture,
    epochs=args.epochs,
    check_GP_fit=True,
)
