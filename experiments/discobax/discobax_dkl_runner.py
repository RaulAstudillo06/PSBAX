import os
import sys
import torch
import json
import pandas as pd
import numpy as np
import argparse

from botorch.settings import debug
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
src_dir = "/".join(script_dir.split("/")[:-2]) # src directory is two levels up
sys.path.append(src_dir)

from src.bax.alg.discobax import SubsetSelect
from src.acquisition_functions.posterior_sampling import gen_posterior_sampling_batch
from src.acquisition_functions.bax_acquisition import BAXAcquisitionFunction
from src.performance_metrics import DiscreteTopKMetric, DiscreteDiscoBAXMetric
from src.experiment_manager import experiment_manager
from src.problems import DiscoBAXObjective

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
debug._set_state(False)

parser = argparse.ArgumentParser()
parser.add_argument('--policy', type=str, default='ps')
parser.add_argument('--problem_idx', type=int, default=3)
parser.add_argument('--use_random', default=False, action='store_true')
parser.add_argument('--do_pca', default=False, action='store_true')
parser.add_argument('--pca_dim', type=int, default=20)
parser.add_argument('--data_size', type=int, default=5000)
parser.add_argument('--trials', type=int, default=5)
parser.add_argument('--first_trial', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--max_iter', type=int, default=100)
parser.add_argument('--n_init', type=int, default=None)
parser.add_argument('--eta_budget', type=int, default=100)
parser.add_argument('--model_type', type=str, default="dkgp")
parser.add_argument('--check_GP_fit', type=bool, default=False)
parser.add_argument('--allow_reselect', type=bool, default=False) # CHANGE
parser.add_argument('--save', '-s', action='store_true', default=False)
parser.add_argument('--restart', '-r', action='store_true', default=False)
args = parser.parse_args()

# === To RUN === # 
# python discobax_dkl_runner.py -s --problem_idx 3 --max_iter 200 --data_size 1700 --n_init 100 --eta_budget 100 --policy ps --first_trial 1 --trials 5

# check gp fit
# python discobax_dkl_runner.py -s --problem_idx 4 --max_iter 10 --policy ps --data_size 1700 --n_init 100 --check_GP_fit 1 --trials 1 

data_path = f"{script_dir}/data/"

problem_lst = [
    "schmidt_2021_ifng",
    "schmidt_2021_il2",
    "zhuang_2019",
    "sanchez_2021_tau",
    "zhu_2021_sarscov2_host_factors",
]
problem = problem_lst[args.problem_idx]

# === Testing === #
TEST = False
if TEST:
    args.problem_idx = 3
    args.num_iter = 100
    args.do_pca = True
    args.pca_dim = 5
    args.data_size = 1700
    args.policy = "ps"
    args.restart = True
    problem = problem_lst[args.problem_idx]
# =============== #
if args.use_random:
    data_path = os.path.join(data_path, f"{problem}_random_{args.data_size}.csv")
else:
    data_path = os.path.join(data_path, f"{problem}_top_{args.data_size}.csv")
    # check if data_path exists
    os.path.exists(data_path)
df = pd.read_csv(data_path, index_col=0)
df_x = df.drop(columns=["y"])
df_y = df["y"]

if not args.do_pca:
    args.pca_dim = df_x.shape[1]

print(f'pca_dim: {args.pca_dim}, data_size: {args.data_size}')

def topk_indices(y: pd.Series, k):
    if isinstance(k, float):
        return list(y.sort_values(ascending=False).index[:int(k * y.shape[0])].values)
    elif isinstance(k, int):
        return list(y.sort_values(ascending=False).index[:k].values)
    

if args.do_pca or args.pca_dim != df_x.shape[1]:
    print("===== Doing PCA =====")
    pca_x = torch.tensor(df_x.values)
    pca_x = StandardScaler().fit_transform(pca_x)
    pca = PCA(n_components=args.pca_dim, random_state=0)
    pca_x = pca.fit_transform(pca_x)
    pca_df = pd.DataFrame(pca_x, index=df.index)
    pca_df["y"] = df_y
    df = pca_df
    df_x = df.drop(columns=["y"])
    df_y = df["y"]

df_x = (df_x - df_x.min()) / (df_x.max() - df_x.min())
df = df_x
df["y"] = df_y


obj_func = DiscoBAXObjective(
    problem, 
    df,
    noise_budget=args.eta_budget,
    noise_type="additive",
    idx_type="str",
    nonneg=False, # NOTE: not taking np.maximum(0, fx) 
)
obj_func.set_dict()

algo_params = {
    "name": "SubsetSelect",
    "k": 10,
}
algo = SubsetSelect(algo_params)

# == DO if not update_objective == #
update_objective = False
fn = f"{script_dir}/data/etas_seed0_size{args.data_size}.txt"
if os.path.exists(fn):
    etas = np.loadtxt(fn)
    if len(etas) == args.eta_budget:
        # obj_func.etas_lst = etas
        obj_func.etas = etas
        print("Loaded etas from file.")
obj_func.initialize(seed=0, verbose=True)
eta_arr = np.array(obj_func.etas) # (eta_budget, args.data_size)
if not os.path.exists(fn):
    try:
        np.savetxt(f"{script_dir}/data/etas_seed0_size{args.data_size}", eta_arr)
    except:
        print("Unable to save etas to data dir.")

algo.set_obj_func(obj_func)

performance_metrics = [
    DiscreteDiscoBAXMetric(
        name="DiscoBAXMetric", 
        algo=algo,
        obj_func=obj_func,
    )
]

for metric in performance_metrics:
    metric.set_algo(algo)

# == DO if not update_objective == #
# update_objective = True
    
problem = data_path.split("/")[-1].split(".")[0]
problem = "discobax" + "_" + problem
policy = args.policy + f"_model{args.model_type}" + f"_dim{args.pca_dim}"

if args.n_init is None:
    n_init = 2 * (args.pca_dim + 1)
else:
    n_init = args.n_init
if args.save:
    results_dir = f"{script_dir}/results/{problem}/{policy}_{args.batch_size}"
    os.makedirs(results_dir, exist_ok=True)
    params_dict = vars(args)
    for k,v in algo_params.items():
        if k not in params_dict:
            params_dict[k] = v

    params_dict["n_init"] = n_init
    with open(os.path.join(results_dir, f"{policy}_params.json"), "w") as file:
        json.dump(params_dict, file)

first_trial = args.first_trial
last_trial = args.first_trial + args.trials - 1

# NOTE: Change DKGP hyperparameters here!
model_architecture = [160, 40, 10, 5] # Excluding input_dim and output_dim
epochs = 5000
experiment_manager(
    problem=problem,
    algorithm=algo,
    obj_func=obj_func,
    performance_metrics=performance_metrics,
    input_dim=args.pca_dim,
    noise_type="noiseless",
    noise_level=0.0,
    policy=policy,
    batch_size=args.batch_size,
    num_init_points=n_init,
    num_iter=args.max_iter,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=args.restart,
    data_df=df,
    save_data=args.save,
    check_GP_fit=args.check_GP_fit,
    update_objective=update_objective,
    model_type=args.model_type,
    epochs=epochs,
    architecture=model_architecture,
    allow_reselect=args.allow_reselect, 
)

