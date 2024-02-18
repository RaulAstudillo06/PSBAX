import os
import sys
import torch
import pandas as pd
import numpy as np
import argparse

from botorch.settings import debug
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
print(script_dir[:-12])
sys.path.append(script_dir[:-12])

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
parser.add_argument('--problem_idx', type=int, default=0)
parser.add_argument('--do_pca', type=bool, default=False)
parser.add_argument('--pca_dim', type=int, default=20)
parser.add_argument('--data_size', type=int, default=10000)
parser.add_argument('--first_trial', type=int, default=1)
parser.add_argument('--last_trial', type=int, default=10)
parser.add_argument('--num_iter', type=int, default=100)
parser.add_argument('--n_init', type=int, default=100)
parser.add_argument('--eta_budget', type=int, default=100)
parser.add_argument('--save', '-s', action='store_true', default=False)
args = parser.parse_args()

# === To RUN === # 
# python discobax_runner.py -s --problem_idx 0 --num_iter 100 --do_pca False --eta_budget 100 --policy bax
# python discobax_runner.py -s --policy ps_test

data_path = "./data/discobax"
if "experiment" not in os.getcwd():
    data_path = "experiments/data/discobax"
problem_lst = [
    "test_schmidt_2021_ifng",
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
    args.first_trial = 1
    args.last_trial = 1
    args.num_iter = 10
    args.save = False
    args.do_pca = False
    problem = problem_lst[0]
    check_GP_fit = True
else:
    check_GP_fit = False
# =============== #

df = pd.read_csv(os.path.join(data_path, problem + ".csv"), index_col=0)
df_x = df.drop(columns=["y"])
df_y = df["y"]

def topk_indices(y: pd.Series, k):
    if isinstance(k, float):
        return list(y.sort_values(ascending=False).index[:int(k * y.shape[0])].values)
    elif isinstance(k, int):
        return list(y.sort_values(ascending=False).index[:k].values)

if args.do_pca or args.pca_dim != df_x.shape[1]:
    keep_idx = topk_indices(df_y, args.data_size)
    keep_df = df.loc[keep_idx]
    keep_x = torch.tensor(keep_df.drop(columns=["y"]).values)
    keep_x = StandardScaler().fit_transform(keep_x)
    pca = PCA(n_components=args.pca_dim)
    keep_x = pca.fit_transform(keep_x)
    keep_df = pd.DataFrame(keep_x, index=keep_df.index)
    keep_df["y"] = df_y.loc[keep_idx]
    df = keep_df
    df_x = df.drop(columns=["y"])
    df_y = df["y"]


df_x = (df_x - df_x.min()) / (df_x.max() - df_x.min())
df = df_x
df["y"] = df_y

obj_func = DiscoBAXObjective(
    problem, 
    df,
    noise_bugget=args.eta_budget,
    noise_type="additive",
    idx_type="str",
    nonneg=False, # NOTE: not taking np.maximum(0, fx) 
)

algo_params = {
    "name": "SubsetSelect",
    "k": 10,
}
algo = SubsetSelect(algo_params)

performance_metrics = [
    DiscreteDiscoBAXMetric("DiscoBAXMetric")
]


experiment_manager(
    problem=problem,
    algorithm=algo,
    obj_func=obj_func,
    performance_metrics=performance_metrics,
    input_dim=args.pca_dim,
    noise_type="noiseless",
    noise_level=0.0,
    policy=args.policy,
    batch_size=1,
    num_init_points=args.n_init,
    num_iter=args.num_iter,
    first_trial=args.first_trial,
    last_trial=args.last_trial,
    restart=False,
    data_df=df,
    save_data=args.save,
    discrete=True,
    check_GP_fit=check_GP_fit,
)

