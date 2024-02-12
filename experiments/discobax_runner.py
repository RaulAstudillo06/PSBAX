import os
import sys
import torch
import pandas as pd
import numpy as np
import argparse

from botorch.settings import debug

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
print(script_dir[:-12])
sys.path.append(script_dir[:-12])

from src.bax.alg.subset_select import SubsetSelect
from src.acquisition_functions.posterior_sampling import gen_posterior_sampling_batch
from src.acquisition_functions.bax_acquisition import BAXAcquisitionFunction
from src.performance_metrics import DiscreteTopKMetric
from src.experiment_manager import experiment_manager

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
debug._set_state(False)

data_path = "experiments/data/discobax"
problem_lst = [
    "test_schmidt_2021_ifng",
    "schmidt_2021_ifng",
    "schmidt_2021_il2",
    "zhuang_2019",
    "sanchez_2021_tau",
    "zhu_2021_sarscov2_host_factors",
]
problem = problem_lst[0]

# load data
df = pd.read_csv(os.path.join(data_path, problem + ".csv"), index_col=0)
df_x = df.drop(columns=["y"])
df_y = df["y"]

parser = argparse.ArgumentParser()
parser.add_argument('--policy', type=str, default='bax')
parser.add_argument('--trials', type=int, default=5)
parser.add_argument('--save', type=bool, default=True)

args = parser.parse_args()
first_trial = 1
last_trial = args.trials

def topk_indices(y: pd.Series, k: float) -> np.ndarray:
    return y.sort_values(ascending=False).index[:int(k * y.shape[0])].values

algo_params = {
    "name": "SubsetSelect",
    "k": 10,
    "eta_type": "additive",
    "budget": 100,
    "df": df,
}

# scale everything in df except for y to [0, 1]
df_x = (df_x - df_x.min()) / (df_x.max() - df_x.min())
df = df_x
df["y"] = df_y

algo = SubsetSelect(algo_params)

performance_metrics = [
    DiscreteTopKMetric("DiscreteJaccardSimilarity", algo, df),
    DiscreteTopKMetric("DiscreteNormDifference", algo, df),
]

experiment_manager(
    problem=problem,
    algorithm=algo,
    performance_metrics=performance_metrics,
    input_dim=20,
    noise_type="noiseless",
    noise_level=0.0,
    policy=args.policy,
    batch_size=1,
    num_init_points=10,
    num_iter=10,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=False,
    data_df=df,
    save_data=args.save,
    discrete=True,
)
