
import os
import sys
import argparse

# set environment variable CUBLAS_WORKSPACE_CONFIG=:16:8
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(script_dir[:script_dir.index("discobax")])

from src.utils import seed_torch
from src.discobax.apps import genedisco_loop

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--acq_func', type=str, default='ps') # "discobax"
parser.add_argument('--max_iter', type=int, default=10)
parser.add_argument('--num_init', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--bax_noise', type=str, default='additive')
parser.add_argument('--eta_budget', type=int, default=10)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--save', '-s', action='store_true', default=False)
args = parser.parse_args()

# === To RUN === # 
# python discobax_bmlp_runner.py -s --seed 1 --acq_func ps --max_iter 100 --batch_size 1 --bax_noise additive

DATASET_NAMES = [
    "schmidt_2021_ifng",
    "schmidt_2021_il2",
    "zhuang_2019_nk",
    "sanchez_2021_neurons_tau",
    "zhu_2021_sarscov2_host_factors",
]

cache_directory = "src/discobax/data"
output_directory = "discobax-scripts/output"
dataset_name = "sanchez_2021_neurons_tau"
acquisition_function_name = args.acq_func
discobax_metric_file = os.path.join(output_directory, "metric.txt")
acquisition_batch_size = args.batch_size
bax_noise_type = args.bax_noise
num_active_learning_cycles = args.max_iter

seed_torch(args.seed)

loop = genedisco_loop.GeneDiscoLoop(
    cache_directory=cache_directory,
    output_directory=output_directory,
    dataset_name=dataset_name,
    acquisition_function_name=acquisition_function_name,
    acquisition_batch_size=acquisition_batch_size,
    num_active_learning_cycles=num_active_learning_cycles,
    bax_noise_type=bax_noise_type,
    eta_budget=args.eta_budget,
    seed=args.seed,
    bax_subset_select_subset_size=5,
    bax_num_samples_EIG=20,
    bax_num_samples_entropy=20,
)

loop.train_model()