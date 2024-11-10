import os
import sys
import torch
import json
import numpy as np
import subprocess

from botorch.test_functions.synthetic import Rosenbrock
from botorch.settings import debug

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
debug._set_state(False)
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
src_dir = "/".join(script_dir.split("/")[:-1]) # src directory is one level up
sys.path.append(src_dir)

from src.algorithms.topk import TopK
from src.experiment_manager import experiment_manager
from src.performance_metrics import JaccardSimilarity, NormDifference, SumOfObjectiveValues
from src.utils import generate_random_points


args = {
    "dim": 3,
    "len_path": 100,
    "k": 4,
    "batch_size": 1,
    "max_iter": 30,
}
function = "rosenbrock"
input_dim = args["dim"]
len_path = args["len_path"]
k = args["k"]
batch_size = args["batch_size"]
max_iter = args["max_iter"]
policies = ["random", "bax", "ps"]

domain = [[-2.0, 2.0]] * input_dim # NOTE: original domain
rescaled_domain = [[0.0, 1.0]] * input_dim

rosenbrock = Rosenbrock(dim=input_dim, negate=True)

def obj_func(X, domain=domain):
    if isinstance(X, np.ndarray):
        X = torch.tensor(X)
    return rosenbrock(X)

problem = f"topk_{function}"
results_dir = f"{script_dir}/results/{problem}"
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, f"params.json"), "w") as file:
    json.dump(args, file)

x_path = generate_random_points(num_points=len_path, input_dim=input_dim, seed=1234).numpy()

for policy in policies:
    algo = TopK({"x_path": x_path, "k": k}, verbose=False)
    algo_metric = algo.get_copy()
    performance_metrics = [
        JaccardSimilarity(algo_metric, obj_func),
        SumOfObjectiveValues(algo_metric, obj_func),
    ]

    experiment_manager(
        problem=problem,
        obj_func=obj_func,
        algorithm=algo,
        performance_metrics=performance_metrics,
        input_dim=input_dim,
        noise_type="noiseless",
        noise_level=0.0,
        policy=policy,
        batch_size=batch_size,
        num_init_points=2 * (input_dim + 1),
        num_iter=max_iter,
        first_trial=1,
        last_trial=5,
        save_data=True,
        restart=False,
        x_batch=np.array(x_path),
    )

# graph results
policies_str = ",".join(policies)
command = " ".join([
    "python",
    f"{script_dir}/graph.py",
    "-s",
    "-t",
    f"--problem {problem}",
    f"--policies {policies_str}",
])

subprocess.run(command, shell=True)

