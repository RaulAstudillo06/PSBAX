#%%
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

from botorch.optim import optimize_acqf_discrete
from copy import deepcopy

cwd = os.getcwd()
src_dir = "/".join(cwd.split("/")[:-1])
sys.path.append(src_dir)

from src.bax.alg.algorithms import TopK
from src.acquisition_functions.posterior_sampling import gen_posterior_sampling_batch
from src.acquisition_functions.bax_acquisition import BAXAcquisitionFunction
from src.fit_model import fit_model
# from src.performance_metrics import evaluate_performance
from src.utils import (
    generate_initial_data,
    generate_random_points,
    get_obj_vals,
    seed_torch,
    optimize_acqf_and_get_suggested_batch,
)
from src.performance_metrics import output_dist_fn_norm, output_dist_fn_jaccard

#%%
problem_dir = os.path.join(cwd, "topk/results/topk_himmelblau") 
bax_dir = os.path.join(problem_dir, "bax_1")
ps_dir = os.path.join(problem_dir, "ps_1")

x_np = np.load(os.path.join(cwd, "topk/data/him_x_np.npy"))
x_path = [list(x) for x in x_np]

k = 4
algo = TopK({"x_path": x_path, "k": k}, verbose=False)
#%%
def himmelblau(X, minimize=False):
    a = X[:, 0]
    b = X[:, 1]
    result = (a ** 2 + b - 11) ** 2 + (a + b ** 2 - 7) ** 2
    if not minimize:
        return -result
    else:
        return result
    
def obj_func(X, domain=[-6, 6]):
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X)
    # rescale X from 0, 1 to domain
    X_rescaled = X.clone()
    X_rescaled = (domain[1] - domain[0]) * X_rescaled + domain[0]
    f_0 = himmelblau
    return f_0(X_rescaled)

policy = "ps"
dir = os.path.join(problem_dir, f"{policy}_1")
trial = 1
inputs = np.loadtxt(os.path.join(dir, "inputs", f"inputs_{trial}.txt"))
obj_vals = np.loadtxt(os.path.join(dir, "obj_vals", f"obj_vals_{trial}.txt"))
inputs = torch.from_numpy(inputs)
obj_vals = torch.from_numpy(obj_vals)
kwargs = {}
model = fit_model(
    inputs,
    obj_vals,
    model_type="gp",
    **kwargs
)

algo_gt = algo.get_copy()
_, output_gt = algo_gt.run_algorithm_on_f(obj_func)
x_top_k = np.array(output_gt.x)

_, output_mf = algo.run_algorithm_on_f(lambda x: model.posterior(x).mean)
x_output = np.array(output_mf.x)
#%%

def create_mesh(xmin, xmax, steps=20):
    length = xmax - xmin
    # xlim = [xmin - 0.05 * length, xmax + 0.05 * length]
    # ax = torch.linspace(xlim[0], xlim[1], steps)
    ax = torch.linspace(xmin + 0.05 * length, xmax - 0.05 * length, steps)
    xx = torch.meshgrid(ax, ax, indexing="ij")
    return xx

def reshape_mesh(xx):
    x1 = xx[0].reshape(-1)
    x2 = xx[1].reshape(-1)
    return torch.stack([x1, x2], dim=1)

x_mesh = create_mesh(-0.1, 1.1, steps=10)
z = obj_func(reshape_mesh(x_mesh).numpy()).reshape(x_mesh[0].shape)
def plot(policy, file_format="pdf", show_title=False):
    colors = {
        "ps" : "b",
        "bax" : "g",
    }
    fig, ax = plt.subplots(
        figsize=(8, 8)
    )
    ax.contourf(x_mesh[0], x_mesh[1], z, cmap="YlOrRd", alpha=0.6, levels=40)
    ax.scatter(
        x_top_k[:, 0], 
        x_top_k[:, 1], 
        c="red", 
        marker="*",
        s=500,
        label="Top k",
    )
    ax.scatter(
        x_np[:, 0],
        x_np[:, 1],
        c="#6B3E26",
        marker=".",
        s=5,
        alpha=0.5,
        label="Discrete Set",
    )
    
    ax.scatter(
        inputs[:, 0], 
        inputs[:, 1], 
        c=colors[policy], 
        marker="o",
        alpha = 0.2,
        s=100,
        label="Sampled",
    )
    ax.scatter(
        x_output[:, 0],
        x_output[:, 1],
        c= "#ff7f0e",
        marker="P",
        s = 100,
        label="Algorithm Output",
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=5)
    if show_title:
        if policy == "ps":
            ax.set_title("Posterior Sampling")
        elif policy == "bax":
            ax.set_title("Expected Information Gain")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(
        os.path.join(problem_dir, f"plots/{policy}.{file_format}"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()

plot(policy)

#%%