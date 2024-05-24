#%%
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

from botorch.optim import optimize_acqf_discrete
from botorch.acquisition.analytic import PosteriorMean
from copy import deepcopy

cwd = os.getcwd()
src_dir = "/".join(cwd.split("/")[:-2])
sys.path.append(src_dir)

from src.bax.alg.levelset import LevelSetEstimator
from src.performance_metrics import F1Score
from src.acquisition_functions.lse import LSE
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


#%%
problem_dir = os.path.join(cwd, "results/levelset_volcano_Raul") 
bax_dir = os.path.join(problem_dir, "bax_1")
ps_dir = os.path.join(problem_dir, "ps_1")

policy = "bax"
dir = os.path.join(problem_dir, f"{policy}_1")
trial = 2
end_iter = None
inputs = np.loadtxt(os.path.join(dir, "inputs", f"inputs_{trial}.txt"))
obj_vals = np.loadtxt(os.path.join(dir, "obj_vals", f"obj_vals_{trial}.txt"))
inputs = torch.from_numpy(inputs)
obj_vals = torch.from_numpy(obj_vals)
if end_iter is not None:
    inputs = inputs[:end_iter]
    obj_vals = obj_vals[:end_iter]
kwargs = {}
model = fit_model(
    inputs,
    obj_vals,
    model_type="gp",
    **kwargs
)



#%%
dim = 2
mat = np.loadtxt(f"{cwd}/data/volcano_maungawhau.csv", delimiter=",")
x1 = np.linspace(0, 1, mat.shape[1]) # (61, )
x2 = np.linspace(0, 1, mat.shape[0]) # (87, )
z_true = mat
mat = mat.flatten()
xx = np.meshgrid(x1, x2)
x_set = np.hstack([xx[0].reshape(-1, 1), xx[1].reshape(-1, 1)]) # (87*61, 2)
x_to_elevation = {tuple(x): mat[i] for i, x in enumerate(x_set)}
threshold = 165
idx = np.argmax(mat)
x_init = torch.from_numpy(np.atleast_2d(x_set[idx]))
def obj_func(X):
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X)
    y = []
    for x in X:
        y.append(x_to_elevation[tuple(x.tolist())])
    return torch.tensor(y)

algo_params = {
    "name" : "SimpleLevelSet",
    "threshold" : threshold,
    "x_set" : x_set,
    # "x_init" : x_init,
    "no_copy" : True,
}
algo = LevelSetEstimator(algo_params)


algo_gt = algo.get_copy()
exe_path_gt, output_gt = algo_gt.run_algorithm_on_f(obj_func)

posterior_mean_func = PosteriorMean(model)
exe_path_mf, output_mf = algo.run_algorithm_on_f(posterior_mean_func)

def f1_score(x_gt, x_pred):
    x_gt_set = set()
    for x in x_gt:
        x_gt_set.add(tuple(x))
    x_pred_set = set()
    for x in x_pred:
        x_pred_set.add(tuple(x))
    tp = len(x_gt_set.intersection(x_pred_set))
    fp = len(x_pred_set.difference(x_gt_set))
    fn = len(x_gt_set.difference(x_pred_set))
    if tp == 0 or (tp + fp) == 0 or (tp + fn) == 0:
        return 0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

score = f1_score(output_gt, output_mf)

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




def plot(policy, file_format="pdf", show_title=False, save_fig=True):
    colors = {
        "ps" : "b",
        "bax" : "g",
    }
    fig, ax = plt.subplots(
        figsize=(8, 8)
    )
    # norm = plt.Normalize(np.quantile(z, 0.2), z.max())
    ax.contourf(
        xx[0], 
        xx[1], 
        z_true, 
        cmap="YlOrRd", 
        alpha=0.6, 
        levels=40, 
        # norm=norm,
    )
    
    
    ax.scatter(
        output_mf[:, 0], 
        output_mf[:, 1], 
        c="red", 
        marker="o",
        s=80,
        # alpha=0.5,
        label="Output",
    )
    ax.scatter(
        output_gt[:, 0],
        output_gt[:, 1],
        c="#6B3E26",
        marker=".",
        s=5,
        # alpha=0.5,
        label="Ground Truth",
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
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=5)
    # if show_title:
    #     if policy == "ps":
    #         ax.set_title("Posterior Sampling")
    #     elif policy == "bax":
    #         ax.set_title("Expected Information Gain")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # add f1 score below
    # ax.text(0.5, 0.05, f"F1 Score: {score:.2f}", ha="center", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_fig:
        dir = os.path.join(problem_dir, "plots")
        os.makedirs(dir, exist_ok=True)
        plt.savefig(
            os.path.join(dir, f"{policy}.{file_format}"),
            bbox_inches="tight",
            dpi=300,
        )

    plt.show()

plot(policy)

#%%