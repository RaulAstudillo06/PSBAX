#%%
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt


# For running in jupyter notebook
print(os.getcwd())
# when run in jupyter notebook: /home/ec2-user/projects/PSBAX/experiments
sys.path.append('../')
from src.performance_metrics import *

results_dir = "./results/"
# problem = "dijkstra"
# problem = "hartmann"
problem = "topk"
path = os.path.join(results_dir, problem)
batch_size = 1
save_fig = True
policies = ["bax", "ps"]

metrics = []

if problem == "topk":
    metrics = ['Jaccard', 'Norm']
elif problem == "dijkstra":
    metrics = ['ShortestPathCost', 'ShortestPathArea']
elif problem == "hartmann":
    metrics = ['ObjValAtMaxPostMean']

algo_performance_arrs = {}
for policy in policies:
    files_dir = os.path.join(path, policy + "_" + str(batch_size))
    arrs = {}

    for f in os.listdir(files_dir):
        if f.endswith(".txt"):
            arr = np.loadtxt(os.path.join(files_dir, f))
            if len(arr.shape) == 1:
                arr = arr[:, None]
            for i, metrics_name in enumerate(metrics):
                arrs[metrics_name] = arrs.get(metrics_name, []) + [arr[:, i]]
    for (metrics_name, arr) in arrs.items():
        arrs[metrics_name] = np.vstack(arr) # (n_trials, n_iter)
    algo_performance_arrs[policy] = arrs

iters = min([arr.shape[1] for arr in algo_performance_arrs[policy].values()])
# plot
for metrics_name in metrics:
    fig, ax = plt.subplots(figsize=(10, 8))
    for policy in policies:
        arrs = algo_performance_arrs[policy]
        ax.plot(np.arange(iters), np.mean(arrs[metrics_name], axis=0), label=policy)
        ax.fill_between(
            np.arange(iters),
            np.mean(arrs[metrics_name], axis=0) - np.std(arrs[metrics_name], axis=0),
            np.mean(arrs[metrics_name], axis=0) + np.std(arrs[metrics_name], axis=0),
            alpha=0.2,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(metrics_name)
    ax.legend()
    ax.set_title(metrics_name + " vs. Iteration")
    if save_fig:
        fig_dir = os.path.join(path, "plots")
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir, exist_ok=True)
        # add strategy to file name
        fig_name = "_".join(policies) + "_" + metrics_name
        fig.savefig(
            os.path.join(
                fig_dir, fig_name + ".png"
            ),
            bbox_inches="tight",
            dpi=300,
        )
    plt.show()


#%%