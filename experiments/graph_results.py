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
# problem = "topk_original"
# problem = "schmidt_2021_ifng_size1700"
# problem = "test_schmidt_2021_ifng"
# problem = "sanchez_2021_tau_size1700"

# problem = "hartmann"
# problem = "rastrigin_10d"
# problem = "dijkstra"
# problem = "california"
problem = "california_bax"
# problem = "sanchez_2021_tau_dim5_size1700"
# problem = "schmidt_2021_ifng_top_10000"
# problem = "schmidt_2021_ifng_top_1700"
# problem = "sanchez_2021_tau_top_1700"
path = os.path.join(results_dir, problem)
batch_size = 1
save_fig = True
policies = [
    "ps", 
    "bax", 
    # "OPT", 
    # "ps_dkgp_643210",
    # "ps_pca_10", 
    # "random",
    # "ps_modelgp",
    # "bax_modelgp",
    # "ps_modelgp_cma",
    # "bax_modelgp_cma",
    # "OPT_modelgp_dim20",
    # "ps_modelgp_dim5",
    # "bax_modelgp_dim5",
    # "OPT_modelgp_dim5"
]
policy_to_hex = {
    "ps": "#1f77b4",
    "bax": "#ff7f0e",
    "OPT": "#2ca02c",
    "random": "#8c564b",
    # "bax_modelgp_dim20": "#e377c2",
    # "ps_modelgp_dim20": "#7f7f7f",
}

policy_to_label = {
    "ps": "Posterior Sampling",
    "bax": "BAX",
    "OPT": "OPT",
    "random": "Random",
}
graph_trials = 3

metrics = []

if "topk" in problem:
    metrics = ['Jaccard', 'Norm']
    # metrics = ['Norm']
elif "dijkstra" in problem or "california" in problem:
    metrics = ['ShortestPathCost', 'ShortestPathArea']
elif "hartmann" in problem or "rastrigin" in problem:
    metrics = ['best_value']
else:
    metrics = ['DiscoBAXMetric']


algo_performance_arrs = {}
for policy in policies:
    files_dir = os.path.join(path, policy + "_" + str(batch_size))
    arrs = {}

    
    for f in os.listdir(files_dir):
        if f.endswith(".txt") and "performance_metric" in f:
            if int(f.split(".")[0].split("_")[-1]) > graph_trials:
                continue
            arr = np.loadtxt(os.path.join(files_dir, f))
            if len(arr.shape) == 1:
                arr = arr[:, None]
            for i, metrics_name in enumerate(metrics):
                arrs[metrics_name] = arrs.get(metrics_name, []) + [arr[:, i]]
    for (metrics_name, arr) in arrs.items():
        arrs[metrics_name] = np.vstack(arr) # (n_trials, n_iter)
    algo_performance_arrs[policy] = arrs

iters = min([arr.shape[1] for arr in algo_performance_arrs[policy].values()])

try:
    OPT_values = algo_performance_arrs["OPT"] # TODO: comment out
except:
    pass
# plot
for metrics_name in metrics:
    fig, ax = plt.subplots(figsize=(10, 8))
    for policy in policies:
        # if policy == "OPT":
        #     continue
        arrs = algo_performance_arrs[policy]
        # plot OPT minus arr
        # if "OPT" in algo_performance_arrs:
        #     arrs[metrics_name] = OPT_values[metrics_name] - arrs[metrics_name]
        for policy_name, hex_color in policy_to_hex.items():
            if policy_name in policy:
                color = hex_color
            else:
                color = None
        
        if "ps" in policy:
            key = "ps"
        elif "bax" in policy:
            key = "bax"
        elif "random" in policy:
            key = "random"
        elif "OPT" in policy:
            key = "OPT"
        else:
            key = None
        
        if key is not None:
            label = policy_to_label[key]
            color = policy_to_hex[key]
            ax.plot(np.arange(iters), np.mean(arrs[metrics_name], axis=0), label=label, color=color)
            ax.fill_between(
                np.arange(iters),
                np.mean(arrs[metrics_name], axis=0) - np.std(arrs[metrics_name], axis=0),
                np.mean(arrs[metrics_name], axis=0) + np.std(arrs[metrics_name], axis=0),
                alpha=0.2,
                color=color,
            )

        
        else:
            ax.plot(np.arange(iters), np.mean(arrs[metrics_name], axis=0), label=policy)
            ax.fill_between(
                np.arange(iters),
                np.mean(arrs[metrics_name], axis=0) - np.std(arrs[metrics_name], axis=0),
                np.mean(arrs[metrics_name], axis=0) + np.std(arrs[metrics_name], axis=0),
                alpha=0.2,
            )

    ax.set_xlabel("Iteration")
    # ax.set_ylabel(metrics_name)
    # ax.legend()
    # ax.set_title(metrics_name + " vs. Iteration")
    print(f"{metrics_name}")

    # set legend to below the plot
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(policies))

    if save_fig:
        fig_dir = os.path.join(path, "plots")
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir, exist_ok=True)
        # add strategy to file name
        fig_name = "_".join(policies) + "_" + metrics_name + "_" + str(arrs[metrics_name].shape[0])
        fig.savefig(
            os.path.join(
                fig_dir, fig_name + ".png"
            ),
            bbox_inches="tight",
            dpi=300,
        )
    plt.show()


#%%
