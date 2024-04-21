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

# problem = "topk_original_200"
# problem = "ackley_10d"
# problem = "dtlz1_6d"
problem = "dtlz1_6d_2obj"
# problem = "dtlz2_3d"
# problem = "dtlz2_3d_2obj"
# problem = "dtlz2_10d"
# problem = "zdt1_30d"
# problem = "hartmann"
# problem = "hartmann_6d"
# problem = "rastrigin_10d"
# problem = "dijkstra"
# problem = "california"
# problem = "california_bax"
# problem = "sanchez_2021_tau_dim5_size1700"
# problem = "schmidt_2021_ifng_top_10000"
# problem = "schmidt_2021_ifng_top_1700"
# problem = "sanchez_2021_tau_top_1700"

policies = [
    "ps", 
    "bax", 
    "random",
    # "OPT", 
    # "ps200",
    # "bax200",
    # "OPT200",
    # "ps_modelgp",
    # "bax_modelgp",
    # "ps_modelgp_cma",
    # "bax_modelgp_cma",
    # "ps_modelgp_mut",
    # "bax_modelgp_mut",
    # "ps_modelgp_dim5",
    # "bax_modelgp_dim5",
    # "OPT_modelgp_dim5"
]
graph_trials = 4
show_title = True
save_fig = True
path = os.path.join(results_dir, problem)
batch_size = 1

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


metrics = []

if "topk" in problem:
    metrics = ['Jaccard', 'Norm']
    # metrics = ['Norm']
elif "dijkstra" in problem:
    metrics = ['ShortestPathCost', 'ShortestPathArea']
elif "california_bax" in problem:
    metrics = ['ShortestPathCost', 'ShortestPathArea', 'Error']
elif "hartmann" in problem or "rastrigin" in problem or "ackley" in problem:
    metrics = ['best_value']
elif "zdt" in problem or "dtlz":
    metrics = ['Hypervolume']
else:
    metrics = ['DiscoBAXMetric']

# bax_iters = 100
algo_performance_arrs = {}
for policy in policies:
    iters = 0
    files_dir = os.path.join(path, policy + "_" + str(batch_size))
    arrs = {}

    
    for f in os.listdir(files_dir):
        if f.endswith(".txt") and "performance_metric" in f:
            if int(f.split(".")[0].split("_")[-1]) > graph_trials:
                continue
            arr = np.loadtxt(os.path.join(files_dir, f))
            if len(arr.shape) == 1:
                arr = arr[:, None]
            
            # if "bax" in policy:
            #     arr = arr[:bax_iters, :]    

            for i, metrics_name in enumerate(metrics):
                vals = arr[:, i]
                arrs[metrics_name] = arrs.get(metrics_name, []) + [vals]
            
    for (metrics_name, arr) in arrs.items():
        arrs[metrics_name] = np.vstack(arr) # (n_trials, n_iter)

    algo_performance_arrs[policy] = arrs

#%%
max_iters = max([arr.shape[1] for arr in algo_performance_arrs[policy].values()])


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

        # if "bax" in policy:
        #     iters = bax_iters
        # else:
        iters = max_iters

        arr = arrs[metrics_name][:, :iters]
        
        x_range = np.arange(iters)
        if key is not None:
            label = policy_to_label[key]
            color = policy_to_hex[key]
            
            ax.plot(x_range, np.mean(arr, axis=0), label=label, color=color)
            ax.fill_between(
                x_range,
                np.mean(arr, axis=0) - 2 * np.std(arr, axis=0, ddof=1),
                np.mean(arr, axis=0) + 2 * np.std(arr, axis=0, ddof=1),
                alpha=0.2,
                color=color,
            )
        
        else:
            ax.plot(x_range, np.mean(arr, axis=0), label=policy)
            # plot +-2 standard err
            ax.fill_between(
                x_range,
                np.mean(arr, axis=0) - 2 * np.std(arr, axis=0, ddof=1),
                np.mean(arr, axis=0) + 2 * np.std(arr, axis=0, ddof=1),
                alpha=0.2,
                color=color,
            )

    ax.set_xlabel("Iteration")
    # ax.set_ylabel(metrics_name)
    # ax.legend()
    if show_title:
        ax.set_title(metrics_name)
    else:
        print(f"{metrics_name}")

    # set legend to below the plot
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(policies))

    if save_fig:
        fig_dir = os.path.join(path, "plots")
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir, exist_ok=True)
        # add strategy to file name
        fig_name = "_".join(policies) + "_" + metrics_name + "_" + str(arr.shape[0])
        fig.savefig(
            os.path.join(
                fig_dir, fig_name + ".png"
            ),
            # bbox_inches="tight",
            dpi=300,
        )
    plt.show()


#%%
