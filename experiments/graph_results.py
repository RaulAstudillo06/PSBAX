#%%
import os
import sys
import torch
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# matplotlib and seaborn settings
#sns.set()
#sns.color_palette("dark")
sns.set_style("white")
fontsize = 22
plt.rc("xtick", labelsize=fontsize)
plt.rc("ytick", labelsize=fontsize)


# For running in jupyter notebook
print(os.getcwd())
# when run in jupyter notebook: /home/ec2-user/projects/PSBAX/experiments
sys.path.append('../')
# from src.performance_metrics import *


problem_setting = [
    "discobax",
    # "single-objective",
    # "multi-objective",
    # "shortest-path",
    # "topk",
    # "level-set",
] # Comment out the rest, only keep one
results_dir = os.path.join(".", problem_setting[0], "results")

# problem = "topk_original"
# problem = "old_topk_himmelblau"
# problem = "topk_himmelblau"
# problem = "ackley_10d"
# problem = "ackley_5d"
# problem = "dtlz1_6d"
# problem = "dtlz1_6d_2obj"
# problem = "dtlz2_3d"
# problem = "dtlz2_3d_2obj"
# problem = "dtlz2_6d_2obj"
# problem = "dtlz2_10d"
# problem = "zdt1_30d"
# problem = "zdt2_6d_2obj"
# problem = "zdt2_6d_2obj_noise0.1"
# problem = "hartmann_6d"
# problem = "ackley_10d"
# problem = "rastrigin_10d"
# problem = "dijkstra"
# problem = "california"
# problem = "california_bax"
# problem = "sanchez_2021_tau_dim5_size1700"
# problem = "schmidt_2021_ifng_top_10000"
# problem = "schmidt_2021_ifng_top_1700"
# problem = "old_discobax_sanchez_2021_tau_top_5000"
# problem = "discobax_sanchez_2021_tau_top_5000"
problem = "discobax_schmidt_2021_ifng_top_5000"
# problem = "dijkstra"
# problem = "lbfgsb_rastrigin_10d"
problem = "levelset_himmelblau"
# problem = "levelset_volcano"
# problem = "levelset_griewank"

policies = [
    # "random",
    # "bax", 
    # "ps", 
    # "lse",
    # "OPT",
    # "random_modelgp", 
    # "bax_modelgp",
    # "ps_modelgp",
    # "random_gp_lbfgsb",
    # "qei_gp_lbfgsb",
    # "bax_gp_lbfgsb",
    # "ps_gp_lbfgsb",
    "random_modelgp_dim5",
    "bax_modelgp_dim5",
    "ps_modelgp_dim5",
    # "OPT_modelgp_dim5",
    # "bax_modelgp_dim20",
    # "ps_modelgp_dim20",
    # "OPT_modelgp_dim20",
    # "bax_dim5_init12",
    # "ps_dim5_init12",
    # "OPT_dim5_init12",
]
graph_trials = [
    1, 
    2, 
    3, 
    4, 
    5, 
    6, 
    7, 
    8, 
    9, 
    10,
]
graph_trials = [i for i in range(1, 31)]
show_title = False
save_fig = True
path = os.path.join(results_dir, problem)
batch_size = 1
log = False
bax_iters = 100
max_iters = 100
# bax_iters = None
optimum = None # 3.32237
file_format = ".png"
file_format = ".pdf"


policy_to_hex = {
    # "ps": "#1f77b4",
    # "bax": "#ff7f0e",
    # "OPT": "#2ca02c",
    # "random": "#8c564b",
    "ps" : 'b',
    "bax" : 'g',
    "qehvi" : 'r',
    "qei" : 'r',
    "random" : '#7f7f7f',
    # "bax_modelgp_dim20": "#e377c2",
    # "ps_modelgp_dim20": "#7f7f7f",
    "lse" : "#e377c2",
}

policy_to_label = {
    "ps": "PS (Ours)",
    "bax": "EIG",
    "qehvi": "EHVI",
    "qei": "EI",
    "lse" "LSE"
    "OPT": "OPT",
    "random": "Random",
}

setting_to_metric = {
    "discobax" : ["DiscoBAXMetric"],
    "single-objective": ["BestValue"],
    "multi-objective": ["HypervolumeDifference"],
    "topk": ["Jaccard", "SumOfValues"],
    "shortest-path": ["ShortestPathCost", "Regret"],
    "level-set": ["F1Score"],
}
metrics = setting_to_metric[problem_setting[0]]

# bax_iters = 100
algo_performance_arrs = {}
for policy in policies:
    iters = 0
    files_dir = os.path.join(path, policy + "_" + str(batch_size))
    arrs = {}
    for f in os.listdir(files_dir):
        if f.endswith(".txt") and "performance_metric" in f:
            # if int(f.split(".")[0].split("_")[-1]) > graph_trials:
            if int(f.split(".")[0].split("_")[-1]) not in graph_trials:
                continue
            arr = np.loadtxt(os.path.join(files_dir, f))

            if "lse" in policy:
                arr = arr[:, 0]
            
            if len(arr.shape) == 1:
                arr = arr[:, None]
            
            if "bax" in policy:
                if arr.shape[0] < bax_iters:
                    print(f)
                arr = arr[:bax_iters, :]    
            else:
                if arr.shape[0] < max_iters:
                    print(f)
                arr = arr[:max_iters, :]

            for i, metrics_name in enumerate(metrics):
                vals = arr[:, i]
                arrs[metrics_name] = arrs.get(metrics_name, []) + [vals]
            
    for (metrics_name, arr) in arrs.items():
        arrs[metrics_name] = np.vstack(arr) # (n_trials, n_iter)
        if optimum is not None:
            arrs[metrics_name] = optimum - arrs[metrics_name]
        if log:
            arrs[metrics_name] = np.log(arrs[metrics_name])

    algo_performance_arrs[policy] = arrs


#%%
# max_iters = max([arr.shape[1] for arr in algo_performance_arrs[policy].values()])


try:
    OPT_values = algo_performance_arrs["OPT"] # TODO: comment out
except:
    pass
# plot
for metrics_name in metrics:
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
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
        elif "lse" in policy:
            key = "lse"
        elif "qehvi" in policy:
            key = "qehvi"
        elif "qei" in policy:
            key = "qei"
        else:
            key = None

        if "bax" in policy:
            iters = bax_iters
        else:
            iters = max_iters

        # if bax_iters is not None:
        #     iters = bax_iters

        arr = arrs[metrics_name][:, :iters]
        print(arr.shape)
        
        x_range = np.arange(iters)
        if key is not None:
            label = policy_to_label[key]
            color = policy_to_hex[key]
            
            ax.plot(x_range, np.mean(arr, axis=0), label=label, color=color)
            ax.fill_between(
                x_range,
                np.mean(arr, axis=0) - 2 * np.std(arr, axis=0)/np.sqrt(arr.shape[0]),
                np.mean(arr, axis=0) + 2 * np.std(arr, axis=0)/np.sqrt(arr.shape[0]),
                alpha=0.2,
                color=color,
            )
        
        else:
            ax.plot(x_range, np.mean(arr, axis=0), label=policy)
            # plot +-2 standard err
            ax.fill_between(
                x_range,
                np.mean(arr, axis=0) - 2 * np.std(arr, axis=0)/np.sqrt(arr.shape[0]),
                np.mean(arr, axis=0) + 2 * np.std(arr, axis=0)/np.sqrt(arr.shape[0]),
                alpha=0.2,
            )

            # plot log scale
    # ax.set_yscale("log")

    ax.set_xlabel("Iteration", fontsize=fontsize)
    # ax.set_ylabel(metrics_name)
    # if log:
    #     ax.set_ylabel("Log Value")
    # else:
    #     ax.set_ylabel("Value")
    # ax.legend()
    if show_title:
        # ax.set_title(metrics_name)
        ax.set_title(f"{problem} q={batch_size} {metrics_name}")
    else:
        # print(f"{metrics_name}")
        print(f"{problem} q={batch_size} {metrics_name}")

    # set legend to below the plot
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(policies), fontsize=fontsize)
    plt.tight_layout()
    if save_fig:
        fig_dir = os.path.join(path, "plots")
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir, exist_ok=True)
        # add strategy to file name
        fig_name = problem + "_" + str.lower(metrics_name.strip(" ")) + "_batch" + str(batch_size)
        fig.savefig(
            os.path.join(
                fig_dir, fig_name + file_format
            ),
            bbox_inches="tight",
            #dpi=300,
        )
    plt.show()


#%%


    