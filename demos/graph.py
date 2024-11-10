import os
import sys
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
src_dir = "/".join(script_dir.split("/")[:-1]) # src directory is one level up
sys.path.append(src_dir)

parser = argparse.ArgumentParser()  
parser.add_argument('--problem', type=str, default='topk_rosenbrock')
parser.add_argument('--policies', type=str, default='ps,bax', help='comma separated list of policies')
parser.add_argument('--save', '-s', action='store_true', default=False)
parser.add_argument('--show_title', '-t', action='store_true', default=False)
parser.add_argument('--file_format', type=str, default='pdf')
args = parser.parse_args()

# to run
# python figures.py --problem topk_rosenbrock --policies ps,bax -s -t

file_format = f".{args.file_format}"
save_fig = args.save
show_title = args.show_title

problem = args.problem

policies = args.policies.split(',')
results_dir = f"{script_dir}/results/{problem}"
params_file = os.path.join(results_dir, "params.json")
with open(params_file, "r") as file:
    params = json.load(file)

batch_size = params["batch_size"]
iters = params["max_iter"]

optimum = None

setting_to_metric = {
    "discobax" : ["DiscoBAXMetric"],
    "local-bo": ["BestValue"],
    "multi-objective": ["HypervolumeDifference"],
    "topk": ["Jaccard"],
    "shortest-path": ["ShortestPathCost", "Regret"],
    "level-set": ["F1Score"],
}
metrics = setting_to_metric[problem.split("_")[0]]

policy_to_hex = {
    "ps" : 'b',
    "bax" : 'g',
    "random" : '#7f7f7f',
    "lse" : "#e377c2",
}
policy_to_label = {
    "ps": "PS (Ours)",
    "bax": "EIG",
    "lse": "LSE",
    "random": "Random",
}


algo_performance_arrs = {}
for policy in policies:
    
    files_dir = os.path.join(results_dir, policy + "_" + str(params["batch_size"]))
    arrs = {}
    for f in os.listdir(files_dir):
        if f.endswith(".txt") and "performance_metric" in f:
            arr = np.loadtxt(os.path.join(files_dir, f))

            if "lse" in policy:
                arr = arr[:, 0]
            
            if len(arr.shape) == 1:
                arr = arr[:, None]

            for i, metrics_name in enumerate(metrics):
                vals = arr[:, i]
                arrs[metrics_name] = arrs.get(metrics_name, []) + [vals]
            
    for (metrics_name, arr) in arrs.items():
        try:
            arrs[metrics_name] = np.vstack(arr) # (n_trials, n_iter)
        except:
            print(f"Error in {policy} {metrics_name}")
            # pass
        if optimum is not None:
            arrs[metrics_name] = optimum - arrs[metrics_name]

    algo_performance_arrs[policy] = arrs


for metrics_name in metrics:
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    for policy in policies:
        
        arrs = algo_performance_arrs[policy]
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
        else:
            key = None

        arr = arrs[metrics_name][:, :iters]

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

    ax.set_xlabel("Iteration")
    ax.set_ylabel(metrics_name)
    ax.legend()
    if show_title:
        ax.set_title(f"{problem} q={batch_size} {metrics_name}")
    else:
        # print(f"{metrics_name}")
        print(f"{problem} q={batch_size} {metrics_name}")

    # set legend to below the plot
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(policies))
    plt.tight_layout()
    if save_fig:
        fig_dir = os.path.join(script_dir, "plots")
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