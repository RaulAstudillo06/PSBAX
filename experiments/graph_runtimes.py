#%%
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt


# For running in jupyter notebook
print(os.getcwd())
cwd = os.getcwd()

# when run in jupyter notebook: /home/ec2-user/projects/PSBAX/experiments
sys.path.append('../')
from src.performance_metrics import *

#%%

# TODO
problem_setting = [
    # "discobax",
    # "single-objective",
    # "multi-objective",
    # "shortest-path",
    "topk",
    # "level-set",
] # Comment out the rest, only keep one

# TODO
# problem = "topk_original"
# problem = "topk_himmelblau"
# problem = "topk_ori_1"
# problem = "topk_him_1"
# problem = "dijkstra"
# problem = "hartmann_6d"
# problem = "ackley_10d"
# problem = "discobax_1"
# problem = "levelset_volcano_Raul"
# problem = "sanchez"
# problem = "schimdt"
# problem = "rosenbrock"
problem = "himmelblau"
results_dir = os.path.join(".", problem_setting[0], "results") 
path = os.path.join(results_dir, problem)

policies = [
    "bax",
    "ps",
    # "random",
    # "bax_gp_lbfgsb",
    # "ps_gp_lbfgsb",
    # "bax_modelgp_dim5",
    # "ps_modelgp_dim5",
]

# TODO
iters = 100
batch_size = 1
trials = np.arange(1, 30)

runtime_arrs = {} # policy -> runtimes per iteration
for policy in policies:

    files_dir = os.path.join(path, policy + "_" + str(batch_size) + "/runtimes")
    
    for f in os.listdir(files_dir):
        if f.endswith(".txt") and "runtimes" in f:
            if int(f.split(".")[0].split("_")[-1]) not in trials:
                continue
            runtimes = np.loadtxt(os.path.join(files_dir, f)).squeeze() # runtimes of one trial
            if iters is None:
                iters = len(runtimes)
            runtime_arrs[policy] = runtime_arrs.get(policy, np.zeros((iters, )))
            try:
                runtime_arrs[policy] += runtimes[:iters]
            except:
                print(f)
                continue
                
#%%
print(problem)
for policy in runtime_arrs:
    print(policy, runtime_arrs[policy].mean())
#%%

for policy in runtime_arrs:
    plt.plot(np.arange(iters), runtime_arrs[policy], label=policy)
plt.xlabel("Iteration")
plt.ylabel("Runtime (s)")
plt.legend()
plt.show()

           
# %%
