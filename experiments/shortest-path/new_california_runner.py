#%%
import os
import sys
sys.setrecursionlimit(10000) 
import torch
import pandas as pd
import numpy as np
import argparse

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
# debug._set_state(False)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
src_dir = "/".join(script_dir.split("/")[:-2]) # src directory is two levels up
sys.path.append(src_dir)

# == if using colab == 
sys.path.append('../')


# from src.bax.alg.dijkstra import Dijkstra
# from src.bax.util.domain_util import unif_random_sample_domain
# from src.bax.util.graph import make_grid, edges_of_path, positions_of_path, area_of_polygons
# from src.bax.util.graph import make_vertices, make_edges
from src.bax.alg.dijkstra_nx import DijkstraNx
from src.experiment_manager import experiment_manager
from src.performance_metrics import NewShortestPathCost

# get list of directories in os.getcwd()
# print(os.listdir(os.getcwd()))

parser = argparse.ArgumentParser()
parser.add_argument('--policy', type=str, default='bax')
parser.add_argument('--trials', type=int, default=3)
parser.add_argument('--save', '-s', action='store_true', default=False)
parser.add_argument('--restart', '-r', action='store_true', default=False)
parser.add_argument('--num_init_points', type=int, default=5)
# ====== To run ======
# python california_runner.py -s --policy bax --trials 5

args = parser.parse_args()
first_trial = 1
last_trial = args.trials

# ======== data processing

if "experiments" in os.listdir(os.getcwd()):
    print(os.listdir(os.getcwd()))
    os.chdir("experiments/")

df_edges = pd.read_csv(f"{script_dir}/data/new_edges.csv", index=False)
df_nodes = pd.read_csv(f"{script_dir}/data/new_nodes.csv", index=False)

df_edges["pos_weight"] = df_edges["mean_elevation"] - df_edges["mean_elevation"].min() 

# find the closest node to the start and end points
start = (-121, 39)
end = (-117, 35)
start_node = df_nodes.iloc[((df_nodes["longitude"] - start[0])**2 + (df_nodes["latitude"] - start[1])**2).idxmin()]
end_node = df_nodes.iloc[((df_nodes["longitude"] - end[0])**2 + (df_nodes["latitude"] - end[1])**2).idxmin()]

edge_positions = df_edges[["norm_longitude", "norm_latitude"]].to_numpy()
edge_pos_to_weight = {tuple(e): w for e, w in zip(edge_positions, df_edges["pos_weight"])}

algo_params = {
    "name" : "DijkstraNx",
    "start": start_node.name,
    "end": end_node.name,
    "df_edges": df_edges,
    "edge_weight": "pos_weight",
}
algo = DijkstraNx(
    algo_params
)


def obj_func(X):
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
    y_values = []
    for x in X:
        x_tuple = tuple(x.tolist())
        y_values.append(edge_pos_to_weight[x_tuple])
    return torch.tensor(y_values)

algo_copy = algo.get_copy()
true_ep, true_output = algo_copy.run_algorithm_on_f(obj_func)
# optimum_cost = true_output[0] = 72.80633717926638

performance_metrics = [
    NewShortestPathCost(
        algo, 
    )
]

input_dim = 2
# Nodes are indices
experiment_manager(
    problem="california",
    algorithm=algo,
    obj_func=obj_func,
    performance_metrics=performance_metrics,
    input_dim=input_dim,
    noise_type="noiseless",
    noise_level=0.0,
    policy=args.policy,
    batch_size=1,
    num_init_points=args.num_init_points,
    num_iter=70,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=args.restart,
    save_data=args.save,
    edge_positions=edge_positions,
    exe_paths=20,
    bax_num_cand=500,
    model_verbose=False,
)



# =================

