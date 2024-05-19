#%%
import os
import sys
import math
sys.setrecursionlimit(10000) 
import torch
import json
import pandas as pd
import numpy as np
import argparse

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
# debug._set_state(False)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
src_dir = "/".join(script_dir.split("/")[:-2]) # src directory is two levels up
sys.path.append(src_dir)


# from src.bax.alg.dijkstra import Dijkstra
# from src.bax.util.domain_util import unif_random_sample_domain
# from src.bax.util.graph import make_grid, edges_of_path, positions_of_path, area_of_polygons
# from src.bax.util.graph import make_vertices, make_edges
from src.bax.alg.dijkstra import DijkstraNx, calculate_work
from src.experiment_manager import experiment_manager
from src.performance_metrics import NewShortestPathCost

# get list of directories in os.getcwd()
# print(os.listdir(os.getcwd()))

parser = argparse.ArgumentParser()
parser.add_argument('--policy', type=str, default='random')
parser.add_argument('--trials', type=int, default=5)
parser.add_argument('--first_trial', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--max_iter', type=int, default=50)
parser.add_argument('--save', '-s', action='store_true', default=False)
parser.add_argument('--restart', '-r', action='store_true', default=False)
args = parser.parse_args()
# ====== To run ======
# python new_california_runner.py -s --policy bax --trials 5


# ======== data processing

# df_edges = pd.read_csv(f"{script_dir}/data/new_edges.csv", index_col="edgeid")
# df_nodes = pd.read_csv(f"{script_dir}/data/new_nodes.csv", index_col="nodeid")
df_edges = pd.read_csv(f"{script_dir}/data/california_edges.csv", index_col="edgeid")
df_nodes = pd.read_csv(f"{script_dir}/data/california_nodes.csv", index_col="nodeid")
df_nodes["norm_longitude"] = (df_nodes["longitude"] - df_nodes["longitude"].min()) / (df_nodes["longitude"].max() - df_nodes["longitude"].min())
df_nodes["norm_latitude"] = (df_nodes["latitude"] - df_nodes["latitude"].min()) / (df_nodes["latitude"].max() - df_nodes["latitude"].min())
df_edges_clone = df_edges.copy()
df_edges_clone["start_nodeid"] = df_edges["end_nodeid"]
df_edges_clone["end_nodeid"] = df_edges["start_nodeid"]
df_edges = pd.concat([df_edges, df_edges_clone], ignore_index=True)


def from_row_to_coord(row):
    u, v = row["start_nodeid"], row["end_nodeid"]
    start_node = df_nodes.loc[u]
    end_node = df_nodes.loc[v]
    edge = np.array([start_node["norm_longitude"], start_node["norm_latitude"], end_node["norm_longitude"], end_node["norm_latitude"]])
    return edge

def from_row_to_work(row):
    u, v = row["start_nodeid"], row["end_nodeid"]
    start_node = df_nodes.loc[u]
    end_node = df_nodes.loc[v]
    tuple1 = (start_node["longitude"], start_node["latitude"], start_node["elevation"])
    tuple2 = (end_node["longitude"], end_node["latitude"], end_node["elevation"])
    # tuple1 = (start_node["norm_longitude"], start_node["norm_latitude"], start_node["elevation"])
    # tuple2 = (end_node["norm_longitude"], end_node["norm_latitude"], end_node["elevation"])
    return calculate_work(tuple1, tuple2)



df_edges["coord"] = df_edges.apply(from_row_to_coord, axis=1)
df_edges["work"] = df_edges.apply(from_row_to_work, axis=1)
edge_coords = np.vstack(df_edges["coord"])
# normalize edge coords to 0, 1
# edge_coords = (edge_coords - edge_coords.min(axis=0)) / (edge_coords.max(axis=0) - edge_coords.min(axis=0))
# assign new coords to df_edges["coord"]
df_edges["coord"] = list(edge_coords)
edge_work = df_edges["work"].to_numpy()
edge_coord_to_work = {tuple(e): w for e, w in zip(edge_coords, edge_work)}


# find the closest node to the start and end points
# start = (-121, 39)
# end = (-117, 35)
# start_node = df_nodes.iloc[((df_nodes["longitude"] - start[0])**2 + (df_nodes["latitude"] - start[1])**2).idxmin()]
# end_node = df_nodes.iloc[((df_nodes["longitude"] - end[0])**2 + (df_nodes["latitude"] - end[1])**2).idxmin()]
# start = start_node.name
# goal = end_node.name
# 720035.454

start = 3939
goal = 446
# true cost = 515605.3191434491
# edge_coords = df_edges[["norm_longitude", "norm_latitude"]].to_numpy()
# edge_pos_to_weight = {tuple(e): w for e, w in zip(edge_coords, df_edges["pos_weight"])}

algo_params = {
    "name" : "DijkstraNx",
    "start": start,
    "end": goal,
    "df_edges": df_edges,
    "df_nodes": df_nodes,
    "softplus" : True,
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
        y_values.append(edge_coord_to_work[x_tuple])
    return torch.tensor(y_values)

algo_copy = algo.get_copy()
true_ep, true_output = algo_copy.run_algorithm_on_f(obj_func)

print(f"True cost: {true_ep.true_cost}")
algo_metric = algo.get_copy()

performance_metrics = [
    NewShortestPathCost(
        algo_metric,
        optimum_cost=true_ep.true_cost
    )
]

# args.save = True
problem = "new_california"
if args.save:
    results_dir = f"./results/{problem}"
    os.makedirs(results_dir, exist_ok=True)
    policy = args.policy
    params_dict = vars(args)
    for k,v in algo_params.items():
        if k == "start" or k == "end":
            v = int(v)
        if k not in params_dict and k != "df_edges" and k != "df_nodes":
            params_dict[k] = v
    # params_dict["euclidean_dist"] = True
    params_dict["friction"] = 10

    with open(os.path.join(results_dir, f"{policy}_{args.batch_size}_params.json"), "w") as file:
        json.dump(params_dict, file)

first_trial = args.first_trial
last_trial = args.first_trial + args.trials - 1

n_dim = 4
# Nodes are indices
experiment_manager(
    problem=problem,
    algorithm=algo,
    obj_func=obj_func,
    performance_metrics=performance_metrics,
    input_dim=n_dim,
    noise_type="noiseless",
    noise_level=0.0,
    policy=args.policy,
    batch_size=args.batch_size,
    num_init_points=2 * (n_dim + 1),
    num_iter=args.max_iter,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=args.restart,
    save_data=args.save,
    edge_coords=edge_coords,
    exe_paths=30,
)



# =================


# %%
