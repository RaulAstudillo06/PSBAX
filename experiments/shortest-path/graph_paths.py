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
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
# debug._set_state(False)

# script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
# src_dir = "/".join(script_dir.split("/")[:-2]) # src directory is two levels up
# sys.path.append(src_dir)
cwd = os.getcwd()
src_dir = "/".join(cwd.split("/")[:-2])
sys.path.append(src_dir)

from src.fit_model import fit_model
from src.bax.alg.dijkstra import DijkstraNx
from src.experiment_manager import experiment_manager
from src.performance_metrics import NewShortestPathCost
#%%

df_edges = pd.read_csv(f"{cwd}/data/california_edges.csv", index_col="edgeid")
df_nodes = pd.read_csv(f"{cwd}/data/california_nodes.csv", index_col="nodeid")
df_nodes["norm_longitude"] = (df_nodes["longitude"] - df_nodes["longitude"].min()) / (df_nodes["longitude"].max() - df_nodes["longitude"].min())
df_nodes["norm_latitude"] = (df_nodes["latitude"] - df_nodes["latitude"].min()) / (df_nodes["latitude"].max() - df_nodes["latitude"].min())
df_edges_clone = df_edges.copy()
df_edges_clone["start_nodeid"] = df_edges["end_nodeid"]
df_edges_clone["end_nodeid"] = df_edges["start_nodeid"]
df_edges = pd.concat([df_edges, df_edges_clone], ignore_index=True)
#%%


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 # Radius of the Earth in kilometers
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

def euclidean(lon1, lat1, lon2, lat2):
    return math.sqrt((lon1 - lon2)**2 + (lat1 - lat2)**2)

def calculate_work(u, v, mu=10):
    '''
    Args: 
        u, v: (longitude, latitude, elevation)
        mu: coefficient of friction
    '''
    g = 9.81  # gravity in m/s^2
    long_u, lat_u, elev_u = u
    long_v, lat_v, elev_v = v
    # d = haversine(lat_u, long_u, lat_v, long_v) * 1000
    d = euclidean(long_u, lat_u, long_v, lat_v) * 1000
    
    delta_elev = elev_v - elev_u
    W_gravity = g * delta_elev # work due to gravity (unit mass)
    W_friction = mu * g * d # work due to friction
    # print("W_gravity:", W_gravity)
    # print("W_friction:", W_friction)
    total_work = W_gravity + W_friction
    return total_work / 500


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
df_edges["coord"] = list(edge_coords)
edge_work = df_edges["work"].to_numpy()
edge_coord_to_work = {tuple(e): w for e, w in zip(edge_coords, edge_work)}

print(df_edges["work"].describe())


start = 3939
goal = 446
start_coord = (df_nodes.loc[start]["longitude"], df_nodes.loc[start]["latitude"])
goal_coord = (df_nodes.loc[goal]["longitude"], df_nodes.loc[goal]["latitude"])


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


problem_dir = os.path.join(cwd, "results/california_normwork")
bax_dir = os.path.join(problem_dir, "bax_1")
ps_dir = os.path.join(problem_dir, "ps_1")
policy = "ps"
dir = os.path.join(problem_dir, f"{policy}_1")
trial = 1

algo_copy = algo.get_copy()
true_ep, true_output = algo_copy.run_algorithm_on_f(obj_func)
print(true_ep.true_cost)


# plot true path
nodes_true = true_ep.nodes
nodes_coord_true_hav = np.array([
    (df_nodes.loc[u]["longitude"], df_nodes.loc[u]["latitude"])
    for u in nodes_true
])

fig, ax = plt.subplots()

points = ax.scatter(df_nodes["longitude"], df_nodes["latitude"], c=df_nodes["elevation"], s=20, cmap="BuGn")
# start and end
ax.scatter(*start_coord, c="red", s=50, marker="x")
ax.scatter(*goal_coord, c="red", s=50, marker="x")

ax.plot(nodes_coord_true[:, 0], nodes_coord_true[:, 1], color="orange", label="euclidean")
ax.plot(nodes_coord_true_hav[:, 0], nodes_coord_true_hav[:, 1], color="blue", label="haversine")
ax.legend()
ax.set_title(f"True Shortest Path")
fig_dir = os.path.join(problem_dir, "plots")
os.makedirs(fig_dir, exist_ok=True)
# plt.savefig(os.path.join(fig_dir, f"true_path.pdf"))
plt.show()

#%%

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
exe_path_gt, output_gt = algo_gt.run_algorithm_on_f(obj_func)
# x_top_k = np.array(output_gt.x)

exe_path_mf, output_mf = algo.run_algorithm_on_f(lambda x: model.posterior(x).mean)
# x_output = np.array(output_mf.x)

#%%


nodes_gt = exe_path_gt.nodes
nodes_coord_gt = np.array([
    (df_nodes.loc[u]["longitude"], df_nodes.loc[u]["latitude"])
    for u in nodes_gt
])

nodes_mf = exe_path_mf.nodes
nodes_coord_mf = np.array([
    (df_nodes.loc[u]["longitude"], df_nodes.loc[u]["latitude"])
    for u in nodes_mf
])

#%%

# Plot the graph
fig, ax = plt.subplots()

points = ax.scatter(df_nodes["longitude"], df_nodes["latitude"], c=df_nodes["elevation"], s=20, cmap="BuGn")
# start and end
ax.scatter(*start_coord, c="red", s=50, marker="x")
ax.scatter(*goal_coord, c="red", s=50, marker="x")

ax.plot(nodes_coord_gt[:, 0], nodes_coord_gt[:, 1], color="orange", label="GT")
ax.plot(nodes_coord_mf[:, 0], nodes_coord_mf[:, 1], color="blue", label="MF")
ax.legend()
ax.set_title(f"{policy} : GT vs MF")
fig_dir = os.path.join(problem_dir, "plots")
os.makedirs(fig_dir, exist_ok=True)
plt.savefig(os.path.join(fig_dir, f"{policy}_trial{trial}_gtmf.pdf"))
plt.show()

# %%
