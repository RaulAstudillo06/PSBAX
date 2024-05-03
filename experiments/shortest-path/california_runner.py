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


from src.bax.alg.dijkstra import Dijkstra
from src.bax.util.domain_util import unif_random_sample_domain
from src.bax.util.graph import make_grid, edges_of_path, positions_of_path, area_of_polygons
from src.bax.util.graph import make_vertices, make_edges

from src.experiment_manager import experiment_manager
from src.performance_metrics import ShortestPathCost, ShortestPathArea, DijkstraBAXMetric

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

df_edges = pd.read_csv("./data/california/ba_edges_new.csv")
df_nodes = pd.read_csv("./data/california/ba_edges_new.csv")

df_edges["elevation"] = (df_edges["elevation_y"] + df_edges["elevation_x"]) / 2
edge_elevation = df_edges["elevation"].to_numpy()

# rescaling
df_nodes["elevation"] = (df_nodes["elevation_y"] + df_nodes["elevation_x"]) / 2
node_elevations = df_nodes["elevation"] / edge_elevation.max() + 0.1

edge_elevation = edge_elevation / edge_elevation.max()  # scale between [0, 1]
edge_elevation = (
    edge_elevation + 0.1
)  # make strictly positive to prevent inv_softmax blowup
edge_positions = df_edges[["mean_longitude", "mean_latitude"]].to_numpy()


def normalize(data, scale):
    data = data - data.min(0, keepdims=True)
    return data / scale


# TODO: check lims
x1_lims = (-123, -119)
x2_lims = (36.8, 39.1)
# TODO: rescale data x1,x2 and y
total_area = (x1_lims[1] - x1_lims[0]) * (x2_lims[1] - x2_lims[0])


# normalize both x and y by longitude
xy_normalization = edge_positions[:, 0].max() - edge_positions[:, 0].min()
edge_positions = normalize(edge_positions, xy_normalization)
print(xy_normalization)

edge_tuples = [tuple(e) for e in df_edges[["start_nodeid", "end_nodeid"]].to_numpy()]
edge_tuples = edge_tuples + [(v, u) for (u, v) in edge_tuples]
# make undirected
edge_to_elevation = dict(
    zip(
        edge_tuples,
        np.concatenate([edge_elevation, edge_elevation]),
    )
)
edge_to_position = dict(
    zip(
        edge_tuples,
        np.concatenate([edge_positions, edge_positions]),
    )
)
edge_position_to_elevation = dict(
    zip([tuple(p) for p in edge_positions], edge_elevation)
)
# right now we use the original node elevation to plot nodes but run experiment
# using scaled elevation
positions = df_nodes[["mean_longitude", "mean_latitude"]].to_numpy()
positions = normalize(positions, xy_normalization)
edge_nodes = df_edges[["start_nodeid", "end_nodeid"]].to_numpy()

has_edge = np.zeros((len(positions), len(positions)))
# make undirected edges
has_edge[edge_nodes[:, 0], edge_nodes[:, 1]] = 1
has_edge[edge_nodes[:, 1], edge_nodes[:, 0]] = 1
#%%

def true_f(x):
    assert x.ndim == 1
    return edge_position_to_elevation[tuple(x)]


def inv_softplus(x):
    return x + np.log(-np.expm1(-x))  # numerical stability
    # return np.log(np.exp(x) - 1)

# NOTE: this is the function we will use
def true_latent_f(x_y):
    return inv_softplus(true_f(x_y))

def softplus(x):
    return np.log1p(np.exp(x))

def cost_func(u, v, f, latent_f=True):
    edge = (u.index, v.index)
    edge_pos = edge_to_position[edge]
    edge_cost = f(edge_pos)
    if latent_f:
        return softplus(edge_cost), [edge_pos], [edge_cost]
    else:
        # cost_func(u, v, true_f, latent_f=False)
        return edge_cost, [edge_pos], [edge_cost]

def true_cost_func(u, v):
    '''Equivalent to cost_func(u, v, true_f, latent_f=False)
    '''
    edge = (u.index, v.index)
    edge_cost = edge_to_elevation[edge]
    return edge_cost, [edge], [edge_cost]


def cost_of_path(path, cost_func):
    cost = 0
    for i in range(len(path) - 1):
        cost += cost_func(path[i], path[i + 1])[0]  # index to get edge_cost
    return cost

def obj_func(X):
    '''What do I do about rescaling?'''
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    y = torch.zeros(X.shape[0])
    for i in range(X.shape[0]):
        # y[i] = true_f(X[i])
        y[i] = edge_position_to_elevation[tuple(X[i])]

    # return torch.log(torch.exp(y) - 1)
    return y + torch.log(-torch.expm1(-y))

    

algo_params = {
    "cost_func": lambda u, v, f: cost_func(u, v, f, latent_f=True),
    "true_cost": lambda u, v: cost_func(u, v, obj_func, latent_f=True),
    # "true_cost": lambda u, v: true_cost_func(u, v),
    # "true_cost": lambda u, v: cost_func(u, v, true_f, latent_f=False),
}

vertices = make_vertices(positions, has_edge)
edges = make_edges(vertices)
start = vertices[3939]  # ~ Santa Cruz
goal = vertices[446]  # ~ Lake Tahoe
# components = connected_components(vertices)
# components = [[v.index for v in c] for c in components]
# # make sure start and goal are in the same connected component
# assert any(all((start.index in c, goal.index in c)) for c in components)

algo = Dijkstra(
    params=algo_params,
    vertices=vertices,
    start=start,
    goal=goal,
    edge_to_position=edge_to_position,
    node_representation="indices",
    verbose=False,
)

# performance_metrics = [
#     ShortestPathCost(algo),
#     ShortestPathArea(algo, obj_func),
# ]
algo_copy = algo.get_copy()
true_ep, true_output = algo_copy.run_algorithm_on_f(obj_func)
# optimum_cost = true_output[0] = 72.80633717926638

performance_metrics = [
    DijkstraBAXMetric(
        algo, 
        obj_func, 
        n_samples=20, 
        total_area=total_area,
        optimum_cost = true_output[0],
    )
]

input_dim = 2
# Nodes are indices
experiment_manager(
    problem="california_bax1",
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

