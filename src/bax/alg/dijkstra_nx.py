import torch
import numpy as np
import math
import networkx as nx

from botorch.acquisition.analytic import PosteriorMean
from argparse import Namespace
from .algorithms import Algorithm
from ..util.misc_util import dict_to_namespace

class DijkstraNx(Algorithm):
    
    def set_params(self, params):
        super().set_params(params)
        params = dict_to_namespace(params)
        self.params.name = getattr(params, "name", "DijkstraNx")
        self.params.start = getattr(params, "start", None)
        self.params.end = getattr(params, "end", None)
        self.params.softplus = getattr(params, "softplus", True)
        self.df_edges = getattr(params, "df_edges", None)
        self.df_nodes = getattr(params, "df_nodes", None)
        # self.params.weight = getattr(params, "weight", "pos_weight")
        G = nx.from_pandas_edgelist(self.df_edges, source="start_nodeid", target="end_nodeid", edge_attr=None)
        self.G = nx.DiGraph(G)
        self.edge_coords = np.vstack(self.df_edges["coord"])

    def initialize(self):
        self.exe_path = Namespace()

    def run_algorithm_on_f(self, f):
        self.initialize()
        if isinstance(f, PosteriorMean):
            X = torch.from_numpy(self.edge_coords).unsqueeze(1)
        else:
            X = torch.from_numpy(self.edge_coords)

        G_copy = self.G.copy()
        f_values = f(X).detach().squeeze().numpy()
        edge_pos_to_f = {tuple(e): w for e, w in zip(self.edge_coords, f_values)}

        def weight_func(u, v):
            coord = (
                self.df_nodes.loc[u]["norm_longitude"], 
                self.df_nodes.loc[u]["norm_latitude"],
                self.df_nodes.loc[v]["norm_longitude"], 
                self.df_nodes.loc[v]["norm_latitude"]
            )
            fx = edge_pos_to_f[coord]
            return fx 
        for u, v, data in G_copy.edges(data=True):
            if self.params.softplus:
                data["work"] = softplus_np(weight_func(u, v))
            else:
                data["work"] = weight_func(u, v)
                
        # use dijkstra algorithm when softplus is used
        if self.params.softplus:
            P = nx.dijkstra_path(G_copy, self.params.start, self.params.end, weight="work")
        else:
            P = nx.bellman_ford_path(self.G, self.params.start, self.params.end, weight="work")
        # get edge id from self.params.df_edges where start_nodeid = P[i] and end_nodeid = P[i+1]
        edge_pos = []
        edges_weights = []
        true_cost = 0
        for i in range(len(P)-1):
            u, v = P[i], P[i+1]
            coord = (
                self.df_nodes.loc[u]["norm_longitude"], 
                self.df_nodes.loc[u]["norm_latitude"],
                self.df_nodes.loc[v]["norm_longitude"], 
                self.df_nodes.loc[v]["norm_latitude"]
            )
            edge_pos.append(coord)
            edges_weights.append(edge_pos_to_f[coord])
            edge = self.df_edges[
                (self.df_edges["start_nodeid"] == u) & (self.df_edges["end_nodeid"] == v)
            ]
            true_cost += edge["work"].values[0]
        
        self.exe_path.nodes = P
        self.exe_path.x = np.array(edge_pos)
        self.exe_path.y = np.array(edges_weights)
        self.exe_path.true_cost = true_cost

        return self.exe_path, self.get_output()

    def get_output(self):
        return self.exe_path.x
    
    def execute(self, f):
        self.run_algorithm_on_f(f)
        return self.exe_path.x
    
def softplus_np(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 # Radius of the Earth in kilometers
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

def calculate_work(u, v, mu=0.1):
    '''
    Args: 
        u, v: (longitude, latitude, elevation)
        mu: coefficient of friction
    '''
    g = 9.81  # gravity in m/s^2
    long_u, lat_u, elev_u = u
    long_v, lat_v, elev_v = v
    d = haversine(lat_u, long_u, lat_v, long_v) * 1000
    
    delta_elev = elev_v - elev_u
    W_gravity = g * delta_elev # work due to gravity (unit mass)
    W_friction = mu * g * d # work due to friction
    # print("W_gravity:", W_gravity)
    # print("W_friction:", W_friction)
    total_work = W_gravity + W_friction
    return total_work