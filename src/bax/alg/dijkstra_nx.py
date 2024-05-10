import torch
import numpy as np
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
        self.params.df_edges = getattr(params, "df_edges", None)
        # self.params.weight = getattr(params, "weight", "pos_weight")
        self.G = None
        self.edge_positions = self.params.df_edges[["norm_longitude", "norm_latitude"]].to_numpy()

    @staticmethod
    def softplus_np(x):
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
        
    def initialize(self):
        self.exe_path = Namespace()
        if self.G is None:
            self.G = nx.from_pandas_edgelist(self.params.df_edges, source="start_nodeid", target="end_nodeid", edge_attr=True)

    def run_algorithm_on_f(self, f):
        self.initialize()
        if isinstance(f, PosteriorMean):
            X = torch.from_numpy(self.edge_positions).unsqueeze(1)
        else:
            X = torch.from_numpy(self.edge_positions)
        f_values = f(X).detach().squeeze().numpy()
        edge_pos_to_f = {tuple(e): w for e, w in zip(self.edge_positions, f_values)}
        def weight_func(u, v, d):
            lon, lat = self.G[u][v]["norm_longitude"], self.G[u][v]["norm_latitude"]
            # x = torch.tensor([lon, lat]).reshape(1, -1)
            fx = edge_pos_to_f[(lon, lat)]
            return self.softplus_np(fx) # use softplus to ensure all edge weights are positive
        P = nx.dijkstra_path(self.G, self.params.start, self.params.end, weight=weight_func)
        # get edge id from self.params.df_edges where start_nodeid = P[i] and end_nodeid = P[i+1]
        edge_pos = []
        edges_weights = []
        true_cost = 0
        for i in range(len(P)-1):
            lon, lat = self.G[P[i]][P[i+1]]["norm_longitude"], self.G[P[i]][P[i+1]]["norm_latitude"]
            edge_pos.append([lon, lat])
            edges_weights.append(weight_func(P[i], P[i+1], None))
            true_cost += self.G[P[i]][P[i+1]]["pos_weight"]
        
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