import torch
import numpy as np

from botorch.acquisition.analytic import PosteriorMean

from argparse import Namespace
from .algorithms import Algorithm
from ..util.misc_util import dict_to_namespace

class TopKTorch(Algorithm):
    def set_params(self, params):
        super().set_params(params)
        params = dict_to_namespace(params)

        self.params.name = getattr(params, "name", "TopKTorch")
        self.params.k = getattr(params, "k", 10)
        self.params.x_path = getattr(params, "x_path", None) # torch.Tensor
        self.params.no_copy = getattr(params, "no_copy", None)

    def initialize(self):
        self.exe_path = Namespace()

    def run_algorithm_on_f(self, f):
        self.initialize()
        if isinstance(f, PosteriorMean):
            x_path = self.params.x_path.unsqueeze(1)
        else:
            x_path = self.params.x_path
        fx = eval_in_batch(f, x_path).detach().squeeze()

        idx = torch.topk(fx, self.params.k, largest=True).indices
        self.exe_path.idx = idx.tolist()
        self.exe_path.x = self.params.x_path[idx] # torch.tensor (10, 80)
        self.exe_path.y = (fx[idx]).reshape(-1, 1) # torch.tensor (10, 1)

        # f[idx] = 1
        # f[~idx] = 0
        return self.exe_path, self.exe_path.x
    
    def execute(self, f):
        self.run_algorithm_on_f(f)
        return self.exe_path.x
    
    def get_output(self):
        return self.exe_path.x
    

def eval_in_batch(f, x_set, max_batch_size=100):
    return torch.cat([f(X_) for X_ in x_set.split(max_batch_size)])
