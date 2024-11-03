import torch
import numpy as np

from argparse import Namespace

from src.utils import optimize_acqf_and_get_suggested_batch
from .algorithm import Algorithm

class LBFGSB(Algorithm):
    def set_params(self, params):
        super().set_params(params)
        self.params.name = params.get("name", "LBFGSB")
        self.params.opt_mode = params.get("opt_mode", "max")
        self.params.n_dim = params.get("n_dim", None)
        self.params.bounds = params.get("bounds", None)
        self.params.num_restarts = params.get("num_restarts", self.params.n_dim * 5)
        self.params.raw_samples = params.get("raw_samples", self.params.n_dim * 100)


    def initialize(self):
        # super().initialize()
        self.exe_path = Namespace() 

    def run_algorithm_on_f(self, f):
        self.initialize()

        standard_bounds = torch.tensor([[0.0] * self.params.n_dim, [1.0] * self.params.n_dim])
        
        max_post_mean_func = optimize_acqf_and_get_suggested_batch(
            acq_func=f,
            bounds=standard_bounds,
            batch_size=1,
            num_restarts=self.params.num_restarts,
            raw_samples=self.params.raw_samples,
            batch_limit=5,
            init_batch_limit=100,
        )
        self.exe_path.x = max_post_mean_func.numpy() # torch.tensor(1, d)
        self.exe_path.y = f(max_post_mean_func).item()
        return self.exe_path, self.get_output()
    
    def get_output(self):
        return self.exe_path.x

    def execute(self, f):
        self.run_algorithm_on_f(f)
        return self.exe_path.x

    

