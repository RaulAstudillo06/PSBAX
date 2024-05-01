import torch
import numpy as np

from argparse import Namespace

from src.utils import optimize_acqf_and_get_suggested_batch
from .algorithms import Algorithm
from ..util.misc_util import dict_to_namespace

class LBFGSB(Algorithm):
    def set_params(self, params):
        super().set_params(params)
        params = dict_to_namespace(params)
        self.params.name = getattr(params, "name", "LBFGSB")
        self.params.opt_mode = getattr(params, "opt_mode", "max")
        self.params.n_dim = getattr(params, "n_dim", None)
        self.params.bounds = getattr(params, "bounds", None)
        self.params.num_restarts = getattr(params, "num_restarts", self.params.n_dim * 6)
        self.params.raw_samples = getattr(params, "raw_samples", self.params.n_dim * 180)


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
        )
        self.exe_path.x = max_post_mean_func.numpy() # torch.tensor(1, d)
        self.exe_path.y = f(max_post_mean_func).item()
        return self.exe_path, self.get_output()
    
    def get_output(self):
        return self.exe_path.x

    def execute(self, f):
        self.run_algorithm_on_f(f)
        return self.exe_path.x

    

