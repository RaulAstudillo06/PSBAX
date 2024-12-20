import torch
import numpy as np

from botorch.acquisition.analytic import PosteriorMean

from argparse import Namespace
from .algorithm import Algorithm
# from ..util.misc_util import dict_to_namespace

class LevelSetEstimator(Algorithm):
    def set_params(self, params):
        super().set_params(params)
        # params = dict_to_namespace(params)
        # self.params.name = getattr(params, "name", "SimpleLevelSet")
        self.params.name = params.get("name", "SimpleLevelSet")
        self.params.threshold = params.get("threshold", 10)
        self.params.x_set = params.get("x_set", None)
        self.params.no_copy = params.get("no_copy", None)

    def initialize(self):
        self.exe_path = Namespace()

    def run_algorithm_on_f(self, f):
        self.initialize()
        if self.params.name == "SimpleLevelSet":
            if isinstance(f, PosteriorMean):
                x_set = torch.from_numpy(self.params.x_set).unsqueeze(1)
                # fx = f(x_set).detach().numpy().flatten()
                fx = eval_in_batch(f, x_set).detach().numpy().flatten()
            else:
                x_set = self.params.x_set
                fx = f(x_set).numpy().flatten()
            
        idx = np.where(fx > self.params.threshold)[0]
        if len(idx) == 0:
            idx = np.argmax(fx)
            self.exe_path.idx = [idx]
        else:
            self.exe_path.idx = list(idx)
        self.exe_path.x = np.atleast_2d(self.params.x_set[idx])
        self.exe_path.y = np.atleast_1d(fx[idx])

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
