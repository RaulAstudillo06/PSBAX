import torch
import numpy as np

from argparse import Namespace

from .algorithms import Algorithm
from ..util.misc_util import dict_to_namespace


class SubsetSelect(Algorithm):
    def set_params(self, params):
        """Set self.params, the parameters for the algorithm."""
        # params : x, df index
        super().set_params(params)
        params = dict_to_namespace(params)
        self.params.name = getattr(params, "name", "SubsetSelect") # use params to be consistent with BAX
        self.params.k = getattr(params, "k", 3)

    def initialize(self):
        self.exe_path = Namespace(x=[], y=[], idxes=[], values=[])
        self.x_torch = self.obj_func.get_x()

    def index_to_x(self, idx):
        return self.obj_func.get_x(idx)

    def set_obj_func(self, obj_func):
        self.obj_func = obj_func
        
    def run_algorithm_on_f(self, f):
        self.initialize()
        if not isinstance(f, np.ndarray):
            x_torch = self.x_torch.unsqueeze(1)
            fx = f(x_torch).detach().numpy() # (n,)
        else:
            fx = f
        n = len(fx)

        values = self.obj_func.get_noisy_f_lst(fx)
        mean_values = np.mean(values, axis=0)
        mx = self.random_argmax(mean_values)
        idxes = [mx]

        for _ in range(self.params.k - 1):
            e_vals = np.zeros(n)
            for j in range(n):
                test_idxes = idxes
                if j not in idxes:
                    test_idxes = idxes + [j]                             
                    test_idxes = np.asarray(test_idxes)
                    e_vals[j] = np.mean(np.max(values[:, test_idxes], axis=-1))
                
            idx_next = self.random_argmax(e_vals)
            idxes.append(idx_next)
            # self.exe_path.y.append(e_vals[idx_next])

        indices = [self.obj_func.df.index[i] for i  in idxes] # list of string indices
        self.exe_path.x = self.x_torch[idxes].numpy() # (k, d)
                                                      
        self.exe_path.values = values
        self.exe_path.idxes = idxes
        self.exe_path.y = fx[idxes]
        return self.exe_path, indices
    
    def get_values_and_selected_indices(self):
        return self.exe_path.values, self.exe_path.idxes
    
    def execute(self, f):
        '''
        Returns:
            output: list of indices of selected points
        '''
        _, output = self.run_algorithm_on_f(f)
        return output
    
    def get_output(self):
        pass
    
    @staticmethod
    def random_argmax(vals):
        max_val = np.max(vals)
        idxes = np.where(vals == max_val)[0]
        return np.random.choice(idxes)
