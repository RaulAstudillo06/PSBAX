import copy
import numpy as np
import torch
from argparse import Namespace
from abc import ABC, abstractmethod


class Algorithm(ABC):
    """Base class for a BAX Algorithm"""

    def __init__(self, params=None, verbose=False):
        """
        Parameters
        ----------
        params : Namespace_or_dict
            Namespace or dict of parameters for this model.
        verbose : bool
            If True, print description string.
        """
        self.verbose_init_arg = verbose
        self.set_params(params)

    def set_params(self, params):
        """Set self.params, the parameters for the algorithm."""
        params = dict_to_namespace(params)
        self.params = Namespace()
        self.params.name = getattr(params, "name", "Algorithm")

    def initialize(self):
        """Initialize algorithm, reset execution path."""
        self.exe_path = Namespace(x=[], y=[])

    def get_next_x(self):
        """
        Given the current execution path, return the next x in the execution path. If
        the algorithm is complete, return None.
        """
        # Default behavior: return a uniform random value 10 times
        next_x = np.random.uniform() if len(self.exe_path.x) < 10 else None
        return next_x

    # def take_step(self, f):
    #     """Take one step of the algorithm.
    #     Args:
    #         f: Function to query. Function takes in torch tensor of shape (1, input_dim).
        
    #     """
    #     x = self.get_next_x() # NOTE: x is a list / np.array(d, )
    #     if x is not None:
    #         y = f(torch.tensor(x).unsqueeze(0)).item()
    #         self.exe_path.x.append(x)
    #         self.exe_path.y.append(y)

    #     return x

    # def run_algorithm_on_f(self, f):
    #     """
    #     Run the algorithm by sequentially querying function f. Return the execution path
    #     and output.
    #     """
    #     self.initialize()

    #     # Step through algorithm
    #     x = self.take_step(f)
    #     while x is not None:
    #         x = self.take_step(f)

    #     # Return execution path and output
    #     return self.exe_path, self.get_output()

    # def get_exe_path_crop(self):
    #     """
    #     Return the minimal execution path for output, i.e. cropped execution path,
    #     specific to this algorithm.
    #     """
    #     # As default, return untouched execution path
    #     return self.exe_path

    def get_copy(self):
        """Return a copy of this algorithm."""
        return copy.deepcopy(self)

    # @abstractmethod
    # def get_output(self):
    #     """Return output based on self.exe_path."""
    #     pass
    
    def execute(self, f):
        """Execute the algorithm on function f.
        Args:
            f: Function to query. Function takes in torch tensor of shape (1, input_dim).
        Returns:
            Output of algorithm.
        """
        exe_path, output = self.run_algorithm_on_f(f)
        return output
    

def dict_to_namespace(params):
    # If params is a dict, convert to Namespace
    if isinstance(params, dict):
        params = Namespace(**params)
    return params