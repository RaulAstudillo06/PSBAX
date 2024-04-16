
import torch
import numpy as np


from argparse import Namespace
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling


from .algorithms import Algorithm
from ..util.misc_util import dict_to_namespace



class PymooAlgorithm(Algorithm):
    def set_params(self, params):
        super().set_params(params)
        params = dict_to_namespace(params)
        self.params.name = getattr(params, "name", "NSGA2")
        self.params.n_dim = getattr(params, "n_dim", None)
        self.params.n_obj = getattr(params, "n_obj", None)
        self.params.n_gen = getattr(params, "n_gen", 100)
        self.params.n_offsprings = getattr(params, "n_offsprings", 100)
        self.params.pop_size = getattr(params, "pop_size", 40)

    def initialize(self):
        # If need full exe_path, refer to https://pymoo.org/getting_started/part_4.html
        self.exe_path = Namespace() # Not the full execution path, just output + function (sample) values
        self.pymoo_algorithm = NSGA2(
            pop_size=self.params.pop_size,
            n_offsprings=self.params.n_offsprings,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )

    def pymoo_problem_functor(self, f):
        class MyProblem(ElementwiseProblem):
            def __init__(self_):
                super().__init__(
                    n_var=self.params.n_dim,
                    n_obj=self.params.n_obj,
                    n_constr=0,
                    xl=np.array([0] * self.params.n_dim),
                    xu=np.array([1] * self.params.n_dim),
                    elementwise_evaluation=True,
                )

            def _evaluate(self_, x, out, *args, **kwargs):
                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(x)
                if len(x.shape) == 1:
                    x = x.view(torch.Size([1, x.shape[0]]))
                f_values = f(x).detach().numpy()
                out["F"] = f_values.flatten()
        return MyProblem()
    
    def run_algorithm_on_f(self, f):
        self.initialize()
        problem = self.pymoo_problem_functor(f)
        result = minimize(
            problem,
            self.pymoo_algorithm,
            ("n_gen", self.params.n_gen),
            save_history=False,
            verbose=False,
        )
        X, F = result.opt.get("X", "F")
        self.exe_path.x = X # X is a numpy array
        self.exe_path.y = F # F is a numpy array
        return X, F
    
    def get_output(self):
        return self.exe_path

    def execute(self, f):
        self.run_algorithm_on_f(f)
        return self.exe_path.x
