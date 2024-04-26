
import torch
import numpy as np


from argparse import Namespace
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning
)


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
        # self.pymoo_algorithm = NSGA2(
        #     pop_size=self.params.pop_size,
        #     n_offsprings=self.params.n_offsprings,
        #     sampling=FloatRandomSampling(),
        #     crossover=SBX(prob=0.9, eta=15),
        #     mutation=PM(eta=20),
        #     eliminate_duplicates=True
        # )
        self.pymoo_algorithm = NSGA2(
            pop_size=self.params.pop_size,
            n_offsprings=self.params.n_offsprings,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
            seed=np.random.randint(0, 1000000),
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


class HypervolumeAlgorithm(Algorithm):
    def set_params(self, params):
        super().set_params(params)
        params = dict_to_namespace(params)
        self.params.name = getattr(params, "name", "NSGA2")
        self.params.n_dim = getattr(params, "n_dim", None)
        self.params.n_obj = getattr(params, "n_obj", None)
        self.params.n_gen = getattr(params, "n_gen", 100)
        self.params.n_offsprings = getattr(params, "n_offsprings", 100)
        self.params.pop_size = getattr(params, "pop_size", 500)
        self.params.num_runs = getattr(params, "num_runs", 10)
        self.params.opt_mode = getattr(params, "opt_mode", "minimize")
        self.params.ref_point = getattr(params, "ref_point", None)
        self.params.output_size = getattr(params, "output_size", 50)


    def initialize(self):
        self.exe_path = Namespace()
        self.weight = -1 if self.params.opt_mode == "maximize" else 1

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
                # NOTE: check if this is correct
                self_.weight = self.weight

            def _evaluate(self_, x, out, *args, **kwargs):
                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(x)
                if len(x.shape) == 1:
                    x = x.view(torch.Size([1, x.shape[0]]))
                f_values = self_.weight * f(x).detach().numpy()

                out["F"] = f_values.flatten()
        return MyProblem()
    
    def pareto_solver(self, f):
        problem = self.pymoo_problem_functor(f)
        pareto_set, pareto_front = [], []

        for i in range(self.params.num_runs):
            
            algorithm=NSGA2(
                pop_size=self.params.pop_size,
                n_offsprings=self.params.n_offsprings,
                sampling=FloatRandomSampling(),
                crossover=SBX(prob=0.9, eta=15),
                mutation=PM(eta=20),
                eliminate_duplicates=True,
                seed=np.random.randint(0, 1000000),
            )
            result = minimize(
                problem,
                algorithm,
                ("n_gen", self.params.n_gen),
                save_history=False,
                verbose=False,
            )
            X, F = result.opt.get("X", "F")
            pareto_set.append(X)
            pareto_front.append(F)
        
        pareto_set = np.vstack(pareto_set)
        pareto_front = self.weight * np.vstack(pareto_front)
        return pareto_set, pareto_front
        
    def run_algorithm_on_f(self, f):
        self.initialize()
        aug_pareto_set, aug_pareto_front = self.pareto_solver(f)
        indices = self.hv_truncation(
            aug_pareto_front,
        )
        self.exe_path.x = aug_pareto_set[indices]
        self.exe_path.y = aug_pareto_front[indices]
        return self.exe_path.x, self.exe_path.y

    def hv_truncation(
        self,
        sample_pf,
    ):  
        '''
        Args:
            output_size: The desired number of Pareto points.
            sample_pf: A `P x M`-dim Numpy array containing the oversampled Pareto front .
        '''
        M = sample_pf.shape[-1]
        indices = []
        sample_pf = torch.from_numpy(sample_pf)
        ref_point = torch.from_numpy(self.params.ref_point)

        for k in range(self.params.output_size):
            if k == 0:
                hypercell_bounds = torch.zeros(2, M)
                hypercell_bounds[0] = ref_point
                hypercell_bounds[1] = 1e+10
            else:
                partitioning = FastNondominatedPartitioning(
                    ref_point=ref_point, Y=fantasized_pf
                )

                hypercell_bounds = partitioning.hypercell_bounds

            # `1 x num_boxes x M`
            lo = hypercell_bounds[0].unsqueeze(0)
            up = hypercell_bounds[1].unsqueeze(0)
            # compute hv
            hvi = torch.max(
                torch.min(sample_pf.unsqueeze(-2), up) - lo,
                torch.zeros(lo.shape)
            ).prod(dim=-1).sum(dim=-1)

            # zero out pending points
            hvi[indices] = 0
            # store update
            am = torch.argmax(hvi).tolist()
            indices = indices + [am]

            if k == 0:
                fantasized_pf = sample_pf[am:am + 1, :]
            else:
                fantasized_pf = torch.cat([fantasized_pf, sample_pf[am:am + 1, :]],
                                        dim=0)

        # indices = torch.tensor(indices)
        return indices
        
    def get_output(self):
        return self.exe_path

    def execute(self, f):
        self.run_algorithm_on_f(f)
        return self.exe_path.x
    


