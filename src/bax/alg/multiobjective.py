
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
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import sample_simplex
from botorch.generation.gen import get_best_candidates
from botorch.models.model import Model
from botorch.utils.gp_sampling import get_gp_samples
from botorch.models.deterministic import GenericDeterministicModel
from botorch.optim.optimize import optimize_acqf

# from .alg_utils import get_function_samples, optimize_acqf_and_get_suggested_batch
# from src.utils.utils import optimize_acqf_and_get_suggested_query
from torch import Tensor
from typing import Optional


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
    
class ScalarizedFunction(AcquisitionFunction):
    r""" """

    def __init__(
        self,
        model: Model,
        f,
        objective: Optional[MCAcquisitionObjective] = None,
    ) -> None:
        r""" """
        super().__init__(model=model)
        self.objective = objective
        self.f = f

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r""" """
        fx = self.f(X).view(1, -1) # need to be `sample_shape x batch_shape x q x m` to pass to self.objective
        scalarized_posterior_mean = self.objective(fx)
        return scalarized_posterior_mean

class ScalarizedParetoSolver(Algorithm):
    def set_params(self, params):
        super().set_params(params)
        params = dict_to_namespace(params)
        self.params.name = getattr(params, "name", "ScalarizedParetoSolver")
        self.params.n_dim = getattr(params, "n_dim", None)
        self.params.n_obj = getattr(params, "n_obj", None)
        self.params.num_runs = getattr(params, "num_runs", 10)
        self.params.set_size = getattr(params, "set_size", 50)
        self.params.opt_mode = getattr(params, "opt_mode", "maximize") # always maximizing
    
    def initialize(self):
        self.exe_path = Namespace()
    
    def set_model(self, model):
        self.model = model

    def run_algorithm_on_f(self, f):
        self.initialize()
        query = []
        mean_train_inputs = self.model.posterior(self.model.train_inputs[0][0]).mean.detach()
        for _ in range(self.params.set_size):
            # model_rff_sample = get_function_samples(model=self.model)
            weights = sample_simplex(mean_train_inputs.shape[-1]).squeeze()
            chebyshev_scalarization = GenericMCObjective(
                get_chebyshev_scalarization(weights=weights, Y=mean_train_inputs)
            )
            acquisition_function = ScalarizedFunction(
                model=self.model, f=f, objective=chebyshev_scalarization
            )
            standard_bounds = torch.tensor(
            [[0.0] * self.params.n_dim, [1.0] * self.params.n_dim]
            )  # This assumes the input domain has been normalized beforehand
            x_next = optimize_acqf_and_get_suggested_batch(
                acq_func=acquisition_function,
                bounds=standard_bounds,
                batch_size=1,
                num_restarts=5 * self.params.n_dim,
                raw_samples=100 * self.params.n_dim,
                batch_limit=5,
                init_batch_limit=100,
            )

            query.append(x_next.clone())
        
        query = torch.cat(query, dim=-2)
        y_vals = f(query.unsqueeze(1)).detach().squeeze()

        self.exe_path.x = np.array(query)
        self.exe_path.y = np.array(y_vals)
        return self.exe_path, self.exe_path.x
    
    def get_output(self):
        return self.exe_path.x
    
    def execute(self, f):
        self.run_algorithm_on_f(f)
        return self.exe_path.x
    


def optimize_acqf_and_get_suggested_batch(
    acq_func: AcquisitionFunction,
    bounds: Tensor,
    batch_size: int,
    num_restarts: int,
    raw_samples: int,
    batch_initial_conditions: Optional[Tensor] = None,
    batch_limit: Optional[int] = 2,
    init_batch_limit: Optional[int] = 30,
) -> Tensor:
    """Optimizes the acquisition function and returns the (approximate) optimum."""

    candidates, acq_values = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        batch_initial_conditions=batch_initial_conditions,
        options={
            "batch_limit": batch_limit,
            "init_batch_limit": init_batch_limit,
            "maxiter": 100,
            "nonnegative": False,
            "method": "L-BFGS-B",
        },
        return_best_only=False,
    )
    candidates = candidates.detach()
    new_x = get_best_candidates(batch_candidates=candidates, batch_values=acq_values)
    return new_x