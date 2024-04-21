#%%
import os
import torch


tkwargs = {
    "dtype": torch.float64,
    "device": torch.device("cpu"),
}

import botorch
from botorch.test_functions.multi_objective import DTLZ1

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP, ModelListGP
from botorch.sampling import SobolQMCNormalSampler
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.gp_sampling import get_gp_samples
from botorch.acquisition.analytic import PosteriorMean
from botorch.models.deterministic import GenericDeterministicModel

import numpy as np
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.indicators.hv import HV
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

#%%

x_dim = 6
num_objectives = 3

problem = get_problem(
    # "zdt1", 
    "dtlz1",
    # "dtlz2",
    n_var=x_dim,
    n_obj=num_objectives,
    # xl=np.array([0] * 5),
    # xu=np.array([1] * 5),
)

obj_func = botorch.test_functions.multi_objective.DTLZ1(
    dim=x_dim,
    num_objectives=num_objectives,
)

ref_point = np.array([obj_func._ref_val] * num_objectives)

# The pareto front of a scaled zdt1 problem
pf = problem.pareto_front()

algorithm = NSGA2(
    pop_size=10,
    n_offsprings=10,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=10),
    eliminate_duplicates=True,
)

result = minimize(
    problem,
    algorithm,
    ("n_gen", 50),
    save_history=True,
    verbose=False,
)

X, F = result.opt.get("X", "F")


ind = HV(ref_point=ref_point)
print("HV found", ind(F))
print("HV PF", ind(pf))


#%%
# Example data - assuming ModelListGP is already trained with your data
train_x = torch.rand(10, x_dim).to(**tkwargs)  # 10 training points, 2-dimensional input
train_y = obj_func(train_x)  # (10, 2)

models = []
for f_i in range(train_y.shape[-1]):
    x_gp = train_x
    y_gp = train_y[:, f_i].view(-1, 1)
    model = SingleTaskGP(
        train_X=x_gp,
        train_Y=y_gp,
        outcome_transform=Standardize(m=y_gp.shape[-1]),
    ).to(**tkwargs)
    models.append(model)
for model in models:
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
model = ModelListGP(*models).to(**tkwargs)


n = 5
q = 4
test_x = torch.rand(n, q, x_dim).to(**tkwargs) # 5 batches, q in batch, 2-dimensional input


posterior = model.posterior(test_x)
cov = posterior.mvn.covariance_matrix # (n, 2q, 2q)
# cov = cov.reshape(n, q, q, n_obj, n_obj)


#%%
import matplotlib.pyplot as plt
# Plot the covariance matrix
plt.imshow(np.log(cov[0].detach().numpy()))
#%%

gp_samples = []
for m in model.models:
    gp_samples.append(
        get_gp_samples(
            model=m,
            num_outputs=1,
            n_samples=1,
            num_rff_features=512,
            )
    )

def aux_func(X):
    val = []
    for gp_sample in gp_samples:
        val.append(gp_sample.posterior(X).mean)
    return torch.cat(val, dim=-1)

obj_func_sample = GenericDeterministicModel(f=aux_func).to(**tkwargs)
    
#%%

test_x = torch.rand(3, dim).to(**tkwargs)
print(obj_func_sample(test_x))

#%%
import pymoo
import numpy as np
from pymoo.core.problem import ElementwiseProblem

class MyProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var=dim,
            n_obj=2,
            n_constr=0,
            xl=np.array([0] * dim),
            xu=np.array([1] * dim),
            elementwise_evaluation=True,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, **tkwargs)

        f_values = obj_func_sample(x).detach().numpy()
        out["F"] = f_values.flatten()  # Ensure correct shape for pymoo

problem = MyProblem()
#%%
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2


algorithm = NSGA2(pop_size=40)

# result = minimize(problem,
#                   algorithm,
#                   ('n_gen', 100),
#                   verbose=True)


result = minimize(
    problem,
    algorithm,
    ("n_gen", 100),
    save_history=True,
    verbose=False
)

X, F = result.opt.get("X", "F")


                
#%%

from pymoo.indicators.hv import HV

# Assuming result.F contains the objective values corresponding to result.X
ind = HV(ref_point=np.array([1.0, 1.0]))

# Calculate the hypervolume
ind(result.F)

#%%
class MyProblem2(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var=dim,
            n_obj=2,
            n_constr=0,
            xl=np.array([0] * dim),
            xu=np.array([1] * dim),
            elementwise_evaluation=True,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, **tkwargs)

        f_values = obj_func_sample(x).detach().numpy()
        print(f"X: {x}, F: {f_values}")
        out["F"] = f_values  # Ensure correct shape for pymoo

problem2 = MyProblem2()

test_x = torch.rand(5, dim).to(**tkwargs)
problem2.evaluate(test_x)
#%%