#%%
import numpy as np
import torch
import os
import sys

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
# debug._set_state(False)

# script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
cwd = os.getcwd()
src_dir = "/".join(cwd.split("/")[:-2]) # src directory is two levels up
sys.path.append(src_dir)


from src.bax.alg.levelset import LevelSetEstimator
from src.performance_metrics import F1Score
from src.experiment_manager import experiment_manager
from src.fit_model import fit_model
from src.utils import (
    generate_initial_data,
    generate_random_points,
    get_obj_vals,
    seed_torch,
    optimize_acqf_and_get_suggested_batch,
    reshape_mesh, 
    get_mesh
)

#%%



dim = 2
steps = 50
tau = 0.5

def get_threshold(f, tau, n=10000):
    x_test = torch.rand(n, dim)
    f_test = f(x_test)
    f_test_sorted, _ = torch.sort(f_test, descending=False)
    idx = int(tau * len(f_test_sorted))
    threshold = f_test_sorted[idx]
    return threshold.item()


bounds = [-6, 6]
def himmelblau(X: torch.Tensor, minimize=False) -> torch.Tensor:
    X = (bounds[1] - bounds[0]) * X + bounds[0]
    a = X[:, 0]
    b = X[:, 1]
    result = (a ** 2 + b - 11) ** 2 + (a + b ** 2 - 7) ** 2
    if not minimize:
        return -result
    return result

threshold = get_threshold(himmelblau, tau) # - 147.96
xx = get_mesh(dim, steps)
x_set = reshape_mesh(xx).numpy()
fx = himmelblau(torch.tensor(x_set))
x_to_elevation = {tuple(x): f for x, f in zip(x_set, fx)}
idx = torch.argmax(fx)
x_init = torch.tensor(x_set[idx]).reshape(1, -1)

def obj_func(X):
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X)
    
    y = []
    for x in X:
        y.append(x_to_elevation[tuple(x.tolist())])
    return torch.tensor(y).reshape(-1, 1)


inputs, obj_vals = generate_initial_data(
    10,
    dim,
    obj_func,
    x_set=x_set,
    x_init=x_init,
)
model = fit_model(
    inputs,
    obj_vals,
    model_type="gp",
)

#%%


class LSE():
    def __init__(self, x_set, threshold):
        self.beta = 3
        self.epsilon = 0.01
        self.x_set = x_set
        self.x_to_C = {}
        for x in self.x_set:
            self.x_to_C[tuple(x)] = np.array([-np.inf, np.inf])
        self.threshold = threshold
        self.H = []
        self.L = []
        
    def get_next_x(self, model):
        # posterior = model.posterior(torch.from_numpy(self.x_set))
        # mean, var = posterior.mean.detach(), posterior.variance.detach()
        x_next = None
        max_acq_val = -np.inf

        idx_to_del = []
        for i, x in enumerate(self.x_set):
            x_post = model.posterior(torch.from_numpy(x.reshape(1, -1)))
            mean = x_post.mean.detach().numpy().squeeze()
            var = x_post.variance.detach().numpy().squeeze()
            C = self.x_to_C[tuple(x)]
            Q = np.array([
                mean - np.sqrt(self.beta * var),
                mean + np.sqrt(self.beta * var)
            ])
            # get the intersection of C and Q
            C = np.array([max(C[0], Q[0]), min(C[1], Q[1])])
            self.x_to_C[tuple(x)] = C
            if C[0] + self.epsilon > self.threshold:
                
                self.x_to_C.pop(tuple(x))
                self.H.append(x)
                if len(self.x_to_C) == 0:
                    return None
                idx_to_del.append(i)
                continue

            if C[1] - self.epsilon < self.threshold:
                self.x_to_C.pop(tuple(x))
                self.L.append(x)
                if len(self.x_to_C) == 0:
                    return None
                idx_to_del.append(i)   
                continue
            acq_val = np.min(
                [C[1] - self.threshold, self.threshold - C[0]]
            )
            if acq_val > max_acq_val:
                max_acq_val = acq_val
                x_next = x
        self.x_set = np.delete(self.x_set, idx_to_del, axis=0)

        return x_next
    
#%%
lse = LSE(x_set, threshold)
for i in range(50):
    
    x_next = torch.from_numpy(lse.get_next_x(model).reshape(1, -1))
    print(x_next)
    if x_next is None:
        break
    y_next = obj_func(x_next)
    inputs = torch.cat([inputs, x_next])
    obj_vals = torch.cat([obj_vals, y_next])
    model = fit_model(
        inputs,
        obj_vals,
        model_type="gp",
    )
# %%
# plot lse.H and lse.L
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.contourf(xx[0], xx[1], fx.reshape(steps, steps), levels=100)
H = np.array(lse.H)
L = np.array(lse.L)
ax.scatter(H[:, 0], H[:, 1], color="red")
ax.scatter(L[:, 0], L[:, 1], color="blue")
plt.show()
# %%
