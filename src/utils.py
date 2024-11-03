#!/usr/bin/env python3

from typing import Callable, Dict, Optional

import os
import numpy as np
import torch
import gpytorch

# from torch import Tensor
from torch import Tensor, distributions as tdist, nn
from copy import deepcopy
from botorch.models.gp_regression import SingleTaskGP
from botorch.models import ModelListGP
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import PosteriorMean
from botorch.generation.gen import get_best_candidates
from botorch.models.model import Model
from botorch.utils.gp_sampling import get_gp_samples
from botorch.models.deterministic import GenericDeterministicModel
from botorch.optim.optimize import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples

from .models.deep_kernel_gp import DKGP

tkwargs = {
    "dtype": torch.float64,
    "device": torch.device("cpu"),
}

def generate_initial_data(
    num_init_points: int,
    input_dim: int,
    obj_func,
    **kwargs,
):  
    noise_type = kwargs.get("noise_type", "noiseless")
    noise_level = kwargs.get("noise_level", None) # a tensor of noise levels as shape (num_obj,)
    seed = kwargs.pop("seed", None)

    inputs = generate_random_points(num_points=num_init_points, input_dim=input_dim, seed=seed, **kwargs)
    
    x_init = kwargs.get("x_init", None)
    if x_init is not None:
        inputs = torch.cat([inputs, x_init], dim=0)
    outputs = get_obj_vals(obj_func, inputs, noise_type, noise_level)
    return inputs, outputs


def generate_random_points(num_points: int, input_dim: int, seed: int = None, **kwargs):
    '''
    Returns:
        torch tensor of shape (num_points, input_dim)
    '''
    edge_coords = kwargs.get("edge_coords", None) # Dijkstra
    x_set = kwargs.get("x_set", None) # Level set
    x_batch = kwargs.get("x_batch", None) # Top k
    if edge_coords is not None:
        idx = np.random.choice(range(edge_coords.shape[0]), num_points, replace=True)
        inputs = torch.tensor(np.atleast_2d(edge_coords[idx]))
        return inputs
    elif x_set is not None:
        idx = np.random.choice(range(x_set.shape[0]), num_points, replace=True)
        inputs = torch.tensor(np.atleast_2d(x_set[idx]))
        return inputs
    elif x_batch is not None:
        idx = np.random.choice(range(x_batch.shape[0]), num_points, replace=True)
        if isinstance(x_batch, torch.Tensor):
            inputs = x_batch[idx]
        else:
            inputs = torch.tensor(np.atleast_2d(x_batch[idx]))
        return inputs

    if seed is not None:
        old_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        inputs = torch.rand([num_points, input_dim])
        torch.random.set_rng_state(old_state)
    else:
        inputs = torch.rand([num_points, input_dim])
    return inputs


def get_obj_vals(obj_func, inputs, noise_type, noise_level):
    '''
    Args:
        obj_func: callable
        inputs: torch tensor, shape (num_points, input_dim)
        noise_type: str, "noiseless" or "noisy"
        noise_level: tensor, shape (1, num_obj)  
    '''

    obj_vals = obj_func(inputs)
    input_dim = inputs.shape[-1]
    if noise_type == "noiseless" or noise_level is None:
        corrupted_obj_vals = obj_vals
    else:
        noise = torch.multiply(noise_level, torch.randn_like(obj_vals))
        corrupted_obj_vals = obj_vals + noise
    return corrupted_obj_vals


def compute_noise_std(obj_func, percent_noise, bounds, num_points=1000, seed=None):
    '''
    Args:
        obj_func: callable
        percent_noise: float, percentage of max and min obj_func value
        bounds: torch tensor, shape (2, input_dim)
        num_points: int, number of function evaluations to estimate the noise level
    '''
    inputs = draw_sobol_samples(bounds=bounds, n=num_points, q=1, seed=seed).squeeze(0)
    obj_vals = obj_func(inputs)
    max_vals = torch.max(obj_vals, dim=0).values # shape (num_obj,)
    min_val = torch.min(obj_vals, dim=0).values # shape (num_obj,)
    noise_stds = (max_vals - min_val) * percent_noise
    return noise_stds
    


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


def get_linear_weight_dist(model, architecture):
    """If GP has a linear kernel, then returns the multivariate Gaussian
    distribution for the equivalent weight vector w, such that
        sampling f ~ GP(𝜇,k)
    is equivalent to
        sampling w ~ N(mn, Sn), then computing f(x) = x @ w + 𝜇(x).

    Usage:
        wdist = gp.get_linear_weight_dist()
        w = wdist.sample()
        fX = X @ w + gp.mean_module(X)  # posterior sample from GP

    Raises:
        Exception: if kernel is nonlinear
    """

    with torch.no_grad():
        X = model.embedding(model.train_inputs[0])  # shape [n, h]
        y = model.train_targets  # shape [n]

        # LinearKernel has S0 = v * I, so S0^{-1} = I / v
        S0_inv = torch.eye(architecture[-1]) / model.covar_module.base_kernel.variance
        sigma2 = model.likelihood.noise.item()
        Sn_inv = S0_inv + X.T @ X / sigma2

        # mn = Sn @ X.T @ (y - self.mean_module(X)) / sigma2
        mn = torch.linalg.solve(Sn_inv, X.T @ (y - model.mean_module(X))) / sigma2

        dist = tdist.MultivariateNormal(loc=mn, precision_matrix=Sn_inv)
        return dist


def get_function_samples(model, **kwargs):
    if isinstance(model, DKGP):
        if isinstance(model.covar_module.base_kernel, gpytorch.kernels.LinearKernel):
            # this call also checks whether the kernel is linear
            architecture = kwargs.get("architecture", None)
            w = get_linear_weight_dist(model, architecture).sample()

            def f(X: Tensor) -> Tensor:
                """
                Args:
                    X: shape [n, d]
                    batch_size: batch size, set to -1 to treat X as a single batch

                Returns:
                    y: Tensor, shape [n, 1], the output dimension is required
                        by GenericDeterministicModel
                """
                emb = model.embedding(X)
                y = emb @ w + model.mean_module(emb)
                return y[:, None]

            obj_func_sample = GenericDeterministicModel(f)
        else: 
            aux_model = deepcopy(model)
            inputs = aux_model.train_inputs[0]
            aux_model.train_inputs = (aux_model.embedding(inputs),)
            gp_layer_sample = get_gp_samples(
                model=aux_model,
                num_outputs=1,
                n_samples=1,
                num_rff_features=1000,
            )

            def aux_obj_func_sample_callable(X):
                return gp_layer_sample.posterior(aux_model.embedding(X)).mean
            
            obj_func_sample = GenericDeterministicModel(f=aux_obj_func_sample_callable)
    elif isinstance(model, SingleTaskGP):
        obj_func_sample = get_gp_samples(
            model=model,
            num_outputs=1,
            n_samples=1,
            num_rff_features=1000,
        )
        obj_func_sample = PosteriorMean(model=obj_func_sample).to(**tkwargs)
        
    elif isinstance(model, ModelListGP):
        
        gp_samples = []
        for m in model.models:
            gp_samples.append(
                get_gp_samples(
                    model=m,
                    num_outputs=1,
                    n_samples=1,
                    num_rff_features=1000,
                    )
            )
        def aux_func(X):
            val = []
            for gp_sample in gp_samples:
                val.append(gp_sample.posterior(X).mean) # (N, 1, 1)
            return torch.cat(val, dim=-1) # (N, 1, 2)
        obj_func_sample = GenericDeterministicModel(f=aux_func).to(**tkwargs)
    
    return obj_func_sample

def seed_torch(seed, verbose=True):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def rand_argmax(tens):
    max_inds, = torch.where(tens == tens.max())
    return np.random.choice(max_inds)

def reshape_mesh(xx):
    '''
    Args:
        xx: list of torch tensors from get_mesh
    Returns:
        torch tensor of shape (num_points, input_dim = len(xx))
    '''
    return torch.hstack([xx[i].reshape(-1, 1) for i in range(len(xx))])

def get_mesh(dim, steps):
    '''
    Args:
        dim: int, input dimension
        steps: int, number of points in each dimension
    Returns:
        len(dim) list of torch tensors of shape (steps, steps, ..., steps)
    '''
    xx = []
    for _ in range(dim):
        ax = torch.linspace(0, 1, steps)
        xx.append(ax)
    xx = torch.meshgrid(*xx, indexing='ij')
    return xx

def f1_score(x_gt, x_pred):
    '''
    Args:
        x_gt: np.array(N, d)
        x_pred: np.array(N, d)
    '''
    x_gt_set = set()
    for x in x_gt:
        x_gt_set.add(tuple(x))
    x_pred_set = set()
    for x in x_pred:
        x_pred_set.add(tuple(x))
    tp = len(x_gt_set.intersection(x_pred_set))
    fp = len(x_pred_set.difference(x_gt_set))
    fn = len(x_gt_set.difference(x_pred_set))
    if tp == 0 or (tp + fp) == 0 or (tp + fn) == 0:
        return 0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

