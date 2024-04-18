#!/usr/bin/env python3

from typing import Callable, Dict, Optional

import os
import numpy as np
import torch

from torch import Tensor
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

from .models.deep_kernel_gp import DKGP

tkwargs = {
    "dtype": torch.float64,
    "device": torch.device("cpu"),
}

def generate_initial_data(
    num_init_points: int,
    input_dim: int,
    obj_func,
    # noise_type,
    # noise_level,
    # seed: int = None,
    **kwargs,
):  
    noise_type = kwargs.get("noise_type", "noiseless")
    noise_level = kwargs.get("noise_level", 0.0)
    seed = kwargs.get("seed", None)

    edge_positions = kwargs.get("edge_positions", None)
    if edge_positions is not None:
        idx = np.random.choice(range(edge_positions.shape[0]), num_init_points, replace=False)
        inputs = torch.tensor(edge_positions[idx])
    else:
        inputs = generate_random_points(num_init_points, input_dim, seed)
    outputs = get_obj_vals(obj_func, inputs, noise_type, noise_level)
    return inputs, outputs


def generate_random_points(num_points: int, input_dim: int, seed: int = None):
    # Generate `num_batches` inputs each constituted by `batch_size` points chosen uniformly at random
    if seed is not None:
        old_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        inputs = torch.rand([num_points, input_dim])
        torch.random.set_rng_state(old_state)
    else:
        inputs = torch.rand([num_points, input_dim])
    return inputs


def get_obj_vals(obj_func, inputs, noise_type, noise_level):
    obj_vals = obj_func(inputs)
    if noise_type == "noiseless":
        corrupted_obj_vals = obj_vals
    return corrupted_obj_vals


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


def get_function_samples(model):
    if isinstance(model, DKGP):
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
                    num_rff_features=512,
                    )
            )
        def aux_func(X):
            val = []
            for gp_sample in gp_samples:
                val.append(gp_sample.posterior(X).mean)
            return torch.cat(val, dim=-1)
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