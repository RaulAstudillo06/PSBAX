#!/usr/bin/env python3

import torch
import numpy as np
from botorch.acquisition.analytic import PosteriorMean
from botorch.utils.gp_sampling import get_gp_samples
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.gp_regression import SingleTaskGP
from botorch.models import ModelListGP
from botorch.optim import optimize_acqf_discrete
from copy import deepcopy

from src.models.deep_kernel_gp import DKGP
from src.utils import get_function_samples, rand_argmax
from .entropy import EntropyAcquisitionFunction

tkwargs = {
    "dtype": torch.float64,
    "device": torch.device("cpu"),
}

# def gen_posterior_sampling_batch(model, algorithm, batch_size, **kwargs):
#     eval_all = kwargs.get("eval_all", False)

#     batch = []
#     for _ in range(batch_size):
#         obj_func_sample = get_function_samples(model)
#         x_output = algorithm.execute(obj_func_sample) # np.array(N, n_dim)
#         x_output = torch.tensor(x_output)
#         if len(x_output.shape) == 1:
#             x_output = x_output.view(torch.Size([1, x_output.shape[0]]))
#         post_obj_x_output = model.posterior(x_output)
#         post_std = post_obj_x_output.stddev.detach().squeeze() # (N, ) or (N, num_objectives)
#         if len(post_std.shape) > 1:
#             # 1/2 log(det(Sigma)) = sum(log(std))
#             entropy = torch.sum(torch.log(post_std), dim=-1) # (N, )
#         else:
#             # log and sqrt are monotonic
#             entropy = post_std
#         max_idx = rand_argmax(entropy)
#         x_cand = x_output[max_idx].unsqueeze(0)
#         batch.append(x_cand)

#     return torch.cat(batch, dim=0)

def gen_posterior_sampling_batch(model, algorithm, batch_size, **kwargs):
    eval_all = kwargs.get("eval_all", False)
    
    if algorithm.params.name == "SubsetSelect":
        batch = []
        while len(batch) < batch_size:
            obj_func_sample = get_function_samples(model)
            idx_output = algorithm.execute(obj_func_sample)
            batch.extend(idx_output)
        x_batch = algorithm.index_to_x(batch)
        acq_func = EntropyAcquisitionFunction(model=model)
        x_next, _ = optimize_acqf_discrete(
                acq_function=acq_func, 
                q=batch_size, 
                choices=x_batch, 
                max_batch_size=100, 
            )
        idx_next = algorithm.obj_func.get_idx_from_x(x_next)
        return idx_next
    else:
        batch = []
        while len(batch) < batch_size:
            obj_func_sample = get_function_samples(model)
            x_output = algorithm.execute(obj_func_sample)
            x_output = torch.tensor(x_output)
            if len(x_output.shape) == 1:
                x_output = x_output.view(torch.Size([1, x_output.shape[0]]))
            for x in x_output:
                batch.append(x)
        
        x_batch = torch.stack(batch) # (batch_size + k, n_dim)

        acq_func = EntropyAcquisitionFunction(model=model)
        x_next, _ = optimize_acqf_discrete(acq_function=acq_func, q=batch_size, choices=x_batch, max_batch_size=100)
        
        return x_next

    
