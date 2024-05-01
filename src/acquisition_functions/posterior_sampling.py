#!/usr/bin/env python3

import torch
import numpy as np
from botorch.acquisition.analytic import PosteriorMean
from botorch.utils.gp_sampling import get_gp_samples
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.gp_regression import SingleTaskGP
from botorch.models import ModelListGP
from copy import deepcopy

from src.models.deep_kernel_gp import DKGP
from src.utils import get_function_samples, rand_argmax

tkwargs = {
    "dtype": torch.float64,
    "device": torch.device("cpu"),
}

def gen_posterior_sampling_batch(model, algorithm, batch_size, **kwargs):
    eval_all = kwargs.get("eval_all", False)

    batch = []
    for _ in range(batch_size):
        obj_func_sample = get_function_samples(model)
        x_output = algorithm.execute(obj_func_sample) # np.array(N, n_dim)
        x_output = torch.tensor(x_output)
        if len(x_output.shape) == 1:
            x_output = x_output.view(torch.Size([1, x_output.shape[0]]))
        post_obj_x_output = model.posterior(x_output)
        post_std = post_obj_x_output.stddev.detach().squeeze() # (N, ) or (N, num_objectives)
        if len(post_std.shape) > 1:
            # 1/2 log(det(Sigma)) = sum(log(std))
            entropy = torch.sum(torch.log(post_std), dim=-1) # (N, )
        else:
            # log and sqrt are monotonic
            entropy = post_std
        max_idx = rand_argmax(entropy)
        x_cand = x_output[max_idx].unsqueeze(0)
        batch.append(x_cand)

    return torch.cat(batch, dim=0)

# algo.run_algorithm_on_f_botorch(obj_func)

def gen_posterior_sampling_batch_discrete(model, algorithm, batch_size, **kwargs):
    eval_all = kwargs.get("eval_all", False)
    batch = []
    for _ in range(batch_size):
        # obj_func_sample = get_gp_samples(
        #     model=model,
        #     num_outputs=1,
        #     n_samples=1,
        #     num_rff_features=1000,
        # )
        # obj_func_sample = PosteriorMean(model=obj_func_sample)

        # Original
        obj_func_sample = get_function_samples(model)
        idx_output = algorithm.execute(obj_func_sample) # np.array(N, n_dim)
        if eval_all:
            return idx_output
        else: 
            x_output = algorithm.index_to_x(idx_output)
            if len(x_output.shape) == 1:
                x_output = x_output.view(torch.Size([1, x_output.shape[0]]))
            post_obj_x_output = model.posterior(x_output)
            selected_idx = rand_argmax(post_obj_x_output.variance.detach().squeeze())
            # FIXME: the variance is really low.

        batch.append(idx_output[selected_idx])
    
    return batch
    #     # New
    # x_cand = []
    # for index in batch:
    #     x_cand.append(algorithm.index_to_x(index))

    # return torch.cat(x_cand, dim=0)



