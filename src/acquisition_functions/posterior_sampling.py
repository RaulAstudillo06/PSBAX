#!/usr/bin/env python3

import torch
from botorch.acquisition.analytic import PosteriorMean
from botorch.utils.gp_sampling import get_gp_samples
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.gp_regression import SingleTaskGP
from copy import deepcopy

from src.models.deep_kernel_gp import DKGP


def gen_posterior_sampling_batch(model, algorithm, batch_size, **kwargs):
    eval_all = kwargs.get("eval_all", False)
    if batch_size > 1:
        raise ValueError("Batch size > 1 currently not supported")
    else:
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
            obj_func_sample = PosteriorMean(model=obj_func_sample)

        x_output = algorithm.execute(obj_func_sample) # np.array(N, n_dim)
        x_output = torch.tensor(x_output)
        if len(x_output.shape) == 1:
            x_output = x_output.view(torch.Size([1, x_output.shape[0]]))
        post_obj_x_output = model(x_output)
        batch = x_output[torch.argmax(post_obj_x_output.variance)].unsqueeze(0)
    return batch

# algo.run_algorithm_on_f_botorch(obj_func)

def gen_posterior_sampling_batch_discrete(model, algorithm, batch_size, **kwargs):
    eval_all = kwargs.get("eval_all", False)
    if batch_size > 1:
        raise ValueError("Batch size > 1 currently not supported")
    else:
        # obj_func_sample = get_gp_samples(
        #     model=model,
        #     num_outputs=1,
        #     n_samples=1,
        #     num_rff_features=1000,
        # )
        # obj_func_sample = PosteriorMean(model=obj_func_sample)
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
            obj_func_sample = PosteriorMean(model=obj_func_sample)
        idx_output = algorithm.execute(obj_func_sample) # np.array(N, n_dim)
        if eval_all:
            return idx_output
        else: 
            x_output = algorithm.index_to_x(idx_output)
            if len(x_output.shape) == 1:
                x_output = x_output.view(torch.Size([1, x_output.shape[0]]))
            post_obj_x_output = model(x_output)
            selected_idx = torch.argmax(post_obj_x_output.variance)
        return [idx_output[selected_idx]]



