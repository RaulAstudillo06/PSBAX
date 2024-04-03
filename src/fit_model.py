#!/usr/bin/env python3

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

import gpytorch

from gpytorch.constraints import Interval

from src.models.deep_kernel_gp import DKGP


def fit_model(inputs: Tensor, outputs: Tensor, model_type: str, **kwargs):
    if len(outputs.shape) == 1:
        outputs = outputs.view(torch.Size([outputs.shape[0], 1]))
    if model_type == "gp":
        # covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_constraint=Interval(1e-5, 1e0)))
        covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        model = SingleTaskGP(
            train_X=inputs,
            train_Y=outputs,
            covar_module=covar_module,
            outcome_transform=Standardize(m=outputs.shape[-1]),
        )

        # == for dijkstra == 
        # model.likelihood.noise_covar.register_constraint("raw_noise", Interval(5e-5, 5e-4))

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        if (verbose := kwargs.pop("model_verbose", False)):
            print(f"covar_module.base_kernel.ls: {model.covar_module.base_kernel.raw_lengthscale}")
            print(f"covar_module.outputscale: {model.covar_module.raw_outputscale}")
            print(f"likelihood.noise: {model.likelihood.noise}")


    elif model_type == "dkgp":
        architecture = kwargs.pop("architecture")
        if architecture is None:
            architecture =  [32, 32, inputs.shape[-1]]
        architecture = [inputs.shape[-1]] + architecture + [outputs.shape[-1]]
        std_outputs = (outputs - outputs.mean())/outputs.std()
        model = DKGP(train_X=inputs, train_Y=std_outputs, architecture=architecture)
        model.train_model(X=inputs, Y=std_outputs.squeeze(-1), lr=1e-2, num_iter=10000)
    return model
