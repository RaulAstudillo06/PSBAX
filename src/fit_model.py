#!/usr/bin/env python3

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

import gpytorch

from gpytorch.constraints import Interval

from src.models.deep_kernel_gp import DKGP




def fit_model(inputs: Tensor, outputs: Tensor, model_type: str, **kwargs):
    if len(outputs.shape) == 1:
        outputs = outputs.view(torch.Size([outputs.shape[0], 1]))
    
    try:
        if model_type == "gp":
            # covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_constraint=Interval(1e-5, 1e0)))
            kernel_type = kwargs.pop("kernel_type", "matern")

            if kernel_type == "rbf":
                covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            else:
                covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())

            models = []
            for f_i in range(outputs.shape[-1]):
                train_y = outputs[:, f_i].view(-1, 1)
                model = SingleTaskGP(
                    train_X=inputs,
                    train_Y=outputs[:, f_i].view(-1, 1),
                    covar_module=covar_module,
                    outcome_transform=Standardize(m=train_y.shape[-1]),
                )
                models.append(model)
            
            if len(models) == 1:
                model = models[0]

                # NOTE: for using a fixed model without training, only for BAX testing
                state_dict = kwargs.pop("state_dict", None)
                if state_dict is not None:
                    model.load_state_dict(state_dict)
                    return model

                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll)
            
            else:
                for model in models:
                    mll = ExactMarginalLogLikelihood(model.likelihood, model)
                    fit_gpytorch_mll(mll)
                model = ModelListGP(*models)
            
            if (verbose := kwargs.pop("model_verbose", False)):
                print(f"covar_module.base_kernel.ls: {model.covar_module.base_kernel.raw_lengthscale}")
                print(f"covar_module.outputscale: {model.covar_module.raw_outputscale}")
                print(f"likelihood.noise: {model.likelihood.noise}")
            
            return model
    
        elif model_type == "dkgp":
            architecture = kwargs.pop("architecture")
            if architecture is None:
                architecture =  [32, 32, inputs.shape[-1]]
            architecture = [inputs.shape[-1]] + architecture + [outputs.shape[-1]]
            std_outputs = (outputs - outputs.mean())/outputs.std()
            model = DKGP(train_X=inputs, train_Y=std_outputs, architecture=architecture)
            model.train_model(X=inputs, Y=std_outputs.squeeze(-1), lr=1e-2, num_iter=10000)
        return model

    except:
        files_path = kwargs.pop("files_path", "results")
        print("Error in fitting model, saving inputs and outputs to files")
        torch.save(inputs, f"{files_path}/inputs.pt")
        torch.save(outputs, f"{files_path}/outputs.pt")
        return None