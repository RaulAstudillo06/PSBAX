#!/usr/bin/env python3

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from src.models.deep_kernel_gp import DKGP


def fit_model(inputs: Tensor, outputs: Tensor, model_type: str):
    if len(outputs.shape) == 1:
        outputs = outputs.view(torch.Size([outputs.shape[0], 1]))
    if model_type == "gp":
        model = SingleTaskGP(
            train_X=inputs,
            train_Y=outputs,
            outcome_transform=Standardize(m=outputs.shape[-1]),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
    elif model_type == "dkgp":
        std_outputs = (outputs - outputs.mean())/outputs.std()
        model = DKGP(train_X=inputs, train_Y=std_outputs, architecture=[inputs.shape[-1], 32, 32, inputs.shape[-1], 1])
        model.train_model(X=inputs, Y=std_outputs.squeeze(-1), lr=1e-2, num_iter=10000)
    return model
