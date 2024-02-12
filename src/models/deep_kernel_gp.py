#!/usr/bin/env python3

from __future__ import annotations

import gpytorch
import torch

from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from torch import Tensor


class FFNN(torch.nn.Sequential):
    """
    Architecture is
    [
        input,
        architecture[1] + activation + dropout,
        ...
        architecture[-2] + activation + dropout,
        architecture[-1]
    ]
    """
    act_dict = {
        "relu": torch.nn.ReLU,
        "leaky_relu": torch.nn.LeakyReLU,
        "swish": torch.nn.SiLU,
        "sigmoid": torch.nn.Sigmoid,
        "tanh": torch.nn.Tanh,
        "softmax": torch.nn.Softmax,
    }

    def __init__(
        self,
        architecture,
        activation="leaky_relu",
    ):
        super().__init__()

        self.architecture = architecture
        act_layer = self.act_dict[activation.lower()]
        self.activation = activation

        for dim in range(len(architecture)):
            name = str(dim + 1)
            if dim + 1 < len(architecture):
                self.add_module(
                    "linear" + name,
                    torch.nn.Linear(architecture[dim], architecture[dim + 1]).double(),
                )
            # don't dropout/activate from output layer
            if dim + 2 < len(architecture):
                self.add_module(activation + name, act_layer())
    
    def embedding(self, x: Tensor) -> Tensor:
        self.eval()
        # penultimate layer
        for dim in range(len(self.architecture)-1):
            x = self._modules["linear"+str(dim + 1)](x)
            if dim + 2 < len(self.architecture):
                x = self._modules[self.activation+str(dim + 1)](x)
        return x
    
    def get_params(self):
        return [{"params": self.parameters()}]
    
    def train_model(self, X, Y, lr, num_iter=100):
        self.train()
        optimizer = torch.optim.Adam(self.get_params(), lr=lr)
        mse = torch.nn.MSELoss()
        for iter in range(num_iter):
            optimizer.zero_grad()
            preds = self.forward(X)
            loss = mse(preds, Y)
            loss.backward()
            optimizer.step()

        self.eval()
        return self, None


class DKGP(SingleTaskGP):
    def __init__(
        self,
        train_X,
        train_Y,
        architecture,
        activation="leaky_relu",
    ):
        """
        """
        # TODO: ACTUALLY USE LIKELIHOOD! THIS MEANS THINGS HAVE PRIORS
        self.feature_extractor = None
        self.architecture = architecture
        if len(architecture) > 2:
            self.dkl = len(architecture) > 2


        covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel( # default is 2.5
                ard_num_dims=architecture[-2],
                num_dims=architecture[-2],
            )
        )

        likelihood = GaussianLikelihood(
                noise_constraint=GreaterThan(
                    1e-4,
                    transform=None,
                ),
            )

        SingleTaskGP.__init__(
            self,
            # TODO this doesn't work still bc training inputs diff
            train_X=train_X,
            train_Y=train_Y,
            covar_module=covar_module,
            likelihood=likelihood,
            # TODO: unclear what should be done here for NN outputs, etc.
            # input_transform=input_transform if not dkl else None,
            outcome_transform=Standardize(m=1),
        )

        if self.dkl:
            # chop GPR part of arc off, from E --> 1
            self.feature_extractor = FFNN(architecture[:-1], activation)

    def forward(self, x: Tensor) -> gpytorch.distributions.MultivariateNormal:
        # We're first putting our data through a deep net (feature extractor)
        emb = self.embedding(x)
        mean_x = self.mean_module(emb)
        covar_x = self.covar_module(emb)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def embedding(self, x: Tensor) -> Tensor:
        if self.dkl:
            return self.feature_extractor(x)
        else:
            return x
        
    def get_params(self):
        if self.dkl:
            return self.feature_extractor.get_params() + [
                {"params": self.covar_module.parameters()},
                {"params": self.mean_module.parameters()},
                {"params": self.likelihood.parameters()},
            ]
        else:
            return [{"params": self.parameters()}]

    def train_model(self, X, Y, lr, num_iter=100):
        self.train()
        self.likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        if not self.dkl:
            self.likelihood, self = self.likelihood, self
            fit_gpytorch_mll(mll)
        else:
            self.feature_extractor.train()
            optimizer = torch.optim.Adam(self.get_params(), lr=lr)
            with gpytorch.settings.fast_pred_var(), gpytorch.settings.use_toeplitz(False):
                for iter in range(num_iter):
                    optimizer.zero_grad()
                    # IMPORTANT: don't use fwd
                    preds = self(X)
                    loss = -mll(preds, Y)
                    # print(loss)
                    loss.backward()
                    optimizer.step()
            self.feature_extractor.eval()

        self.eval()
        self.likelihood.eval()
        return None
