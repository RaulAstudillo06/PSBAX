import math
import torch
import numpy as np

from botorch.acquisition.multi_objective.monte_carlo import MultiObjectiveMCAcquisitionFunction
from botorch.utils.transforms import (
    concatenate_pending_points,
    t_batch_mode_transform,
)
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
)

class EntropyAcquisitionFunction(MultiObjectiveMCAcquisitionFunction):
    '''
    Args:
        - model
        - algo
    '''
    def __init__(
            self, 
            model, 
            **kwargs
    ):
        super().__init__(
            model=model, 
            # posterior_transform=ScalarizedPosteriorTransform(weights=torch.ones(n_obj, **tkwargs)),
            objective=IdentityMCMultiOutputObjective(),
        )
        self.model = model

    
    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X):

        posterior = self.model.posterior(X)
        cov = posterior.covariance_matrix # (batch_size, n_obj, n_obj)
        log_det = torch.logdet(cov) # (batch_size, )
        entropy = 0.5 * torch.log(torch.tensor(2 * math.pi)) + 0.5 * log_det
        return entropy