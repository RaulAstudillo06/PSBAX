import torch
import numpy as np

from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal

from src.utils import seed_torch

torch.set_default_dtype(torch.float64)

class DiscreteObjective:
    def __init__(self, name, df, idx_type="str"):
        self.name = name
        self.df = df
        self.idx_type = idx_type
    
    def __call__(self, idx):
        '''
        Args:
            idx: str or list of str
        Returns:
            torch.tensor: (N,)
        '''

        return torch.tensor(self.df.loc[idx, "y"].values)
    
    def get_x(self, idx=None):
        if idx is None:
            return torch.tensor(self.df.drop(columns=["y"]).values)
        return self.index_to_x(idx)
    
    def get_y(self, idx=None):
        if idx is None:
            return torch.tensor(self.df["y"].values)
        return torch.tensor(self.df.loc[idx, "y"].values)
    
    def get_idx(self):
        return self.df.index.tolist()
    
    def index_to_x(self, idx):
        result = self.df.drop(columns=["y"]).loc[idx].values
        if len(result.shape) == 1:
            result = result.reshape(1, -1)
        return torch.tensor(result)
    
    def index_to_int_index(self, idx):
        return [self.df.index.get_loc(i) for i in idx]
    
    def update_df(self, new_df):
        self.df = new_df

class DiscoBAXObjective(DiscreteObjective):
    def __init__(
            self, 
            name, 
            df, 
            noise_bugget=20,
            noise_type="additive",
            idx_type="str",
            **kwargs
        ):
        super().__init__(name, df, idx_type)
        self.noise_budget = noise_bugget
        self.noise_type = noise_type
        self.eta_func_lst = None
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def set_noise(self, lengthscale=1.0, outputscale=1.0):
        x = self.get_x()
        n = x.shape[0]
        if self.noise_type == "additive":
            with torch.no_grad():
                kernel = ScaleKernel(
                    RBFKernel(lengthscale=lengthscale), outputscale=outputscale
                )
                cov = kernel(x)
            mean = torch.zeros(n)
            eta_func_lst = []
            for _ in range(self.noise_budget):
                eta = MultivariateNormal(mean, cov).sample().detach().numpy()
                if self.nonneg:
                    eta_func_lst.append(
                        lambda fx: np.maximum(0, fx + eta)
                    )
                else:
                    eta_func_lst.append(
                        lambda fx: fx + eta
                    )

        elif self.noise_type == "multiplicative":
            with torch.no_grad():
                kernel = ScaleKernel(
                    RBFKernel(lengthscale=lengthscale), outputscale=outputscale
                )
                cov = kernel(x)
            mean = torch.zeros(n)
            eta_func_lst = []
            for _ in range(self.noise_budget):
                l = MultivariateNormal(mean, cov).sample().detach().numpy()
                p = 1 / (1 + np.exp(-l))
                eta = np.random.binomial(1, p)
                if self.nonneg:
                    eta_func_lst.append(
                        lambda fx: np.maximum(0, fx + eta)
                    )
                else:
                    eta_func_lst.append(
                        lambda fx: fx + eta
                    )
        self.eta_func_lst = eta_func_lst

    def get_noisy_f_lst(self, fx):
        return np.asarray([eta(fx) for eta in self.eta_func_lst])
    
    def initialize(self, **kwargs):
        seed = kwargs.get("seed", None)
        if seed is not None:
            seed_torch(seed)
        self.set_noise()