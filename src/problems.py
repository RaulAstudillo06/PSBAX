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
            noise_budget=20,
            noise_type="additive",
            idx_type="str",
            **kwargs
        ):
        super().__init__(name, df, idx_type)
        self.noise_budget = noise_budget
        self.noise_type = noise_type
        self.etas_lst = None
        self.fout_lst = None
        self.verbose = False
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_etas(self, lengthscale=1.0, outputscale=1.0):
        x = self.get_x()
        n = x.shape[0]
        if self.etas_lst is None:
            
            if self.verbose:
                # current_seed = torch.get_rng_state()
                print(f"==== Generating {self.noise_budget} etas. ====")
            etas_lst = []

            with torch.no_grad():
                kernel = ScaleKernel(
                    RBFKernel(lengthscale=lengthscale), outputscale=outputscale
                )
                cov = kernel(x)
            mean = torch.zeros(n)
            mvn = MultivariateNormal(mean, cov)


            if self.noise_type == "additive":
                # with torch.no_grad():
                #     kernel = ScaleKernel(
                #         RBFKernel(lengthscale=lengthscale), outputscale=outputscale
                #     )
                #     cov = kernel(x)
                # mean = torch.zeros(n)
                # etas_lst = []
                # sample from the multivariate normal distribution self.noise_budget times
                # etas = MultivariateNormal(mean, cov).sample((self.noise_budget,)).detach().numpy()
                # for _ in range(self.noise_budget):
                #     eta = MultivariateNormal(mean, cov).sample().detach().numpy()
                #     etas_lst.append(eta)

                etas = mvn.rsample(torch.Size([self.noise_budget])).detach().numpy()
                for eta in etas:
                    etas_lst.append(eta)
                
            elif self.noise_type == "multiplicative":
                # with torch.no_grad():
                #     kernel = ScaleKernel(
                #         RBFKernel(lengthscale=lengthscale), outputscale=outputscale
                #     )
                #     cov = kernel(x)
                # mean = torch.zeros(n)

                ls = mvn.rsample(torch.Size([self.noise_budget])).detach().numpy()
                for l in ls:
                    p = 1 / (1 + np.exp(-l))
                    eta = np.random.binomial(1, p)
                    etas_lst.append(eta)

                # for _ in range(self.noise_budget):
                #     l = MultivariateNormal(mean, cov).sample().detach().numpy()
                #     p = 1 / (1 + np.exp(-l))
                #     eta = np.random.binomial(1, p)
                #     etas_lst.append(eta)
            self.etas_lst = etas_lst
        assert np.asarray(self.etas_lst).shape == (self.noise_budget, n), "etas_lst needs initialization."
        return self.etas_lst

    
    def set_noise(self, lengthscale=1.0, outputscale=1.0):
        self.etas_lst = self.get_etas(lengthscale, outputscale)
        fout_lst = []
        for eta in self.etas_lst:
            if self.nonneg:
                fout_lst.append(
                    lambda fx: np.maximum(0, fx + eta)
                )
            else:
                fout_lst.append(
                    lambda fx: fx + eta
                )
        
        self.fout_lst = fout_lst

    def get_noisy_f_lst(self, fx):
        assert self.fout_lst is not None, "Noisy functions hasn't been generated."
        return np.asarray([fout(fx) for fout in self.fout_lst])
    
    def initialize(self, **kwargs):
        seed = kwargs.get("seed", None)
        if seed is not None:
            seed_torch(seed)
        self.set_noise()


# class CaliforniaObjective():
