import numpy as np
import torch

from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal

from .algorithms import Algorithm
from ..util.misc_util import dict_to_namespace

class SubsetSelect(Algorithm):
    def set_params(self, params):
        """Set self.params, the parameters for the algorithm."""
        # params : x, df index
        super().set_params(params)
        params = dict_to_namespace(params)

        self.params.name = getattr(params, "name", "SubsetSelect")
        self.params.k = getattr(params, "k", 3)
        self.params.noise_type = getattr(params, "eta_type", "additive")
        self.params.budget = getattr(params, "budget", 10)
        self.params.df = getattr(params, "df", None)
        

    def initialize(self):
        super().initialize() # self.exe_path = Namespace(x=[], y=[])
        self.x_torch = torch.tensor(self.params.df.drop(columns=["y"]).values, dtype=torch.float64)
        self.y_torch = torch.tensor(self.params.df["y"].values, dtype=torch.float64)

    def run_algorithm_on_f(self, f):
        self.initialize()
        x_torch = self.x_torch.unsqueeze(1)
        fx = f(x_torch).detach().numpy()
        n = len(fx)

        if self.params.noise_type == "additive":
            sampler = lambda f_val: self.gaussian_noise_sampler(f_val)
            # values = np.asarray([gaussian_noise_sampler(fx) for _ in range(self.params.budget)])
        elif self.params.noise_type == "multiplicative":
            sampler = lambda f_val: self.bernoulli_noise_sampler(f_val)
            # values = np.asarray([bernoulli_noise_sampler(fx) for _ in range(self.params.budget)])

        
        values = np.asarray([sampler(fx) for _ in range(self.params.budget)])
        mean_values = np.mean(values, axis=0)
        mx = self.random_argmax(mean_values)
        idxes = [mx]
        
        for _ in range(self.params.k - 1):
            e_vals = np.zeros(n)
            for j in range(n):
                test_idxes = idxes
                if j not in idxes:
                    test_idxes = idxes + [j]
                    test_idxes = np.asarray(test_idxes)
                    e_vals[j] = np.mean(np.max(values[:, test_idxes], axis=-1))
            idxes.append(self.random_argmax(e_vals))

        indices = [self.params.df.index[i] for i in idxes]
        self.exe_path.x = self.x_torch[idxes].numpy() # (k, d)
        self.exe_path.y = mean_values[idxes] # (k,)
        return self.exe_path, indices
    
    def index_to_x(self, idx):
        return self.params.df.drop(columns=["y"]).loc[idx].values

    def index_to_y(self, idx):
        return self.params.df.at[idx, "y"]

    def get_next_x(self):
        raise NotImplementedError

    def execute(self, f):
        '''
        Returns:
            output: list of indices of selected points
        '''
        _, output = self.run_algorithm_on_f(f)
        return output
    
    def get_output(self):
        pass


    def gaussian_noise_sampler(self, fx, lengthscale=1.0, outputscale=1.0, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            # kernel = ScaleKernel(
            #     RBFKernel(lengthscale=lengthscale), outputscale=outputscale
            # ).cuda()
            
            # cov = kernel(torch.tensor(self.params.x).float().cuda())
            
            kernel = ScaleKernel(
                RBFKernel(lengthscale=lengthscale), outputscale=outputscale
            )
            cov = kernel(self.x_torch)
            mean = torch.zeros(fx.shape)
            eta = MultivariateNormal(mean, cov).sample().detach().numpy()

        return np.maximum(0, fx + eta)
    
    def bernoulli_noise_sampler(self, fx, lengthscale=1.0, outputscale=1.0, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        with torch.no_grad():
            kernel = ScaleKernel(
                RBFKernel(lengthscale=lengthscale), outputscale=outputscale
            ).cuda()
            cov = kernel(torch.tensor(self.params.x).float().cuda())
        mean = torch.zeros(fx.shape).float().cuda()
        l = MultivariateNormal(mean, cov).sample().detach().cpu().numpy()
        p = 1 / (1 + np.exp(-l))
        eta = np.random.binomial(1, p)
        return np.maximum(0, fx * eta)
    
    @staticmethod
    def random_argmax(vals):
        max_val = np.max(vals)
        idxes = np.where(vals == max_val)[0]
        return np.random.choice(idxes)


def select_builder(xs, threshold=0.75, subset_size=15):
    '''From toy_experiment
    '''
    if len(xs.shape)==2:
        xs = xs.reshape(-1)
    def subset_select(ys):
        selected = []
        available = list(range(len(ys)))

        for i in range(subset_size):
            if len(available) > 0:
                avail_arr = np.asarray(available).astype(int)
                max_idx = np.argmax(ys[avail_arr])
                idx = avail_arr[max_idx]
                selected.append(idx)
                for j in avail_arr:
                    if np.linalg.norm(xs[j] - xs[idx]) < threshold:
                        available.remove(j)
        return np.asarray(selected)
    return subset_select


def noise_subset_select(noise_type, x, lengthscale=1.0, outputscale=1.0):
    if noise_type == "additive":
        return lambda f: gaussian_noise_sampler(x, f, lengthscale, outputscale)
    elif noise_type == "multiplicative":
        return lambda f: bernoulli_noise_sampler(x, f, lengthscale, outputscale)



def subset_select(v, h_sampler, subset_size, budget=20):
    # for moment, just do monte carlo estimate
    # h_sampler : v -> h(v, eta), with eta sampled from some distribution
    # out_fn = either gaussian additive noise or multiplicative bernoulli sampled from GP classifier
    values = np.asarray([h_sampler(v) for _ in range(budget)])
    mx = random_argmax(np.mean(values, axis=0))
    idxes = [mx]
    n = len(v)
    for i in range(subset_size - 1):
        e_vals = np.zeros(n)
        for j in range(len(v)):
            test_idxes = idxes
            if j not in idxes:
                test_idxes = idxes + [j]
                test_idxes = np.asarray(test_idxes)
                e_vals[j] = np.mean(np.max(values[:, test_idxes], axis=-1))
        idxes.append(random_argmax(e_vals))
    return idxes


def gaussian_noise_sampler(x, fx, lengthscale=1.0, outputscale=1.0, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    with torch.no_grad():
        kernel = ScaleKernel(
            RBFKernel(lengthscale=lengthscale), outputscale=outputscale
        ).cuda()
        cov = kernel(torch.tensor(x).float().cuda())
    mean = torch.zeros(fx.shape).float().cuda()
    eta = MultivariateNormal(mean, cov).sample().detach().cpu().numpy()
    return np.maximum(0, fx + eta)


def bernoulli_noise_sampler(x, fx, lengthscale=1.0, outputscale=1.0, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    with torch.no_grad():
        kernel = ScaleKernel(
            RBFKernel(lengthscale=lengthscale), outputscale=outputscale
        ).cuda()
        cov = kernel(torch.tensor(x).float().cuda())
    mean = torch.zeros(fx.shape).float().cuda()
    l = MultivariateNormal(mean, cov).sample().detach().cpu().numpy()
    p = 1 / (1 + np.exp(-l))
    eta = np.random.binomial(1, p)
    return np.maximum(0, fx * eta)



