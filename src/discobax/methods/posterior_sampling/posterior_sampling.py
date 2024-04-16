"""
Copyright 2022 Pascal Notin, University of Oxford
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
from typing import AnyStr, List
import torch
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal

from slingpy import AbstractDataSource, AbstractBaseModel
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import (
    BaseBatchAcquisitionFunction,
)


"""BAX acquisition function for top k estimation """


class PSAcquisition(BaseBatchAcquisitionFunction):
    def __init__(
        self,
        subset_size: int = 20,  # subset size in subset select (DiscoBAX only)
        noise_type: str = "additive",  # noise type in subset select [additive|multiplicative] (DiscoBAX only)
        noise_lengthscale: float = 1.0,  # lengthscale param for noise computation (DiscoBAX only)
        noise_outputscale: float = 1.0,  # output scale param for noise computation (DiscoBAX only)
        # num_samples_EIG: int = 20,  # number of samples to compute EIG (All BAX)
        # num_samples_entropy: int = 20,  # number of samples for entropy computation (All BAX)
        # entropy_average_mode: str = "harmonic",  # type of average across samples in entropy computation [harmonic|arithmetic] (All BAX)
        noise_budget: int = 20,  # budget for eta sampling in subset select (DiscoBAX only)
    ):

        self.noise_type = noise_type
        self.noise_lengthscale = noise_lengthscale
        self.noise_outputscale = noise_outputscale
        self.subset_size = subset_size
        self.noise_budget = noise_budget
        self.nonneg = False
        self.mvn = None

    def __call__(
        self,
        dataset_x: AbstractDataSource,
        acquisition_batch_size: int,
        available_indices: List[AnyStr],
        last_selected_indices: List[AnyStr],
        cumulative_indices: List[AnyStr] = None,
        model: AbstractBaseModel = None,
        dataset_y: AbstractDataSource = None,
        temp_folder_name: str = "tmp/model/model.pt",
    ) -> List:
        # dataset_x_avail = dataset_x.subset(available_indices) # TODO: what is the use of dataset_x_avail?
        # hxs = []
        # model.save(temp_folder_name) # FIXME
        model.save_folder(temp_folder_name)
        outputs = []
        # We obtain several MC samples to estimate the second term in Equation 1

        
            # Sample (f_ip)_j values

        best_indices = []
        for _ in range(acquisition_batch_size):
            model.load_folder(temp_folder_name)
            f = (
                model.get_model_prediction(dataset_x, return_multiple_preds=False)[0]
                .flatten()
                .detach()
                .numpy()
            )  # Use consistent MC dropout to ensure the same mask is used for all input x points
            # f is of shape (len(available_indices), )
            # f, _ = model.predict(dataset_x.get_data()[0])
            # x = dataset_x.get_data()[0]

            etas_lst = []
            if self.noise_type == "additive":
                etas = self.mvn.rsample(torch.Size([self.noise_budget])).detach().numpy()
                for eta in etas:
                    etas_lst.append(eta)
            elif self.noise_type == "multiplicative":
                ls = self.mvn.rsample(torch.Size([self.noise_budget])).detach().numpy()
                for l in ls:
                    p = 1 / (1 + np.exp(-l))
                    eta = np.random.binomial(1, p)
                    etas_lst.append(eta)
            self.etas_lst = etas_lst
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

            out = new_subset_select(
                f,
                fout_lst,
                self.subset_size,
            )
            outputs.append(out)
            dataset_x_indices = dataset_x.get_row_names()
            output_indices = [dataset_x_indices[i] for i in out]
            dataset_x_outputs = dataset_x.subset(output_indices)
            
            y_stds = model.predict(
                dataset_x_outputs,
                return_std_and_margin=True,
            )[1]
            best_index = np.argmax(y_stds)
            best_indices.append(output_indices[best_index])
        
        return best_indices 


def new_subset_select(v, fouts, subset_size):
    '''
    v : a single sample of f from the model (GP / MLP)
    '''
    # for moment, just do monte carlo estimate
    # h_sampler : v -> h(v, eta), with eta sampled from some distribution
    # out_fn = either gaussian additive noise or multiplicative bernoulli sampled from GP classifier
    values = np.asarray([fout(v) for fout in fouts]) # sampled budget # of etas (budget, len(v))
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


def noise_subset_select(noise_type, x, lengthscale=1.0, outputscale=1.0):
    if noise_type == "additive":
        return lambda f: gaussian_noise_sampler(x, f, lengthscale, outputscale)
    elif noise_type == "multiplicative":
        return lambda f: bernoulli_noise_sampler(x, f, lengthscale, outputscale)


def top_k_idx(v, k):
    idxes = np.argsort(v)[-k:]
    return idxes


def level_set(v, c):
    idxes = np.where(v > c)
    return idxes


def subset_select(v, h_sampler, subset_size, budget=20):
    '''
    v : a single sample of f from the model (GP / MLP)
    '''
    # for moment, just do monte carlo estimate
    # h_sampler : v -> h(v, eta), with eta sampled from some distribution
    # out_fn = either gaussian additive noise or multiplicative bernoulli sampled from GP classifier
    values = np.asarray([h_sampler(v) for _ in range(budget)]) # sampled budget # of etas (budget, len(v))
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
    # with torch.no_grad():
    #     kernel = ScaleKernel(
    #         RBFKernel(lengthscale=lengthscale), outputscale=outputscale
    #     ).cuda()
    #     cov = kernel(torch.tensor(x).float().cuda())
    # mean = torch.zeros(fx.shape).float().cuda()
    # eta = MultivariateNormal(mean, cov).sample().detach().cpu().numpy()
    # return np.maximum(0, fx + eta)
    with torch.no_grad():
        kernel = ScaleKernel(
            RBFKernel(lengthscale=lengthscale), outputscale=outputscale
        )
        cov = kernel(torch.tensor(x).float())
    mean = torch.zeros(fx.shape).float()
    eta = MultivariateNormal(mean, cov).sample().detach().numpy()
    return np.maximum(0, fx + eta)


def bernoulli_noise_sampler(x, fx, lengthscale=1.0, outputscale=1.0, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    # with torch.no_grad():
    #     kernel = ScaleKernel(
    #         RBFKernel(lengthscale=lengthscale), outputscale=outputscale
    #     ).cuda()
    #     cov = kernel(torch.tensor(x).float().cuda())
    # mean = torch.zeros(fx.shape).float().cuda()
    # l = MultivariateNormal(mean, cov).sample().detach().cpu().numpy()
    # p = 1 / (1 + np.exp(-l))
    # eta = np.random.binomial(1, p)
    # return np.maximum(0, fx * eta)
    with torch.no_grad():
        kernel = ScaleKernel(
            RBFKernel(lengthscale=lengthscale), outputscale=outputscale
        )
        cov = kernel(torch.tensor(x).float())
    mean = torch.zeros(fx.shape).float()
    l = MultivariateNormal(mean, cov).sample().detach().numpy()
    p = 1 / (1 + np.exp(-l))
    eta = np.random.binomial(1, p)
    return np.maximum(0, fx * eta)


def random_argmax(vals):
    max_val = np.max(vals)
    idxes = np.where(vals == max_val)[0]
    return np.random.choice(idxes)
