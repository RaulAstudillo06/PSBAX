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


from . import abstract_numpy_class


"""BAX acquisition function for top k estimation """


class BaxAcquisition(BaseBatchAcquisitionFunction):
    def __init__(
        self,
        subset_size: int = 20,  # subset size in subset select (DiscoBAX only)
        noise_type: str = "additive",  # noise type in subset select [additive|multiplicative] (DiscoBAX only)
        noise_lengthscale: float = 1.0,  # lengthscale param for noise computation (DiscoBAX only)
        noise_outputscale: float = 1.0,  # output scale param for noise computation (DiscoBAX only)
        num_samples_EIG: int = 20,  # number of samples to compute EIG (All BAX)
        num_samples_entropy: int = 20,  # number of samples for entropy computation (All BAX)
        entropy_average_mode: str = "harmonic",  # type of average across samples in entropy computation [harmonic|arithmetic] (All BAX)
        batch_selection_mode: str = "topk_EIG",  # method to select all points in acquisition batch [topk_EIG, best_subset]. best_subset only valid if subset_size = batch size (All BAX)
        noise_budget: int = 20,
    ):
        self.noise_type = noise_type
        self.noise_lengthscale = noise_lengthscale
        self.noise_outputscale = noise_outputscale
        self.subset_size = subset_size
        self.num_samples_EIG = num_samples_EIG
        self.num_samples_entropy = num_samples_entropy
        self.batch_selection_mode = batch_selection_mode
        self.entropy_average_mode = entropy_average_mode
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
        dataset_x_avail = dataset_x.subset(available_indices) # TODO: what is the use of dataset_x_avail?
        hxs = []
        # model.save(temp_folder_name) # FIXME
        model.save_folder(temp_folder_name)
        outputs = []
        # We obtain several MC samples to estimate the second term in Equation 1
        for j in range(self.num_samples_EIG):
            # model.load(temp_folder_name)
            model.load_folder(temp_folder_name)
            # Sample (f_ip)_j values
            f = (
                model.get_model_prediction(dataset_x, return_multiple_preds=False)[0]
                .flatten()
                .detach()
                .numpy()
            )  # Use consistent MC dropout to ensure the same mask is used for all input x points
            # f is of shape (len(available_indices), )
            # f, _ = model.predict(dataset_x.get_data()[0])
            x = dataset_x.get_data()[0]
            # Compute S_j based on the sampled (f_ip)_j

            if self.noise_type == "additive":
                etas = self.mvn.rsample(torch.Size([self.noise_budget])).detach().numpy() # (budget, 17176)
                values = etas + f  # (budget, 17176)

            elif self.noise_type == "multiplicative":
                ls = self.mvn.rsample(torch.Size([self.noise_budget])).detach().numpy()
                def stable_sigmoid(x):
                    return np.piecewise(
                            x,
                            [x > 0],
                            [lambda i: 1 / (1 + np.exp(-i)), lambda i: np.exp(i) / (1 + np.exp(i))],
                        )
                p = stable_sigmoid(ls) # (budget, 17176)
                etas = np.random.binomial(1, p) # (budget, 17176)
                values = np.multiply(etas, f)

            self.etas = etas
            if self.nonneg:
                values = np.maximum(0, values)

            out = new_subset_select(
                f,
                values,
                self.subset_size,
            )

            # out = subset_select(
            #     f,
            #     noise_subset_select(
            #         self.noise_type,
            #         x,
            #         self.noise_lengthscale,
            #         self.noise_outputscale,
            #     ),
            #     self.subset_size,
            # )
            outputs.append(out)
            new_x = abstract_numpy_class.NumpyDataSource(
                np.concatenate(
                    [dataset_x.subset(cumulative_indices).get_data()[0], x[out]]
                )
            )
            new_y = abstract_numpy_class.NumpyDataSource(
                np.concatenate(
                    [
                        dataset_y.subset(cumulative_indices).get_data()[0],
                        f[out].reshape(-1, 1),
                    ]
                ).flatten()
            )
            # Fit a new model on the concatenation of previously selected points and selected subset S_j
            model.fit(new_x, new_y)
            # Get entropy values for all points that can be selected in the next cycle, for the MC sample indexed by j
            new_post_entropy = [
                list(
                    entropy(
                        dataset_x_avail,
                        model,
                        self.num_samples_entropy,
                        avg_mode=self.entropy_average_mode,
                    )
                )
            ]
            hxs.append(new_post_entropy)

        # Compute information gain
        hxs = np.array(hxs)
        hxs = hxs.reshape(self.num_samples_EIG, len(available_indices))
        eigs = []
        model.load_folder(temp_folder_name)
        hx = entropy(
            dataset_x_avail,
            model,
            self.num_samples_entropy,
            avg_mode=self.entropy_average_mode,
        )  # Outside of for loop to make the most of GPU parallelism
        for i in range(len(available_indices)):
            eigs.append(hx[i] - np.mean(hxs[:, i]))
        if acquisition_batch_size == 1:
            best_indices = [random_argmax(eigs)]
        elif self.batch_selection_mode == "topk_EIG":
            best_indices = top_k_idx(eigs, acquisition_batch_size)
        elif self.batch_selection_mode == "best_subset":
            assert (
                acquisition_batch_size == self.subset_size
            ), "best_subset selection only implemented for when batch size = subset_size"
            best_indices = None
            best_item = random_argmax(eigs)
            for i in range(self.num_samples_EIG):
                if best_item in outputs[i]:
                    best_indices = outputs[i]
                    break
            if best_indices is None:
                best_indices = top_k_idx(eigs, acquisition_batch_size)
        proposal = np.array(available_indices)[best_indices]
        return proposal.tolist()

def new_subset_select(v, values, subset_size):
    '''
    v : a single sample of f from the model (GP / MLP)
    '''
    # for moment, just do monte carlo estimate
    # h_sampler : v -> h(v, eta), with eta sampled from some distribution
    # out_fn = either gaussian additive noise or multiplicative bernoulli sampled from GP classifier
    # values = np.asarray([fout(v) for fout in fouts]) # sampled budget # of etas (budget, len(v))
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


def entropy(
    dataset_x, model, num_samples_entropy=20, precision=1.0, avg_mode="arithmetic"
):
    with torch.no_grad():
        f = (
            model.get_model_prediction(
                dataset_x,
                return_multiple_preds=True,
                num_target_samples=num_samples_entropy,
            )[0]
            .detach()
            .view(-1, num_samples_entropy)
        ) # (dataset_x.get_shape()[0][0], num_samples_entropy)
    # As per equation (22) in Gal & Ghahramani, "Dropout as a Bayesian Approximation:Representing Model Uncertainty in Deep Learning"
    E = 0
    for sample_index in range(num_samples_entropy):
        yi = f[:, sample_index].view(-1, 1) # (17156, 1)
        if avg_mode == "arithmetic":
            E += torch.logsumexp(-precision / 2 * (f - yi) ** 2, dim=1)
        elif avg_mode == "harmonic":
            E -= torch.logsumexp(precision / 2 * (f - yi) ** 2, dim=1)
    E = (
        E / num_samples_entropy
        - np.log(num_samples_entropy)
        - 0.5 * np.log(2 * np.pi)
        - 0.5 * np.log(1 / precision)
    ) # (17156, )
    return -E.numpy()


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
