"""
Copyright 2022 Clare Lyle, University of Oxford
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
from slingpy import AbstractDataSource
from slingpy.models.abstract_base_model import AbstractBaseModel
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import (
    BaseBatchAcquisitionFunction,
)
from discobax.models.toy_experiment_models import rbf
from discobax.models import toy_experiment_models

# TODO: fix duplication in algos.

"""BAX acquisition function for top k estimation """


class TopKBaxAcquisition(BaseBatchAcquisitionFunction):
    def __call__(
        self,
        dataset_x: AbstractDataSource,
        select_size: int,
        available_indices: List[AnyStr],
        last_selected_indices: List[AnyStr] = None,
        model: AbstractBaseModel = None,
        budget=20,
        k=1,
    ) -> List:
        # sample top k outputs from model
        outputs = []
        hxs = []
        # save OG model state
        og_save_dir = model.save_path_dir
        model.save_path_dir = "/tmp/model"
        model.save()
        n = len(dataset_x)
        for i in range(budget):
            model.load()
            f = model.sample(dataset_x)
            out = top_k_idx(f, k)
            outputs.append(out)
            new_x = np.concatenate([model.X, dataset_x[out]])
            new_y = np.concatenate([model.mu, f[out]])
            model.fit(new_x, new_y)
            new_post_entropy = [entropy(x, model) for x in dataset_x[available_indices]]
            hxs.append(new_post_entropy)

        # Compute information gain
        hxs = np.asarray(hxs)
        hxs = hxs.reshape(budget, len(available_indices))
        eigs = []
        model.load()
        model.save_path_dir = og_save_dir
        for i, j in enumerate(available_indices):
            hx = entropy(dataset_x[j], model)
            eigs.append(hx - np.mean(hxs[:, i]))
        # plt.plot(eigs)
        # plt.scatter(last_selected_indices, 0)
        # plt.savefig("eig.png")
        if select_size == 1:
            best_indices = [random_argmax(eigs)]
        else:
            best_indices = top_k_idx(eigs, select_size)
        proposal = np.asarray(available_indices)[best_indices]

        return proposal


""" BAX acquisition for super-level set estimation.
    Note: c is the threshold for the super-level set."""


class LevelSetBaxAcquisition(BaseBatchAcquisitionFunction):
    def __call__(
        self,
        dataset_x: AbstractDataSource,
        select_size: int,
        available_indices: List[AnyStr],
        last_selected_indices: List[AnyStr] = None,
        model: AbstractBaseModel = None,
        budget=20,
        c=1.0,
    ) -> List:
        # sample top k outputs from model
        outputs = []
        hxs = []
        # save OG model state
        og_save_dir = model.save_path_dir
        model.save_path_dir = "/tmp/model"
        model.save()
        n = len(dataset_x)
        for i in range(budget):
            model.load()
            f = model.sample(dataset_x)
            out = level_set(f, c)
            outputs.append(out)
            new_x = np.concatenate([model.X, dataset_x[out]])
            new_y = np.concatenate([model.mu, f[out]])
            model.fit(new_x, new_y)
            new_post_entropy = [entropy(x, model) for x in dataset_x[available_indices]]
            hxs.append(new_post_entropy)

        # Compute information gain
        hxs = np.asarray(hxs)
        hxs = hxs.reshape(budget, len(available_indices))
        eigs = []
        model.load()
        model.save_path_dir = og_save_dir
        for i, j in enumerate(available_indices):
            hx = entropy(dataset_x[j], model)
            eigs.append(hx - np.mean(hxs[:, i]))
        best_indices = top_k_idx(eigs, select_size)
        proposal = np.asarray(available_indices)[best_indices]

        return proposal


"""BAX acquisition for the SubsetSelect algorithm, which maximizes
    the expected value of the maximal output of an unknown random function
    over some fixed set, where we assume some prior knowledge of the distribution
    from which this random function is drawn."""


class SubsetSelectBaxAcquisition(BaseBatchAcquisitionFunction):
    def __call__(
        self,
        dataset_x: AbstractDataSource,
        select_size: int,
        available_indices: List[AnyStr],
        cumulative_indices: List[AnyStr] = None,
        model: AbstractBaseModel = None,
        budget=20,
        noise="multiplicative",
        subset_size=5,
    ) -> List:
        # sample top k outputs from model
        outputs = []
        hxs = []
        # save OG model state
        og_save_dir = model.save_path_dir
        model.save_path_dir = "/tmp/model"
        model.save()
        n = len(dataset_x)
        if noise == "additive":
            sampler = lambda f: gaussian_noise_sampler(dataset_x, f)
        elif noise == "multiplicative":
            sampler = lambda f: bernoulli_noise_sampler(dataset_x, f)
        for i in range(budget):
            model.load()
            f = model.sample(dataset_x)
            out = subset_select(f, sampler, subset_size)
            outputs.append(out)
            new_x = np.concatenate([model.X, dataset_x[out]])
            new_y = np.concatenate([model.mu, f[out]])
            model.fit(new_x, new_y)
            new_post_entropy = [entropy(x, model) for x in dataset_x[available_indices]]
            hxs.append(new_post_entropy)

        # Compute information gain
        hxs = np.asarray(hxs)
        hxs = hxs.reshape(budget, len(available_indices))
        eigs = []
        model.load()
        model.save_path_dir = og_save_dir
        for i, j in enumerate(available_indices):
            hx = entropy(dataset_x[j], model)
            eigs.append(hx - np.mean(hxs[:, i]))
        if select_size == 1:
            best_indices = [random_argmax(eigs)]
        else:
            best_indices = top_k_idx(eigs, select_size)
        proposal = np.asarray(available_indices)[best_indices]
        return proposal


# Implicitly assumes a single d-dimensional data point
def entropy(x, model):
    mu, var = model.predict(x.reshape(1, -1))
    var = var.reshape(())
    h = 0.5 * np.log(2 * np.pi * var) + 0.5
    return h


def top_k_idx(v, k):
    idxes = np.argsort(v)[-k:]
    return idxes


def level_set(v, c):
    idxes = np.where(v > c)
    return idxes


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


def gaussian_noise_sampler(x, fx, lengthscale=1.0, seed=None):
    outer_state = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
    cov = rbf(x, x, lengthscale=lengthscale)
    eta = np.random.multivariate_normal(np.zeros(fx.shape), cov)
    if seed is not None:
        np.random.set_state(outer_state)
    return np.maximum(0, fx + eta)


def bernoulli_noise_sampler(x, fx, lengthscale=1.0, seed=None):
    outer_state = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
    cov = rbf(x, x, lengthscale=lengthscale)
    l = np.random.multivariate_normal(np.zeros(fx.shape), cov)
    p = 1 / (1 + np.exp(-l))
    eta = np.random.binomial(1, p)
    if seed is not None:
        np.random.set_state(outer_state)
    return np.maximum(0, fx * eta)


def random_argmax(vals):
    max_val = np.max(vals)
    idxes = np.where(vals == max_val)[0]
    return np.random.choice(idxes)


def main():
    return


if __name__ == "__main__":
    main()
