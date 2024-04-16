"""
Copyright 2022 Arash Mehrjou, GlaxoSmithKline plc, Clare Lyle, University of Oxford, Pascal Notin, University of Oxford
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
import torch
import os
import numpy as np
import pickle as pkl
from typing import AnyStr, Type
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
import pdb 
import matplotlib.pyplot as plt

import slingpy as sp
from slingpy.utils.logging import info, warn
from slingpy.data_access.data_sources.hdf5_data_source import HDF5DataSource


class BayesianLinearModel:
    def __init__(
        self,
        input_dims: int,
        prior_sigma: float,
        noise_sigma: float,
        save_path_dir: AnyStr,
    ):
        self.input_dims = input_dims
        self.prior_sigma = prior_sigma
        self.noise_sigma = noise_sigma
        self.w = np.zeros(self.input_dims)
        self.Lambda = 0.01 * np.eye(self.input_dims)
        self.save_path_dir = save_path_dir

    def fit(self, X, y):
        info(X.shape, y.shape, self.w.shape, self.Lambda.shape)
        Lambda_n = X.T @ X + self.Lambda
        beta = np.linalg.pinv(X.T @ X) @ X.T @ y
        self.w = np.linalg.inv(Lambda_n) @ (X.T @ X) @ beta
        self.Lambda = Lambda_n
        info("Error:", np.linalg.norm(y - np.mean(y)), np.linalg.norm(X @ self.w - y))
        return self.w

    def get_save_file_extension(self, *args):
        return "blr_model.pkl"

    def get_model_save_file_name(self):
        return os.path.join(self.save_path_dir, self.get_save_file_extension())

    def load(self, *args):
        if os.path.exists(self.get_model_save_file_name()):
            with open(self.get_model_save_file_name(), "rb") as f:
                w = pkl.load(f)
                f.close()
            self.w = w

    def save(self, *args):
        if not os.path.exists(self.save_path_dir):
            os.makedirs(self.save_path_dir)
        with open(self.get_model_save_file_name(), "wb") as f:
            pkl.dump(self.w, f)
            f.close()
        pass

    def get_hyperopt_parameter_ranges(self):
        hyperopt_ranges = {
            "prior_sigma": (0.5,1.0,2.0),
            "noise_sigma": (0.01,0.1,1.0),
        }
        return hyperopt_ranges


class GaussianProcessModel:
    def __init__(
        self,
        kernel,
        noise_sigma: float,
        save_path_dir: AnyStr,
        kernel_lengthscale: float = 0.1,
    ):
        if kernel is None:
            kernel = lambda x, y: rbf(x, y, kernel_lengthscale)
        self.kernel = kernel
        self.noise_sigma = noise_sigma
        self.save_path_dir = save_path_dir
        self.X = None
        self.mu = None
        self.Kinv = None
        self.Kinvf = None

    def fit(self, X, y):
        if X is None or y is None:
            self.Kinvf = 0
            self.mu = 0
            return None
        self.Kinv = np.linalg.inv(
            self.kernel(X, X) + self.noise_sigma * np.eye(X.shape[0])
        )
        self.mu = y
        self.X = X
        self.Kinvf = self.Kinv @ y
        return self.Kinvf

    def predict(self, x, return_std_and_margin=True):
        if self.X is None:
            return np.zeros(x.shape), self.kernel(x=x, y=x)
        kxX = self.kernel(self.X, x)
        mu = kxX.T @ self.Kinvf
        sigma2 = self.kernel(x, x) - kxX.T @ self.Kinv @ kxX
        return mu, sigma2

    def get_save_file_extension(self, *args):
        return "gp_model.pkl"

    def get_model_save_file_name(self):
        return os.path.join(self.save_path_dir, self.get_save_file_extension())

    def load(self, *args):
        if os.path.exists(self.get_model_save_file_name()):
            with open(self.get_model_save_file_name(), "rb") as f:
                w = pkl.load(f)
                f.close()
            self.Kinvf = w["Kinvf"]
            self.mu = w["mu"]
            self.X = w["X"]
            self.Kinv = w["Kinv"]
        return self

    def save(self, *args):
        if not os.path.exists(self.save_path_dir):
            os.makedirs(self.save_path_dir)
        with open(self.get_model_save_file_name(), "wb") as f:
            pkl.dump(
                {"Kinvf": self.Kinvf, "mu": self.mu, "X": self.X, "Kinv": self.Kinv}, f
            )
            f.close()

    def sample(self, x):
        mean, cov = self.predict(x, return_std_and_margin=True)
        f = np.random.multivariate_normal(mean, cov)
        return f
    
    def get_hyperopt_parameter_ranges(self):
        hyperopt_ranges = {
            "noise_sigma": (0.01, 0.1, 1.0),
            "kernel_lengthscale": (0.01, 0.1, 1.0),
        }
        return hyperopt_ranges

def rbf(x, y, lengthscale=0.1, sigma_f=1.5):
    if len(x.shape)==1 or len(x.shape)==0:
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        return sigma_f**2 * np.exp(- (x - y.T)**2/(2 * lengthscale))
    elif len(x.shape)==2:
        outs = np.zeros((x.shape[0], y.shape[0]))
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                outs[i,j] =  np.linalg.norm(x[i] - y[j])**2/lengthscale
        k = sigma_f**2 * np.exp(-1*outs/2)
        return k
    else:
        raise NotImplementedError


if __name__ == "__main__":
    print(rbf( np.ones((1,2)), 5*np.ones((1,2)), lengthscale=1.0))