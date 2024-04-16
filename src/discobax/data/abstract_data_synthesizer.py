"""
Copyright 2022 Arash Mehrjou, GlaxoSmithKline plc
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
import os
import numpy as np
import sklearn.datasets as dt
from abc import ABCMeta, abstractmethod
from slingpy.utils.path_tools import PathTools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, AnyStr
import pickle


class AbastractDataSynthesizer(metaclass=ABCMeta):
    """The abstract base class for data synthesizers."""

    def __init__(
        self,
        mode: AnyStr = "regression",
        n_samples: int = 100,
        n_features: Optional[int] = 2,
        n_targets: Optional[int] = 1,
        noise_std: Optional[float] = 0.0,
        seed: Optional[int] = 909,
        saved_directory: Optional[AnyStr] = None,
    ):

        """
        Args: Initialize the data synthesizer class.

        Args:
            mode (str): [classification|regression|friedman1|friedman2|friedman3].
            n_samples (int): The number of generated samples.
            n_features (int): The number of features.
            n_targets (int): The dimension of the y output vector associated with a sample (relevant for regression only).
            noise (float): The standard deviation of the gaussian noise applied to the output.
            seed (int): The random seed. Keep it fixed to generate the same datasets.
            saved_directory (str): save data on disk.
        """
        self.mode = mode
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_targets = n_targets
        self.noise_std = noise_std
        self.seed = seed
        self.saved_directory = saved_directory
        self.dataset = None

    @abstractmethod
    def _generate_data(self):
        """Generate the data matrix and assing it to self.dataset

        Returns: None
        """
        raise NotImplementedError()

    def load_data(self):
        if self.dataset is None:
            self._generate_data()
        return self.dataset

    def save_data(self):
        if self.dataset is None:
            raise ValueError("The dataset has not yet been created.")
        if self.saved_directory:
            datadict = dict()
            datadict["metadata"] = self.get_dataset_metadata()
            datadict["data"] = self.dataset
            filepath = os.path.join(self.saved_directory, f"datadict_{self.mode}.pkl")
            PathTools.mkdir_if_not_exists(self.saved_directory)
            with open(filepath, "wb") as fp:
                pickle.dump(datadict, fp, pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError("Saving directory for dataset is not provided.")

    def get_dataset_metadata(self) -> dict:
        """Returns a dictionary containing the metadata of the dataset.

        Returns:
            dataset_metadata (dict)
        """
        dataset_dict = {
            "mode": self.mode,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_targets": self.n_targets,
            "noise_std": self.noise_std,
            "seed": self.seed,
        }
        return dataset_dict

    @abstractmethod
    def get_groundtruth_function(self) -> Callable[[np.ndarray], np.ndarray]:
        """Return the ground truth function object."""
        pass
