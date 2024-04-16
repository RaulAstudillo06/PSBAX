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

import os
import torch
import pickle as pkl
import numpy as np
from typing import Any, AnyStr, Dict, List, Tuple, Union, Optional
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import slingpy as sp
from slingpy.models import torch_model
from slingpy.data_access.data_sources.hdf5_data_source import HDF5DataSource

# from discobax.apps import abstract_base_application as aba
# from discobax.models import toy_experiment_models
# from discobax.models import meta_models
# from discobax.models import pytorch_models

from . import abstract_base_application as aba
from ..models import toy_experiment_models
from ..models import meta_models
from ..models import pytorch_models

from genedisco.datasets.features.achilles import Achilles
from genedisco.datasets.features.ccle_protein_quantification import (
    CCLEProteinQuantification,
)

# from discobax.data.ccle_protein_quantification import CCLEProteinQuantification
from genedisco.datasets.features.string_embedding import STRINGEmbedding
from genedisco.datasets.screens.schmidt_2021_t_cells_ifng import Schmidt2021TCellsIFNg
from genedisco.datasets.screens.schmidt_2021_t_cells_il2 import Schmidt2021TCellsIL2
from genedisco.datasets.screens.sanchez_2021_neurons_tau import Sanchez2021NeuronsTau
from genedisco.datasets.screens.zhuang_2019_nk_cancer import Zhuang2019NKCancer
from genedisco.datasets.screens.zhu_2021_sarscov2_host_factors import (
    Zhu2021SARSCoV2HostFactors,
)

SklearnRandomForestRegressor = meta_models.SklearnRandomForestRegressor


def update_dictionary_keys_with_prefixes(input_dict: Dict[AnyStr, Any], prefix: AnyStr):
    """Adds a prefix to the keys of a dictionary."""
    output_dict = dict((prefix + key, value) for (key, value) in input_dict.items())
    return output_dict


class CustomLoss(sp.TorchLoss):
    def __init__(self):
        self.loss_fn = torch.nn.MSELoss()

    def __call__(
        self, y_pred: List[torch.Tensor], y_true: List[torch.Tensor]
    ) -> torch.Tensor:
        loss = self.loss_fn(y_pred[0].flatten(), y_true[0].flatten().double())
        return loss


class SingleCycleApplication(aba.AbstractBaseApplication):
    DATASET_NAMES = [
        "schmidt_2021_ifng",
        "schmidt_2021_il2",
        "zhuang_2019_nk",
        "sanchez_2021_tau",
        "zhu_2021_sarscov2",
    ]

    FEATURE_SET_NAMES = ["achilles", "ccle", "string"]

    def __init__(
        self,
        dataset_name: AnyStr = DATASET_NAMES[0],
        feature_set_name: AnyStr = FEATURE_SET_NAMES[0],
        output_directory: AnyStr = "",
        cache_directory: AnyStr = "",
        schedule_on_slurm: bool = False,
        remote_execution_time_limit_days: int = 3,
        remote_execution_mem_limit_in_mb: int = 2048,
        remote_execution_virtualenv_path: AnyStr = "",
        remote_execution_num_cpus: int = 1,
        split_index_outer: int = 0,
        split_index_inner: int = 0,
        num_splits_outer: int = 2,
        num_splits_inner: int = 2,
        model_name: AnyStr = "mlp",
        evaluate_against: AnyStr = "test",
        selected_indices_file_path: AnyStr = "",
        test_indices_file_path: AnyStr = "",
        train_ratio: float = 0.8,
        hyperopt: bool = False,
        num_hyperopt_runs: int = 15,
        hyperopt_offset: int = 0,
        hyperopt_metric_name: AnyStr = "MeanAbsoluteError",
        train: bool = True,
        gp_noise_sigma: float = 0.1,  # gp hyperparams
        gp_kernel_lengthscale: float = 0.1,  # gp hyperparams
        lbm_prior_sigma: float = 1.0,  # linear bayesian model hyperparams
        lbm_noise_sigma: float = 0.1,  # linear bayesian model hyperparams
        rf_max_depth: int = -1,  # randomforest hyperparams
        rf_num_estimators: int = 100,  # ensemble_model_hyperparms
        dn_num_layers: int = 2,  # deep net hyperparams
        dn_hidden_layer_size: int = 8,  # deep net hyperparams
        seed: int = 0,
    ):
        model_hyperparams = {
            "gp_noise_sigma": gp_noise_sigma,
            "gp_kernel_lengthscale": gp_kernel_lengthscale,
            "lbm_prior_sigma": lbm_prior_sigma,
            "lbm_noise_sigma": lbm_noise_sigma,
            "rf_max_depth": rf_max_depth,
            "rf_num_estimators": rf_num_estimators,
            "dn_num_layers": dn_num_layers,
            "dn_hidden_layer_size": dn_hidden_layer_size,
        }
        self.ignore_param_names = [
            "available_indices",
            "selected_indices",
            "test_indices",
        ]
        self.gp_noise_sigma = gp_noise_sigma
        self.gp_kernel_lengthscale = gp_kernel_lengthscale
        self.lbm_prior_sigma = lbm_prior_sigma
        self.lbm_noise_sigma = lbm_noise_sigma
        self.rf_max_depth = rf_max_depth
        self.rf_num_estimators = rf_num_estimators
        self.dn_num_layers = dn_num_layers
        self.dn_hidden_layer_size = dn_hidden_layer_size
        self.train = train
        self.cache_directory = cache_directory
        self.output_directory = output_directory
        self.test_indices_file_path = test_indices_file_path
        self.selected_indices_file_path = selected_indices_file_path
        self.train_ratio = train_ratio
        self.dataset_name = dataset_name
        self.feature_set_name = feature_set_name
        self.hyperopt_offset = hyperopt_offset
        self.hyperopt_metric_name = hyperopt_metric_name
        self.seed = seed
        self.model_name = model_name
        self.model = self.get_model(**model_hyperparams)

        with open(self.test_indices_file_path, "rb") as fp:
            self.test_indices = pkl.load(fp)
        with open(self.selected_indices_file_path, "rb") as fp:
            self.selected_indices = pkl.load(fp)

        super(SingleCycleApplication, self).__init__(
            output_directory=output_directory,
            schedule_on_slurm=schedule_on_slurm,
            split_index_outer=split_index_outer,
            split_index_inner=split_index_inner,
            num_splits_outer=num_splits_outer,
            num_splits_inner=num_splits_inner,
            evaluate_against=evaluate_against,
            single_run=True,
            nested_cross_validation=False,
            save_predictions=False,
            save_predictions_file_format="tsv",
            evaluate=False,
            hyperopt=hyperopt,
            num_hyperopt_runs=num_hyperopt_runs,
            seed=seed,
            remote_execution_time_limit_days=remote_execution_time_limit_days,
            remote_execution_virtualenv_path=remote_execution_virtualenv_path,
            remote_execution_num_cpus=remote_execution_num_cpus,
            remote_execution_mem_limit_in_mb=remote_execution_mem_limit_in_mb,
        )

    def get_metrics(self, set_name: AnyStr) -> List[sp.AbstractMetric]:
        return [
            sp.metrics.MeanAbsoluteError(),
            sp.metrics.RootMeanSquaredError(),
            sp.metrics.SymmetricMeanAbsolutePercentageError(),
            sp.metrics.SpearmanRho(),
            sp.metrics.TopKRecall(0.1, 0.1),
        ]

    @staticmethod
    def get_dataset_y(dataset_name, cache_directory):
        # if dataset_name == SingleCycleApplication.DATASET_NAMES[0]:
        #     dataset_y = Schmidt2021TCellsIFNg.load_data(cache_directory)
        # elif dataset_name == SingleCycleApplication.DATASET_NAMES[1]:
        #     dataset_y = Schmidt2021TCellsIL2.load_data(cache_directory)
        # elif dataset_name == SingleCycleApplication.DATASET_NAMES[2]:
        #     dataset_y = Zhuang2019NKCancer.load_data(cache_directory)
        # elif dataset_name == SingleCycleApplication.DATASET_NAMES[3]:
        #     dataset_y = Sanchez2021NeuronsTau.load_data(cache_directory)
        # elif dataset_name == SingleCycleApplication.DATASET_NAMES[4]:
        #     dataset_y = Zhu2021SARSCoV2HostFactors.load_data(cache_directory)
        # else:
        #     raise NotImplementedError(f"{dataset_name} is not implemented.")
        h5_file = os.path.join(cache_directory, f"{dataset_name}.h5")
        dataset_y = HDF5DataSource(h5_file, duplicate_merge_strategy=sp.MeanMergeStrategy())

        return dataset_y

    @staticmethod
    def get_dataset_x(feature_set_name, cache_directory):
        if "discobax-scripts" in os.getcwd():
            os.chdir("..")
        

        if feature_set_name == SingleCycleApplication.FEATURE_SET_NAMES[0]:
            # dataset = Achilles.load_data(cache_directory)
            h5_file = os.path.join(cache_directory, "achilles.h5")
            dataset = HDF5DataSource(h5_file, fill_missing_value=0)
            dataset_x = sp.CompositeDataSource([dataset])

        elif feature_set_name == SingleCycleApplication.FEATURE_SET_NAMES[1]:
            dataset = CCLEProteinQuantification.load_data(cache_directory)
        elif feature_set_name == SingleCycleApplication.FEATURE_SET_NAMES[2]:
            dataset = STRINGEmbedding.load_data(cache_directory)
        else:
            raise NotImplementedError()

        dataset_x = sp.CompositeDataSource([dataset])
        return dataset_x

    def load_data(self) -> Dict[AnyStr, sp.AbstractDataSource]:
        dataset_x = SingleCycleApplication.get_dataset_x(
            self.feature_set_name, self.cache_directory
        )
        dataset_y = SingleCycleApplication.get_dataset_y(
            self.dataset_name, self.cache_directory
        )

        # Subset dataset_y by the overlap of genes present in both dataset_x and dataset_y.
        avail_names = sorted(
            list(
                set(dataset_x.get_row_names()).intersection(
                    set(dataset_y.get_row_names())
                )
            )
        )
        dataset_y = dataset_y.subset(avail_names)
        dataset_x = dataset_x.subset(avail_names) # To.get_data()

        dataset_y = dataset_y.subset(self.selected_indices)

        stratifier = sp.StratifiedSplit()
        training_indices, validation_indices = stratifier.split(
            dataset_y,
            test_set_fraction=1 - self.train_ratio,
            split_index=self.split_index_inner,
        )
        if self.train_ratio == 1.0:
            return {
                "training_set_x": dataset_x.subset(training_indices),
                "training_set_y": dataset_y.subset(training_indices),
                "validation_set_x": None,
                "validation_set_y": None,
                "test_set_x": dataset_x.subset(self.test_indices),
                "test_set_y": dataset_y.subset(self.test_indices),
            }
        else:
            return {
                "training_set_x": dataset_x.subset(training_indices),
                "training_set_y": dataset_y.subset(training_indices),
                "validation_set_x": dataset_x.subset(validation_indices),
                "validation_set_y": dataset_y.subset(validation_indices),
                "test_set_x": dataset_x.subset(self.test_indices),
                "test_set_y": dataset_y.subset(self.test_indices),
            }

    def _get_model_path(self):
        return os.path.join(self.output_directory, "models")

    def get_model(self, **kwargs):
        if self.model_name == "gp":
            # Note: Current GP model only accepts 1D or 2D inputs. Does not scale to GeneDisco input feature size -- so have to do PCA to reduce dim to 2D
            save_path_dir = os.path.join(self._get_model_path(), "gpmodel")
            sp_model = toy_experiment_models.GaussianProcessModel(
                None, # kernel is None, gets initializes to rbf in model
                noise_sigma=kwargs.get("gp_noise_sigma"),
                save_path_dir=save_path_dir,
                kernel_lengthscale=kwargs.get("gp_kernel_lengthscale"),
            )
        elif self.model_name == "linear":
            # Need to adaptively update num dimensions
            save_path_dir = os.path.join(self.output_directory, "linearmodel")
            sp_model = toy_experiment_models.BayesianLinearModel(
                input_dims=SingleCycleApplication.get_dataset_x(
                    self.feature_set_name, self.cache_directory
                ).get_shape()[0][-1],
                prior_sigma=kwargs.get("lbm_prior_sigma"),
                noise_sigma=kwargs.get("lbm_noise_sigma"),
                save_path_dir=save_path_dir,
            )
        elif self.model_name == "randomforest":
            rf_max_depth = kwargs.get("rf_max_depth")
            if rf_max_depth == -1:
                rf_max_depth = None
            sp_model = SklearnRandomForestRegressor(
                base_module=RandomForestRegressor(
                    n_estimators=kwargs.get("rf_num_estimators"),
                    max_depth=rf_max_depth,
                    random_state=self.seed,
                )
            )
        elif self.model_name == "bayesian_mlp":
            super_base_module = torch_model.TorchModel(
                base_module=pytorch_models.BayesianMLP(
                    input_size=SingleCycleApplication.get_dataset_x(
                        self.feature_set_name, self.cache_directory
                    ).get_shape()[0][-1],
                    hidden_size=self.dn_hidden_layer_size,
                ),
                loss=CustomLoss(),
                batch_size=64,
                num_epochs=100,
            )
            sp_model = meta_models.PytorchMLPRegressorWithUncertainty(
                model=super_base_module, num_target_samples=100
            )
        else:
            raise NotImplementedError
        return sp_model

    def train_model(self) -> Optional[sp.AbstractBaseModel]:
        # X = self.datasets.training_set_x.get_data()[0]
        # y = self.datasets.training_set_y.get_data()[0]
        self.model.fit(self.datasets.training_set_x, self.datasets.training_set_y)
        # self.model.fit(X, y)
        return self.model

    def get_hyperopt_parameter_ranges(self) -> Dict[AnyStr, Union[List, Tuple]]:
        """
        Get hyper-parameter optimization ranges.train_model

        Returns:
            A dictionary with each item corresponding to a named hyper-parameter and its associated discrete
            (represented as a Tuple) or continuous (represented as a List[start, end]) value range.
        """
        model_hyperopt_parameter_ranges = self.get_model_hyperparameter_ranges()
        hyperopt_ranges = {}
        hyperopt_ranges.update(model_hyperopt_parameter_ranges)
        return hyperopt_ranges

    def get_model_hyperparameter_ranges(self) -> Dict[AnyStr, Union[AnyStr, int]]:
        """
        Get the hyper-parameter optimization ranges from the model.

        Returns:
            A dictionary with each item corresponding to a named hyper-parameter and its associated discrete
            (represented as a Tuple) or continuous (represented as a List[start, end]) value range.
        """
        prefixes = {
            "randomforest": "rf_",
            "bayesian_mlp": "dn_",
            "linear": "lbm_",
            "gp": "gp_",
        }
        model_hyperopt_parameter_ranges = self.model.get_hyperopt_parameter_ranges()
        model_hyperopt_parameter_ranges = update_dictionary_keys_with_prefixes(
            model_hyperopt_parameter_ranges, prefixes[self.model_name]
        )
        return model_hyperopt_parameter_ranges


if __name__ == "__main__":
    app = sp.instantiate_from_command_line(SingleCycleApplication)
    results = app.run()
