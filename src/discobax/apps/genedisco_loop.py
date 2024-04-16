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

import pickle as pkl
import os
import sys
import torch
import numpy as np
import pandas as pd
import slingpy as sp

from typing import AnyStr, Dict, List, Optional
from collections import namedtuple
from slingpy import AbstractMetric, AbstractBaseModel, AbstractDataSource
from slingpy.utils.logging import info
from slingpy.utils.path_tools import PathTools
from slingpy.models.abstract_base_model import AbstractBaseModel
from slingpy.data_access.data_sources.hdf5_data_source import HDF5DataSource
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal

# print(os.getcwd())
# print directories in the current working directory
# print(os.listdir(os.getcwd()))



sys.path.append(os.getcwd())
print(sys.path)
from src.discobax.apps import abstract_base_application as aba
from src.discobax.apps.genedisco_single_cycle_experiment import SingleCycleApplication
from src.discobax.models import clustering
from src.discobax.methods.bax_acquisition import bax_sampling
from src.discobax.methods.posterior_sampling import posterior_sampling

DataSet = namedtuple(
    "DataSet",
    "training_set_x, training_set_y, validation_set_x, validation_set_y, test_set_x, test_set_y",
)


class GeneDiscoLoop(aba.AbstractBaseApplication):
    def __init__(
        self,
        cache_directory: AnyStr = "discobax/data",
        output_directory: AnyStr = "discobax/output",
        dataset_name: AnyStr = "sanchez_2021_tau", #  ["zhu_2021_sarscov2", "schmidt_2021_ifng", "schmidt_2021_il2", "zhuang_2019", "sanchez_2021_tau"]
        feature_set_name: AnyStr = "achilles",  # Dattaset used for interventtion representation ["achilles" "ccle" "string"]
        model_name: AnyStr = "bayesian_mlp",
        acquisition_function_name: AnyStr = "discobax",
        num_init: int = 20,
        acquisition_batch_size: int = 20,
        num_active_learning_cycles: int = 20,
        test_ratio: float = 0.0,
        seed: int = 1000,
        topk_percent: float = 0.1,
        bax_topk_kvalue: int = 5,
        bax_level_set_c: float = 1.0,
        bax_subset_select_subset_size: int = 20,
        bax_noise_type: str = "additive",
        eta_budget: int = 20,
        bax_noise_lengthscale: float = 1.0,
        bax_noise_outputscale: float = 1.0,
        bax_num_samples_EIG: int = 20,
        bax_num_samples_entropy: int = 20,
        bax_entropy_average_mode: str = "arithmetic",
        bax_batch_selection_mode: str = "topk_EIG",
        num_topk_clusters: int = 20,
    ):
        self.acquisition_function_name = acquisition_function_name

        self.run_name = "_".join(
            [
                acquisition_function_name,
                str(seed),
            ]
        )

        self.output_directory = os.path.join(output_directory, dataset_name, self.run_name)
        os.makedirs(self.output_directory, exist_ok=True)

        self.temp_folder_name = self.output_directory + os.sep + "tmp"
        # PathTools.mkdir_if_not_exists(self.temp_folder_name)
        os.makedirs(self.temp_folder_name, exist_ok=True)
        # self.performance_file_location = performance_file_location
        self.performance_file_location = os.path.join(self.output_directory, "performance_file.csv")
        self.discobax_metric_file_location = os.path.join(self.output_directory, "metric.txt")

        self.bax_topk_kvalue = bax_topk_kvalue
        self.bax_subset_select_subset_size = bax_subset_select_subset_size
        self.bax_level_set_c = bax_level_set_c
        self.bax_noise_type = bax_noise_type
        self.eta_budget = eta_budget
        self.bax_noise_lengthscale = bax_noise_lengthscale
        self.bax_noise_outputscale = bax_noise_outputscale
        self.bax_num_samples_EIG = bax_num_samples_EIG
        self.bax_num_samples_entropy = bax_num_samples_entropy
        self.bax_entropy_average_mode = bax_entropy_average_mode
        self.bax_batch_selection_mode = bax_batch_selection_mode

        self.acquisition_function = self.get_acquisition_function()
        self.num_init = num_init
        self.acquisition_batch_size = acquisition_batch_size
        self.num_active_learning_cycles = num_active_learning_cycles
        self.dataset_name = dataset_name
        self.feature_set_name = feature_set_name
        self.model_name = model_name
        self.test_ratio = test_ratio
        self.cache_directory = cache_directory
        self.topk_percent = topk_percent
        self.topk_indices = None
        self.topk_clusters_to_item = None
        self.topk_item_to_clusters = None
        self.num_topk_clusters = num_topk_clusters
        super(GeneDiscoLoop, self).__init__(
            output_directory=self.output_directory,
            seed=seed,
            evaluate=False,
            hyperopt=False,
            single_run=True,
            save_predictions=False,
        )

        self.kernel = None

    def get_acquisition_function(self):
        if self.acquisition_function_name == "discobax":
            return bax_sampling.BaxAcquisition(
                subset_size=self.bax_subset_select_subset_size,
                noise_type=self.bax_noise_type,
                noise_budget=self.eta_budget,
            )

        elif self.acquisition_function_name == "ps":
            return posterior_sampling.PSAcquisition(
                subset_size=self.bax_subset_select_subset_size,
                noise_type=self.bax_noise_type,
                noise_budget=self.eta_budget,
            )
        else:
            raise NotImplementedError()

    def initialize_pool(self):
        dataset_x = SingleCycleApplication.get_dataset_x(
            self.feature_set_name, self.cache_directory
        )
        
        dataset_y = SingleCycleApplication.get_dataset_y(
            self.dataset_name, self.cache_directory
        )
        

        available_indices = sorted(
            list(
                set(dataset_x.get_row_names()).intersection(
                    set(dataset_y.get_row_names())
                )
            )
        )
        test_indices = sorted(
            list(
                np.random.choice(
                    available_indices,
                    size=int(self.test_ratio * len(available_indices)),
                    replace=False,
                )
            )
        )
        available_indices = sorted(list(set(available_indices) - set(test_indices)))
        dataset_x = dataset_x.subset(available_indices) # pd.DataFrame(dataset_x.get_data()[0]) -> (17176, 808)
        dataset_y = dataset_y.subset(available_indices) # pd.DataFrame(dataset_y.get_data()[0]) -> (17176, 1)
        self.topk_indices = self.get_topk_indices(dataset_y, self.topk_percent)

        # Get basic stats
        try:
            df = pd.DataFrame(dataset_y.get_data()[0])
            print(df.describe())
        except:
            print("Errors getting stats")

        return dataset_x, available_indices, test_indices

    def load_data(self) -> Dict[AnyStr, AbstractDataSource]:
        return {}

    def get_metrics(self, set_name: AnyStr) -> List[AbstractMetric]:
        return []

    def get_model(self, **kwargs):
        return None

    def get_topk_indices(
        self, dataset_y: AbstractDataSource, topk_percent: float = 0.1
    ):
        y_indices = dataset_y.get_row_names()
        y_values = pd.DataFrame(
            dataset_y.get_data()[0].flatten(), index=y_indices, columns=["y_values"]
        )
        y_values.sort_values(by=["y_values"], ascending=False, axis=0, inplace=True)
        num_k = int(topk_percent * len(y_values.y_values))
        topk_indices = list(y_values.iloc[:num_k].index) # 1717 = 0.1 * 17176
        return topk_indices

    def one_iter(self):
        pass

    def train_model(self) -> Optional[AbstractBaseModel]:
        single_cycle_application_args = {
            "model_name": self.model_name,
            "seed": self.seed,
        }
        cumulative_indices = []
        dataset_x, available_indices, test_indices = self.initialize_pool()
        dataset_y = SingleCycleApplication.get_dataset_y(
            self.dataset_name, self.cache_directory
        )
        dataset_y = dataset_y.subset(available_indices)
        last_selected_indices = sorted(
            list(
                np.random.choice(
                    available_indices,
                    size=int(self.num_init),
                    replace=False,
                )
            )
        )
        cumulative_indices += last_selected_indices
        discobax_metrics = []
        result_records = list()
        
        for cycle_index in range(self.num_active_learning_cycles):
            current_cycle_directory = os.path.join(
                self.output_directory, f"cycle_{cycle_index}"
            )

            PathTools.mkdir_if_not_exists(current_cycle_directory)

            cumulative_indices_file_path = os.path.join(
                current_cycle_directory, "selected_indices.pkl"
            )
            with open(cumulative_indices_file_path, "wb") as fp:
                pkl.dump(cumulative_indices, fp)
            test_indices_file_path = os.path.join(
                current_cycle_directory, "test_indices.pkl"
            )
            with open(test_indices_file_path, "wb") as fp:
                pkl.dump(test_indices, fp)
            app = SingleCycleApplication(
                dataset_name=self.dataset_name,
                feature_set_name=self.feature_set_name,
                cache_directory=self.cache_directory,
                output_directory=current_cycle_directory,
                train_ratio=1.0,
                selected_indices_file_path=cumulative_indices_file_path,
                test_indices_file_path=test_indices_file_path,
                **single_cycle_application_args,
            )
            results = app.run().run_result
            info(results.test_scores, "test scores")
            result_records.append(results.test_scores)
            available_indices = sorted(
                list(set(available_indices) - set(last_selected_indices))
            )

            trained_model = app.model.load(results.model_path)
            
            if self.acquisition_function.mvn is None:
                mvn = self.get_mvn(dataset_x)
                self.acquisition_function.mvn = mvn
            
            discobax_metric = self.get_discobax_objective(
                trained_model, 
                dataset_x, 
                dataset_y,
                budget=self.eta_budget,
            )
            discobax_metrics.append(discobax_metric)

            if trained_model is None:
                print("Could not find trained model at specified path in results")
                trained_model = app.get_model() # getting gp model

            last_selected_indices = self.acquisition_function(
                dataset_x=dataset_x,
                acquisition_batch_size=self.acquisition_batch_size,
                available_indices=available_indices,
                last_selected_indices=last_selected_indices,
                cumulative_indices=cumulative_indices,
                model=trained_model,
                dataset_y=dataset_y,
                temp_folder_name=self.temp_folder_name,
            )
            cumulative_indices.extend(last_selected_indices)
            cumulative_indices = sorted(list(set(cumulative_indices)))
            assert len(last_selected_indices) == self.acquisition_batch_size
            
            performance_file_exists = os.path.exists(self.performance_file_location)
            with open(self.performance_file_location, "a") as performance_record:
                if not performance_file_exists:
                    header = "dataset_name,feature_set_name,model_name,acquisition_function_name,acquisition_batch_size,num_active_learning_cycles,seed,num_total_items"
                    performance_record.write(header + "\n")
                record = ",".join(
                    [
                        str(x)
                        for x in [
                            self.dataset_name,
                            self.feature_set_name,
                            self.model_name,
                            self.acquisition_function_name,
                            self.acquisition_batch_size,
                            self.num_active_learning_cycles,
                            self.seed,
                            len(dataset_y),
                        ]
                    ]
                )
                performance_record.write(record + "\n")
            
            save_metric = np.atleast_1d(discobax_metrics)
            np.savetxt(self.discobax_metric_file_location, save_metric)

        results_path = os.path.join(self.output_directory, "results.pkl")
        with open(results_path, "wb") as fp:
            pkl.dump(result_records, fp)
        return None

    def get_mvn(self, dataset_x):
        x = dataset_x.get_data()[0]
        n = x.shape[0]
        x_torch = torch.from_numpy(x)
        with torch.no_grad():
            kernel = ScaleKernel(
                RBFKernel(lengthscale=self.bax_noise_lengthscale), outputscale=self.bax_noise_outputscale
            )
            cov = kernel(x_torch)
        mean = torch.zeros(n)
        mvn = MultivariateNormal(mean, cov)
        self.mvn = mvn
        return mvn
    
    def get_discobax_objective(self, model, dataset_x, dataset_y, budget=20, nonneg=False):
        x = dataset_x.get_data()[0]
        n = x.shape[0]

        posterior_mean = model.predict(dataset_x, return_std_and_margin=True)[0]
        y = dataset_y.get_data()[0].squeeze() # (17176,)
        fout_lst = []
        if self.bax_noise_type == "additive":
            etas = self.mvn.rsample(torch.Size([budget])).detach().numpy()
            for eta in etas:
                if nonneg:
                    fout_lst.append(
                        lambda fx: np.maximum(0, fx + eta)
                    )
                else:
                    fout_lst.append(
                        lambda fx: fx + eta
                    )
        elif self.bax_noise_type == "multiplicative":
            ls = self.mvn.rsample(torch.Size([budget])).detach().numpy()
            for l in ls:
                p = 1 / (1 + np.exp(-l))
                eta = np.random.binomial(1, p)
                if nonneg:
                    fout_lst.append(
                        lambda fx: np.maximum(0, fx + eta)
                    )
                else:
                    fout_lst.append(
                        lambda fx: fx + eta
                    )
        # Run algorithm on posterior values and obtain selected indices
        post_values = np.asarray([fout(posterior_mean) for fout in fout_lst]) # (20, 17176)
        post_mean_values = np.mean(post_values, axis=0)
        post_mx = random_argmax(post_mean_values)
        post_idxes = [post_mx]
        for _ in range(self.bax_subset_select_subset_size - 1):
            e_vals = np.zeros(n)
            for j in range(n):
                test_idxes = post_idxes
                if j not in post_idxes:
                    test_idxes = post_idxes + [j]                             
                    test_idxes = np.asarray(test_idxes)
                    e_vals[j] = np.mean(np.max(post_values[:, test_idxes], axis=-1))
                
            idx_next = random_argmax(e_vals)
            post_idxes.append(idx_next)
        
        # Run algorithm on true values and obtain selected indices
        true_values = np.asarray([fout(y) for fout in fout_lst]) # (20, 17176)
        true_mean_values = np.mean(true_values, axis=0)
        true_mx = random_argmax(true_mean_values)
        true_idxes = [true_mx]
        for _ in range(self.bax_subset_select_subset_size - 1):
            e_vals = np.zeros(n)
            for j in range(n):
                test_idxes = true_idxes
                if j not in true_idxes:
                    test_idxes = true_idxes + [j]
                    test_idxes = np.asarray(test_idxes)
                    e_vals[j] = np.mean(np.max(true_values[:, test_idxes], axis=-1))
            
            idx_next = random_argmax(e_vals)
            true_idxes.append(idx_next)
        
        # Return true value of selected indices
        return [np.mean(np.max(true_values[:, post_idxes], axis=-1)), np.mean(np.max(true_values[:, true_idxes], axis=-1))]


def random_argmax(vals):
    max_val = np.max(vals)
    idxes = np.where(vals == max_val)[0]
    return np.random.choice(idxes)


    

# if __name__ == "__main__":
#     GeneDisco_loop = sp.instantiate_from_command_line(GeneDiscoLoop)
#     GeneDisco_loop.run()
