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
import os
import pickle
import torch
import pickle as pkl
import numpy as np
from typing import AnyStr, Dict, List, Optional
from collections import namedtuple
import matplotlib.pyplot as plt

import slingpy as sp
from slingpy.utils.logging import info, warn
from slingpy.utils.path_tools import PathTools
from slingpy import AbstractMetric, AbstractBaseModel, AbstractDataSource

from discobax.data import gt_data_synthesizer as ds
from discobax.methods.bayesian_optimization import ucb_sampling
from discobax.methods.bax_acquisition import bax_sampling_toy_experiment
from discobax.methods.thompson_sampling import thompson_sampling_toy_experiment
from discobax.methods.random import random
from discobax.models import toy_experiment_models

NUM_SAMPLES = 64
PLOT = True
DataSet = namedtuple(
    "DataSet",
    "training_set_x, training_set_y, validation_set_x, validation_set_y, test_set_x, test_set_y",
)


class CustomLoss(sp.TorchLoss):
    def __init__(self):
        self.loss_fn = torch.nn.MSELoss()

    def __call__(
        self, y_pred: List[torch.Tensor], y_true: List[torch.Tensor]
    ) -> torch.Tensor:
        loss = self.loss_fn(y_pred[0], y_true[0].float())
        return loss


class SingleCycleVizApplication(sp.AbstractBaseApplication):
    def __init__(
        self,
        model_name: AnyStr = "randomforest",
        acquisition_function_name: AnyStr = "random",
        output_directory: AnyStr = "",
        test_ratio: float = 0.2,
        hyperopt_children: bool = True,
        seed: int = 0,
        dataset_name: AnyStr = "mog",
        feature_set_name="",
        selected_indices_file_path="",
        test_indices_file_path="",
        **kwargs,
    ):
        self.model_name = model_name
        self.acquisition_function_name = acquisition_function_name
        self.hyperopt_children = hyperopt_children
        self.test_ratio = test_ratio
        self.dataset_name = dataset_name
        self.feature_set_name = feature_set_name
        self.selected_indices_file_path = selected_indices_file_path
        self.test_indices_file_path = test_indices_file_path
        with open(test_indices_file_path, "rb") as fp:
            self.test_indices = pickle.load(fp)
        with open(selected_indices_file_path, "rb") as fp:
            self.selected_indices = pkl.load(fp)

        super(SingleCycleVizApplication, self).__init__(
            output_directory=output_directory,
            seed=seed,
            evaluate=True,
            hyperopt=False,
            single_run=True,
            save_predictions=False,
        )

        self.initialize_pool()
        self.model = self.get_model()

    def initialize_pool(self):
        """Get the data as (x, value) pairs
        Raises:
            NotImplementedError: _description_
        """
        n = NUM_SAMPLES
        n_train = int(0.9 * n)
        available_indices = np.random.choice(n, n_train)
        test_indices = list(filter(lambda i: i not in available_indices, range(n)))
        available_indices = list(range(n))
        test_indices = available_indices

        if self.dataset_name == "mog":
            # Generate synthetic dataset
            gen_d = ds.GTDataSynthesizer(mode="mog1d", n_samples=NUM_SAMPLES)
            self.data_synthesizer = gen_d

            self.data_synthesizer._generate_data()

            dataset_x = gen_d.dataset[0]
            dataset_y = gen_d.dataset[1]

            self.dataset = DataSet(**self.load_data())
            # self.datasets = DataSet(**self.load_data())

            return dataset_x, available_indices, test_indices
        elif self.dataset_name == "sine":
            # Generate synthetic dataset
            gen_d = ds.GTDataSynthesizer(mode="sine1d", n_samples=NUM_SAMPLES)
            self.data_synthesizer = gen_d

            self.data_synthesizer._generate_data()

            dataset_x = gen_d.dataset[0]
            dataset_y = gen_d.dataset[1]

            self.dataset = DataSet(**self.load_data())
            # self.datasets = DataSet(**self.load_data())

            return dataset_x, available_indices, test_indices
        elif self.dataset_name == "sine2d":
            # Generate synthetic dataset
            gen_d = ds.GTDataSynthesizer(mode="sine2d", n_samples=NUM_SAMPLES)
            self.data_synthesizer = gen_d

            self.data_synthesizer._generate_data()

            dataset_x = gen_d.dataset[0]
            dataset_y = gen_d.dataset[1]

            self.dataset = DataSet(**self.load_data())
            # self.datasets = DataSet(**self.load_data())

            return dataset_x, available_indices, test_indices
        else:
            print(self.dataset_name)
            raise NotImplementedError()

    def init_data(self):
        self.datasets = DataSet(**self.load_data())

    # This is what the active learning loop app does in genedisco as well
    def load_data(self) -> Dict[AnyStr, AbstractDataSource]:
        return {
            "training_set_x": self.data_synthesizer.dataset[0][self.selected_indices],
            "training_set_y": self.data_synthesizer.dataset[1][self.selected_indices],
            "validation_set_x": self.data_synthesizer.dataset[0][self.test_indices],
            "validation_set_y": self.data_synthesizer.dataset[1][self.test_indices],
            "test_set_x": self.data_synthesizer.dataset[0][self.test_indices],
            "test_set_y": self.data_synthesizer.dataset[1][self.test_indices],
        }

    def get_metrics(self, set_name: AnyStr) -> List[sp.AbstractMetric]:
        return [
            sp.metrics.MeanAbsoluteError(),
            sp.metrics.RootMeanSquaredError(),
            sp.metrics.SymmetricMeanAbsolutePercentageError(),
            sp.metrics.SpearmanRho(),
            sp.metrics.TopKRecall(0.2, 0.1),
        ]

    def train_model(self) -> Optional[AbstractBaseModel]:
        self.model.fit(
            self.data_synthesizer.dataset[0][self.selected_indices],
            self.data_synthesizer.dataset[1][self.selected_indices],
        )

        return self.model

    def get_model(self) -> Optional[AbstractBaseModel]:
        # Hardcoded
        if self.model_name == "gp":
            ls = 1.0 if self.dataset_name == "sine2d" else 0.1
            sp_model = toy_experiment_models.GaussianProcessModel(
                None, noise_sigma=0.01, save_path_dir="./gpmodel", kernel_lengthscale=ls
            )
        elif self.model_name == "linear":
            # Need to adaptively update num dimensions
            sp_model = toy_experiment_models.BayesianLinearModel(
                1, prior_sigma=1.0, noise_sigma=0.1, save_path_dir="./linearmodel"
            )
        return sp_model

    def get_acquisition_function(
        acquisition_function_name: AnyStr, acquisition_function_path: AnyStr
    ):
        if acquisition_function_name == "ucb":
            return ucb_sampling.UCBAcquisition()
        else:
            return random.RandomBatchAcquisitionFunction()

    def evaluate_model(
        self,
        model: AbstractBaseModel,
        dataset_x: AbstractDataSource,
        dataset_y: AbstractDataSource,
        with_print: bool = True,
        set_name: AnyStr = "",
        threshold=None,
    ):
        """
        Evaluates model performance.
        Args:
            model: The model to evaluate.
            dataset: The dataset used for evaluation.
            with_print: Whether or not to print results to stdout.
            set_name: The name of the dataset being evaluated.
            threshold: An evaluation threshold (derived in sample) for discrete classification metrics,
             or none if a threshold should automatically be selected.
        Returns:
            A dictionary with each entry corresponding to an evaluation metric with one or more associated values.
        """
        preds, cov = model.predict(dataset_x)
        uncs = np.diag(cov)
        scores = {}

        for m in self.get_metrics(""):
            s = m.evaluate(preds.reshape(-1, 1), dataset_y.reshape(-1, 1), threshold)
            scores[str(m).split(" ")[0].split(".")[-1]] = s

        # Bespoke top-k for the moment
        top_y = sorted(set(np.argsort(dataset_y)[-10:]))
        top_pred = sorted(set(np.argsort(preds)[-10:]))

        recall = len(top_y.intersection(top_pred)) / len(top_pred)
        ds_score = downstream_score(
            dataset_x, dataset_y, np.asarray(list(top_pred)).astype(int)
        )
        scores["MyRecall"] = recall
        scores["ExpectedMax"] = ds_score
        return scores


class VizLoop(sp.AbstractBaseApplication):
    ACQUISITION_FUNCTIONS = [
        "random",
        "ucb",
        "topk_bax",
        "levelset_bax",
        "subsetmax_bax",
    ]

    def __init__(
        self,
        model_name: AnyStr = "gp",
        acquisition_function_name: AnyStr = "random",
        acquisition_function_path: AnyStr = "custom",
        acquisition_batch_size: int = 1,
        num_active_learning_cycles: int = 20,
        feature_set_name: AnyStr = None,
        dataset_name: AnyStr = "sine2d",
        cache_directory: AnyStr = "cache",
        output_directory: AnyStr = "output_sine2d",
        test_ratio: float = 0.2,
        hyperopt_children: bool = True,
        schedule_on_slurm: bool = False,
        schedule_children_on_slurm: bool = False,
        remote_execution_time_limit_days: int = 1,
        remote_execution_mem_limit_in_mb: int = 2048,
        remote_execution_virtualenv_path: AnyStr = "",
        remote_execution_num_cpus: int = 1,
        seed: int = 0,
        save_fig_dir: str = "toy_figures_nd",
    ):
        self.acquisition_function_name = acquisition_function_name
        self.acquisition_function_path = acquisition_function_path

        # Ensure the output directory depends on acquisition function
        output_directory = os.path.join(
            output_directory, acquisition_function_name, str(seed)
        )
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        PathTools.mkdir_if_not_exists(output_directory)
        self.acquisition_function = VizLoop.get_acquisition_function(
            self.acquisition_function_name, self.acquisition_function_path
        )

        self.acquisition_batch_size = acquisition_batch_size
        self.num_active_learning_cycles = num_active_learning_cycles
        self.dataset_name = dataset_name
        self.feature_set_name = feature_set_name
        self.model_name = model_name
        self.hyperopt_children = hyperopt_children
        self.test_ratio = test_ratio
        self.cache_directory = cache_directory
        self.schedule_children_on_slurm = schedule_children_on_slurm
        self.save_fig_dir = save_fig_dir

        self.initialize_pool()
        if PLOT:
            print(self.data_synthesizer.dataset[1].shape, "\n plotting \n")
            plt.plot(self.data_synthesizer.dataset[1], color="black", linestyle="-.")
        super(VizLoop, self).__init__(
            output_directory=output_directory,
            seed=seed,
            evaluate=False,
            hyperopt=False,
            single_run=True,
            save_predictions=False,
            schedule_on_slurm=schedule_on_slurm,
            remote_execution_num_cpus=remote_execution_num_cpus,
            remote_execution_time_limit_days=remote_execution_time_limit_days,
            remote_execution_mem_limit_in_mb=remote_execution_mem_limit_in_mb,
            remote_execution_virtualenv_path=remote_execution_virtualenv_path,
        )

    @staticmethod
    def get_acquisition_function(
        acquisition_function_name: AnyStr, acquisition_function_path: AnyStr
    ):
        if acquisition_function_name == "random":
            return ucb_sampling.RandomPlottingAcquisition()
        elif acquisition_function_name == "ucb":
            return ucb_sampling.UCBAcquisition()
        elif acquisition_function_name == "topk_bax":
            return bax_sampling_toy_experiment.TopKBaxAcquisition()
        elif acquisition_function_name == "levelset_bax":
            return bax_sampling_toy_experiment.LevelSetBaxAcquisition()
        elif acquisition_function_name == "subsetmax_bax":
            return bax_sampling_toy_experiment.SubsetSelectBaxAcquisition()
        elif acquisition_function_name == "thompson_sampling":
            return thompson_sampling_toy_experiment.ThompsonSamplingcquisition()
        else:
            raise NotImplementedError()

    def initialize_pool(self):
        """Get the data as (x, value) pairs
        Raises:
            NotImplementedError: _description_
        """
        n = NUM_SAMPLES
        n_train = int(0.9 * n)
        available_indices = np.random.choice(n, n_train)
        test_indices = list(filter(lambda i: i not in available_indices, range(n)))
        # Just use all jointly to get a sense of how well able to optimize over whole domain
        available_indices = list(range(n))
        test_indices = list(range(n))
        print(self.dataset_name, "Dataset in init")
        if self.dataset_name == "mog":
            # Generate synthetic dataset
            gen_d = ds.GTDataSynthesizer(mode="mog1d", n_samples=NUM_SAMPLES)
            self.data_synthesizer = gen_d
            gen_d._generate_data()
            dataset_x = gen_d.dataset[0]
            return dataset_x, available_indices, test_indices
        elif self.dataset_name == "sine":
            # Generate synthetic dataset
            gen_d = ds.GTDataSynthesizer(mode="sine1d", n_samples=NUM_SAMPLES)
            self.data_synthesizer = gen_d
            gen_d._generate_data()
            dataset_x = gen_d.dataset[0]
            return dataset_x, available_indices, test_indices
        elif self.dataset_name == "sine2d":
            # Generate synthetic dataset
            gen_d = ds.GTDataSynthesizer(mode="sine2d", n_samples=NUM_SAMPLES)
            self.data_synthesizer = gen_d
            gen_d._generate_data()
            dataset_x = gen_d.dataset[0]
            return dataset_x, available_indices, test_indices
        else:
            print(self.dataset_name)
            return NotImplementedError

    def load_data(self) -> Dict[AnyStr, AbstractDataSource]:
        return {}

    def get_metrics(self, set_name: AnyStr) -> List[AbstractMetric]:
        return [
            sp.metrics.MeanAbsoluteError(),
            sp.metrics.RootMeanSquaredError(),
            sp.metrics.SymmetricMeanAbsolutePercentageError(),
            sp.metrics.SpearmanRho(),
        ]

    def get_model(self):
        return None

    def train_model(self) -> Optional[AbstractBaseModel]:
        single_cycle_application_args = {
            "model_name": self.model_name,
            "seed": self.seed,
            "remote_execution_time_limit_days": self.remote_execution_time_limit_days,
            "remote_execution_mem_limit_in_mb": self.remote_execution_mem_limit_in_mb,
            "remote_execution_virtualenv_path": self.remote_execution_virtualenv_path,
            "remote_execution_num_cpus": self.remote_execution_num_cpus,
            "schedule_on_slurm": self.schedule_children_on_slurm,
        }
        cumulative_indices = []
        dataset_x, available_indices, test_indices = self.initialize_pool()
        last_selected_indices = sorted(
            list(
                np.random.choice(
                    available_indices,
                    size=int(self.acquisition_batch_size),
                    replace=False,
                )
            )
        )
        cumulative_indices += last_selected_indices
        result_records = list()
        for cycle_index in range(self.num_active_learning_cycles):
            current_cycle_directory = os.path.join(
                self.output_directory, f"cycle_{cycle_index}"
            )
            PathTools.mkdir_if_not_exists(current_cycle_directory)

            cumulative_indices_file_path = os.path.join(
                current_cycle_directory, "selected_indices.pickle"
            )
            with open(cumulative_indices_file_path, "wb") as fp:
                pickle.dump(cumulative_indices, fp)
            test_indices_file_path = os.path.join(
                current_cycle_directory, "test_indices.pickle"
            )
            with open(test_indices_file_path, "wb") as fp:
                pickle.dump(test_indices, fp)

            app = SingleCycleVizApplication(
                dataset_name=self.dataset_name,
                feature_set_name=self.feature_set_name,
                cache_directory=self.cache_directory,
                output_directory=current_cycle_directory,
                train_ratio=0.8,
                hyperopt=self.hyperopt_children,
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

            trained_model_path = results.model_path

            # trained_model = app.model.load(trained_model_path)
            trained_model = app.model.load(app.model.get_model_save_file_name())
            if trained_model is None:
                trained_model = app.get_model()
            last_selected_indices = self.acquisition_function(
                dataset_x,
                self.acquisition_batch_size,
                available_indices,
                last_selected_indices,
                trained_model,
            )
            cumulative_indices.extend(last_selected_indices)
            cumulative_indices = sorted(list(set(cumulative_indices)))
            assert len(last_selected_indices) == self.acquisition_batch_size
            if PLOT:
                self.plot_mean_and_sd(trained_model, dataset_x, cycle_index)
            plt.scatter(last_selected_indices, 0)

        results_path = os.path.join(self.output_directory, "results.pickle")
        with open(results_path, "wb") as fp:
            pickle.dump(result_records, fp)
        return None

    def plot_mean_and_sd(self, model, dataset_x, iteration):
        # Plot results

        ns = len(dataset_x)
        plot_x = dataset_x
        plot_mean, plot_std = model.predict(dataset_x, return_std_and_margin=True)
        plot_std = np.diag(plot_std)
        plt.plot(plot_mean)
        plt.fill_between(
            range(len(plot_x)), plot_mean - plot_std, plot_mean + plot_std, alpha=0.1
        )
        print(self.acquisition_function_name, "\n \n", self.save_fig_dir)
        plt.savefig(
            f"{self.save_fig_dir}/mean_unc_{self.acquisition_function_name}_{iteration}.png"
        )

    def evaluate_model(
        self,
        model: AbstractBaseModel,
        dataset_x: AbstractDataSource,
        dataset_y: AbstractDataSource,
        with_print: bool = True,
        set_name: AnyStr = "",
        threshold=None,
    ):
        """
        Evaluates model performance.
        Args:
            model: The model to evaluate.
            dataset: The dataset used for evaluation.
            with_print: Whether or not to print results to stdout.
            set_name: The name of the dataset being evaluated.
            threshold: An evaluation threshold (derived in sample) for discrete classification metrics,
             or none if a threshold should automatically be selected.
        Returns:
            A dictionary with each entry corresponding to an evaluation metric with one or more associated values.
        """
        return NotImplementedError()


def downstream_score(
    x,
    true_y,
    test_indices,
    noise_dist=bax_sampling_toy_experiment.bernoulli_noise_sampler,
    noise_type="multiplicative",
    num_samples=5,
):
    maxes = []
    ys = []
    for i in range(num_samples):
        if noise_type == "multiplicative":
            downstream_y = noise_dist(x, true_y, lengthscale=1.0, seed=i)
        else:
            return NotImplementedError()
        maxes.append(np.max(downstream_y[test_indices]))
        ys.append(downstream_y)
    return np.mean(maxes)


def main():
    print("Starting")
    # toy_loop = sp.instantiate_from_command_line(VizLoop)
    for meth in [
        # "random",
        # "ucb",
        # "topk_bax",
        # "levelset_bax",
        # "subsetmax_bax",
        "thompson_sampling"
    ]:
        num_seeds = 5 if (meth == "random" or meth == "thompson_sampling") else 1
        for s in range(num_seeds):
            for ds in ["sine"]:  # , 'mog']:
                toy_loop = VizLoop(
                    acquisition_function_name=meth,
                    seed=s,
                    dataset_name=ds,
                    output_directory=f"output_{ds}",
                )
                toy_loop.run()
                plt.clf()


if __name__ == "__main__":
    main()
