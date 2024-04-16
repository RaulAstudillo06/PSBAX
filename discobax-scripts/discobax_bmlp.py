#%%
import os
import sys
import torch
import pandas as pd
import numpy as np
import argparse

from botorch.settings import debug
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
# print(script_dir[:-12])
# sys.path.append(script_dir[:-12])
sys.path.append('/home/ec2-user/projects/PSBAX/')

from src.bax.alg.discobax import SubsetSelect
from src.acquisition_functions.posterior_sampling import gen_posterior_sampling_batch
from src.acquisition_functions.bax_acquisition import BAXAcquisitionFunction
from src.performance_metrics import DiscreteTopKMetric, DiscreteDiscoBAXMetric
from src.experiment_manager import experiment_manager
from src.problems import DiscoBAXObjective

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
debug._set_state(False)


data_path = "experiments/data/discobax/sanchez_2021_tau_random_1700.csv"

# df = pd.read_csv(data_path, index_col=0)
# df_x = df.drop(columns=["y"])
# df_y = df["y"]

#%%

import slingpy as sp
from slingpy.models import torch_model
from src.discobax.apps import abstract_base_application as aba
from src.discobax.models import toy_experiment_models
from src.discobax.models import meta_models
from src.discobax.models import pytorch_models

from src.discobax.apps.genedisco_single_cycle_experiment import SingleCycleApplication, CustomLoss

# dataset_x = SingleCycleApplication.get_dataset_x(
#     self.feature_set_name, self.cache_directory
# )
# dataset_y = SingleCycleApplication.get_dataset_y(
#     self.dataset_name, self.cache_directory
# )

from genedisco.datasets.features.achilles import Achilles
from slingpy.data_access.data_sources.hdf5_data_source import HDF5DataSource
cache_directory = "../experiments/data/discobax/"
# dataset = Achilles.load_data(cache_directory)

# GeneDiscoLoop.train_model()
# GeneDiscoLoop.initialize_pool()
h5_file = os.path.join(cache_directory, "achilles.h5")
dataset = HDF5DataSource(h5_file, fill_missing_value=0)
dataset_x = sp.CompositeDataSource([dataset])
h5_file = os.path.join(cache_directory, "sanchez_2021_neurons_tau.h5")
dataset_y = HDF5DataSource(h5_file, duplicate_merge_strategy=sp.MeanMergeStrategy())

test_ratio = 0.0
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
            size=int(test_ratio * len(available_indices)),
            replace=False,
        )
    )
)
available_indices = sorted(list(set(available_indices) - set(test_indices)))
dataset_x = dataset_x.subset(available_indices) # pd.DataFrame(dataset_x.get_data()[0]) -> (17176, 808)
dataset_y = dataset_y.subset(available_indices) # pd.DataFrame(dataset_y.get_data()[0]) -> (17176, 1)

# return to GeneDiscoLoop.train_model()
#%%

h5_file = os.path.join(cache_directory, "sanchez_2021_neurons_tau.h5")
dataset_y = HDF5DataSource(h5_file, duplicate_merge_strategy=sp.MeanMergeStrategy())
dataset_y = dataset_y.subset(available_indices)

acquisition_batch_size = 20 

last_selected_indices = sorted(
    list(
        np.random.choice(
            available_indices,
            size=int(acquisition_batch_size),
            replace=False,
        )
    )
)

cumulative_indices = []
cumulative_indices += last_selected_indices # This becomes self.selected_indices
#%%

# Since SingleCycleApplication is also AbstractBaseApplication
# app = SingleCycleApplication()

# In app.__init__, self.model = self.get_model()

super_base_module = torch_model.TorchModel(
    base_module=pytorch_models.BayesianMLP(
        # input_size=SingleCycleApplication.get_dataset_x(
        #     self.feature_set_name, self.cache_directory
        # ).get_shape()[0][-1],
        input_size=808,
        # hidden_size=self.dn_hidden_layer_size,
        hidden_size=8,
    ),
    loss=CustomLoss(),
    batch_size=64,
    num_epochs=100,
)
sp_model = meta_models.PytorchMLPRegressorWithUncertainty(
    model=super_base_module, num_target_samples=100
)
#%%
import pickle as pkl

cumulative_indices_file_path = 'sanchez_2021_tau_achilles_0.1_bayesian_mlp_discobax_20_1000/cycle_0/selected_indices.pkl'
test_indices_file_path = 'sanchez_2021_tau_achilles_0.1_bayesian_mlp_discobax_20_1000/cycle_0/test_indices.pkl'
selected_indices_file_path = cumulative_indices_file_path

with open(test_indices_file_path, "rb") as fp:
    test_indices = pkl.load(fp)
with open(selected_indices_file_path, "rb") as fp:
    selected_indices = pkl.load(fp)

# enter load_data in SingleCyleApplication.run_single()\
# datasets = self.load_data()

h5_file = os.path.join(cache_directory, "achilles.h5")
dataset = HDF5DataSource(h5_file, fill_missing_value=0)
dataset_x = sp.CompositeDataSource([dataset])
h5_file = os.path.join(cache_directory, "sanchez_2021_neurons_tau.h5")
dataset_y = HDF5DataSource(h5_file, duplicate_merge_strategy=sp.MeanMergeStrategy())

avail_names = sorted(
    list(
        set(dataset_x.get_row_names()).intersection(
            set(dataset_y.get_row_names())
        )
    )
)
dataset_y = dataset_y.subset(avail_names)
dataset_x = dataset_x.subset(avail_names) # To.get_data()

dataset_y = dataset_y.subset(selected_indices)

stratifier = sp.StratifiedSplit()
train_ratio = 1.0
split_index_inner = 0
training_indices, validation_indices = stratifier.split(
    dataset_y,
    test_set_fraction=1 - train_ratio,
    split_index=split_index_inner,
)
if train_ratio == 1.0:
    datasets = {
        "training_set_x": dataset_x.subset(training_indices),
        "training_set_y": dataset_y.subset(training_indices),
        "validation_set_x": None,
        "validation_set_y": None,
        "test_set_x": dataset_x.subset(test_indices),
        "test_set_y": dataset_y.subset(test_indices),
    }

dataset_holder = aba.DatasetHolder()
for name, data_source in datasets.items():
    setattr(dataset_holder, name, data_source)
datasets = dataset_holder
#%%

# model = self.get_model()
# SingleCycleApplication.get_model()
super_base_module = torch_model.TorchModel(
    base_module=pytorch_models.BayesianMLP(
        input_size=808,
        hidden_size=8,
    ),
    loss=CustomLoss(),
    batch_size=64,
    num_epochs=100,
)
sp_model = meta_models.PytorchMLPRegressorWithUncertainty(
    model=super_base_module, num_target_samples=100
)

sp_model.fit(datasets.training_set_x, datasets.training_set_y)
save_path = 'sanchez_2021_tau_achilles_0.1_bayesian_mlp_discobax_20_1000/cycle_0/model.tar.gz'
sp_model.save(save_path)
# finished results = app.run().run_result
#%%
model = sp_model

available_indices = sorted(
    list(set(available_indices) - set(last_selected_indices))
)

# BAXAcquisitionFunction.__call__()
temp_folder_name = 'sanchez_2021_tau_achilles_0.1_bayesian_mlp_discobax_20_1000/tmp'
model.save_folder(temp_folder_name)

num_samples_EIG = 2
for j in range(num_samples_EIG):
        # model.load(temp_folder_name)
        model.load_folder(temp_folder_name)
        # Sample (f_ip)_j values
        f = (
            model.get_model_prediction(dataset_x, return_multiple_preds=False)[0]
            .flatten()
            .detach()
            .numpy()
        )

#%%




