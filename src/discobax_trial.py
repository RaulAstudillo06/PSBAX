#!/usr/bin/env python3

from typing import Callable, Dict, Optional, List

import numpy as np
import os
import sys
import time
import torch
from botorch.models.model import Model
from torch import Tensor


import pandas as pd
import pickle as pkl
from collections import defaultdict


from src.acquisition_functions.posterior_sampling import gen_posterior_sampling_batch, gen_posterior_sampling_batch_discrete
from src.acquisition_functions.bax_acquisition import BAXAcquisitionFunction
from src.fit_model import fit_model
from src.performance_metrics import evaluate_performance

from src.utils import (
    generate_initial_data,
    generate_random_points,
    get_obj_vals,
)




# See experiment_manager.py for parameters
def discobax_trial(
    problem: str,
    df: pd.DataFrame,
    algorithm,
    performance_metrics: List,
    noise_type: str,
    noise_level: float,
    policy: str,
    batch_size: int,
    num_init_points: int,
    num_iter: int,
    trial: int,
    restart: bool,
    model_type: str,
    ignore_failures: bool,
    policy_params: Optional[Dict] = None,
    save_data: bool = False,
    **kwargs,
) -> None:
    test_ratio = kwargs.get("test_ratio", 0)
    topk_percent = kwargs.get("topk_percent", 0.1)
    seed = kwargs.get("seed", 0)
    df_x = df.drop(columns=["y"])
    df_y = df["y"]

    policy_id = policy + "_" + str(batch_size)  # Append q to policy ID

    # Get script directory
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    project_path = script_dir[:-11]
    results_folder = (
        project_path + "/experiments/results/" + problem + "/" + policy_id + "/"
    )

    if restart:
        pass
    else:
        # NOTE: Reduce dimensionality w/ PCA on whole dataset? How to match PCA values with indices?

        available_indices = list(df.index)
        
        
        test_indices = sorted(list(np.random.choice(available_indices,size=int(test_ratio * len(available_indices)),replace=False,)))
        available_indices = sorted(list(set(available_indices) - set(test_indices)))
        cumulative_indices = []
        last_selected_indices = []

        last_selected_indices = list(np.random.choice(available_indices, num_init_points))
        cumulative_indices += last_selected_indices

        result_records = list()
        cumulative_recall_topk = list()
        cumulative_precision_topk = list()
        cumulative_proportion_top_clusters_recovered = list()

        inputs = torch.tensor(df_x.loc[last_selected_indices].values)
        obj_vals = torch.tensor(df_y.loc[last_selected_indices].values)
        t0 = time.time()
        model = fit_model(
            inputs,
            obj_vals,
            model_type=model_type,
        )
        t1 = time.time()
        model_training_time = t1 - t0

        performance_metrics_vals = [
            evaluate_performance(performance_metrics, model)
        ]
        runtimes = []
        iteration = 0

    while iteration < num_iter:
        iteration += 1
        print("Problem: " + problem)
        print("Sampling policy: " + policy_id)
        print("Trial: " + str(trial))
        print("Iteration: " + str(iteration))

        available_indices = list(set(available_indices) - set(cumulative_indices))

        # New suggested batch
        t0 = time.time()
        if policy == "random":
            last_selected_indices = np.random.choice(available_indices, num_init_points)
        elif policy == "ps":
            last_selected_indices = gen_posterior_sampling_batch_discrete(
                model, algorithm, batch_size
            )
        elif policy == "bax":
            acq_func = BAXAcquisitionFunction(model=model, algo=algorithm, )
            acq_func.initialize()
            x_cand = torch.tensor(df_x.loc[available_indices].values)
            acq_vals = acq_func(x_cand)
            top_acq_vals = torch.argsort(acq_vals)[-batch_size:]
            last_selected_indices = available_indices[top_acq_vals]
        
        t1 = time.time()
        acquisition_time = t1 - t0
        runtimes.append(acquisition_time + model_training_time)

        # Get obj vals at new batch
        cumulative_indices.append(last_selected_indices)
        new_obj_vals = df_y.loc[last_selected_indices]
        x_new = df_x.loc[last_selected_indices].values
        # check if x_new is 1d
        if len(x_new.shape) == 1:
            x_new = torch.tensor(x_new.reshape(1, -1))
        else:
            x_new = torch.tensor(x_new)
        inputs = torch.cat([inputs, x_new])   
        obj_vals = torch.cat([obj_vals, torch.tensor(new_obj_vals).unsqueeze(0)])

        # Fit GP model
        t0 = time.time()
        model = fit_model(
            inputs,
            obj_vals,
            model_type=model_type,
        )
        t1 = time.time()
        model_training_time = t1 - t0

        # Append current objective value at the maximum of the posterior mean
        # current_performance_metrics = compute_performance_metrics(
        #     obj_func, model, performance_metrics
        # )
        # TODO: is this necessary?
        current_performance_metrics = evaluate_performance(performance_metrics, model)

        # for i, performance_metric_id in enumerate(performance_metrics.keys()):
        #     print(performance_metric_id + ": " + str(current_performance_metrics[i]))
        
        for i, metric in enumerate(performance_metrics):
            print(metric.name + ": " + str(current_performance_metrics[i]))

        performance_metrics_vals.append(current_performance_metrics)
        # Save data
        
        if save_data:
            try:
                if not os.path.exists(results_folder):
                    os.makedirs(results_folder)
                if not os.path.exists(results_folder + "inputs/"):
                    os.makedirs(results_folder + "inputs/")
                if not os.path.exists(results_folder + "obj_vals/"):
                    os.makedirs(results_folder + "obj_vals/")
                if not os.path.exists(results_folder + "runtimes/"):
                    os.makedirs(results_folder + "runtimes/")
            except:
                pass

            inputs_reshaped = inputs.numpy().reshape(inputs.shape[0], -1)
            np.savetxt(
                results_folder + "inputs/inputs_" + str(trial) + ".txt", inputs_reshaped
            )
            np.savetxt(
                results_folder + "obj_vals/obj_vals_" + str(trial) + ".txt",
                obj_vals.numpy(),
            )
            np.savetxt(
                results_folder + "runtimes/runtimes_" + str(trial) + ".txt",
                np.atleast_1d(runtimes),
            )
            np.savetxt(
                results_folder + "performance_metrics_" + str(trial) + ".txt",
                np.atleast_1d(performance_metrics_vals),
            )


# def get_dataset(name="sanchez", path_to_data="", processed=False):
#     if name == "sanchez":
#         dataset_x = HDF5DataSource("achilles.h5", fill_missing_value=0)
#         dataset_y = HDF5DataSource("sanchez_2021_neurons_tau.h5", fill_missing_value=0)
#     dataset_x = sp.CompositeDataSource([dataset_x]) # 17654
#     dataset_y = sp.CompositeDataSource([dataset_y]) # 17983
#     return dataset_x, dataset_y
    
# def get_topk_indices(
#         dataset_y: AbstractDataSource, topk_percent: float = 0.1
#     ):
#     y_indices = dataset_y.get_row_names()
#     y_values = pd.DataFrame(
#         dataset_y.get_data()[0].flatten(), index=y_indices, columns=["y_values"]
#     )
#     y_values.sort_values(by=["y_values"], ascending=False, axis=0, inplace=True)
#     num_k = int(topk_percent * len(y_values.y_values))
#     topk_indices = list(y_values.iloc[:num_k].index) # 1717 = 0.1 * 17176
#     return topk_indices

# def find_optimal_number_clusters(dataset_x, min_components=2, max_components=100, num_repeats=10, plot_location="temp.png"):
#     n_components = np.arange(min_components, max_components)
#     models = [GMM(n, covariance_type='full', random_state=0, max_iter=200).fit(dataset_x) for n in n_components for repeat in range(num_repeats)]
#     bic = np.array([m.bic(dataset_x) for m in models]).reshape(len(n_components),num_repeats).mean(axis=1).flatten()
#     aic = np.array([m.aic(dataset_x) for m in models]).reshape(len(n_components),num_repeats).mean(axis=1).flatten()
#     # plt.plot(n_components, bic, label='BIC')
#     # plt.plot(n_components, aic, label='AIC')
#     argmin_AIC = min_components + np.argmin(np.array(aic))
#     # plt.axvline(x = argmin_AIC, color = 'r', label = 'Opt. # clus: {}'.format(argmin_AIC), linestyle='dashed')
#     # plt.legend(loc='best')
#     # plt.xlabel('n_components')
#     # plt.savefig(plot_location)
#     # plt.title('AIC & BIC Vs number of GMM components')
#     return argmin_AIC

# def get_top_target_clusters(
#         dataset_x, 
#         topk_indices, 
#         num_clusters=20, 
#         n_components=20,
#         path_cache_cluster_files_prefix="clusters"
#     ):
#     dataset_x = dataset_x.subset(topk_indices)
#     row_names = dataset_x.get_row_names()
#     x = dataset_x.get_data()[0]
#     dict_index_to_item = {}
#     for index,item in enumerate(row_names):
#         dict_index_to_item[index] = item
#     #Reduce dimensionality w/ PCA before performing clustering (most Genedisco datasets have several hundrerds of input dimensions)
#     x = StandardScaler().fit_transform(x)
#     pca = PCA(n_components=n_components)
#     x = pca.fit_transform(x)

#     #Get optimal number of clusters
#     if num_clusters is None:
#         optimal_num_clusters = find_optimal_number_clusters(x, min_components=2, max_components=int(len(row_names)/4), num_repeats=10, plot_location=plot_location)
#         print("Optimal number of clusters {}".format(optimal_num_clusters))
#     else:
#         optimal_num_clusters = num_clusters
#     #Refit GMM with optimal number of clusters
#     GMM_model = GMM(n_components=optimal_num_clusters, covariance_type='full', max_iter=1000, n_init=20, tol=1e-4).fit(x)
#     labels = GMM_model.predict(x)
#     dict_cluster_id_item_name=defaultdict(list)
#     dict_item_name_cluster_id={}
#     for index in range(len(row_names)):
#         dict_cluster_id_item_name[labels[index]].append(dict_index_to_item[index])
#         dict_item_name_cluster_id[dict_index_to_item[index]] = labels[index]
#     # TODO: add seed to save path
#     with open(path_cache_cluster_files_prefix+f'_topk_{optimal_num_clusters}_clusters_to_items.pkl', "wb") as fp:
#         pkl.dump(dict_cluster_id_item_name, fp)
#     with open(path_cache_cluster_files_prefix+f'_topk_items_to_{optimal_num_clusters}_clusters.pkl', "wb") as fp:
#         pkl.dump(dict_item_name_cluster_id, fp)
#     return x, dict_cluster_id_item_name, dict_item_name_cluster_id


