#!/usr/bin/env python3

from typing import Callable, Dict, Optional, List

import numpy as np
import os
import sys
import time
import torch
from botorch.models.model import Model
from torch import Tensor
import matplotlib.pyplot as plt


import pandas as pd
import pickle as pkl
from collections import defaultdict

from botorch.optim import optimize_acqf_discrete

from src.acquisition_functions.posterior_sampling import gen_posterior_sampling_batch, gen_posterior_sampling_batch_discrete
from src.acquisition_functions.bax_acquisition import BAXAcquisitionFunction
from src.fit_model import fit_model
from src.performance_metrics import evaluate_performance

from src.utils import (
    generate_initial_data,
    generate_random_points,
    get_obj_vals,
    seed_torch,
    rand_argmax,
)




# See experiment_manager.py for parameters
def discobax_trial(
    problem: str,
    obj_func,
    algorithm,
    performance_metrics: List,
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
    seed = kwargs.get("seed", trial)
    eval_all = kwargs.get("eval_all", False)
    check_GP_fit = kwargs.get("check_GP_fit", False)
    update_objective = kwargs.get("update_objective", False)
    allow_reselect = kwargs.get("allow_reselect", True)
    seed_torch(seed)

    policy_id = policy + "_" + str(batch_size)  # Append q to policy ID

    # Get script directory
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    project_path = script_dir[:-11]
    results_folder = (
        project_path + "/experiments/results/" + problem + "/" + policy_id + "/"
    )

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # save stderr to a file
    try:
        sys.stderr = open(results_folder + "stderr.txt", "w")
    except:
        print("Not writing to stderr.txt")
        pass

    if update_objective:
        obj_func.initialize(seed=seed)
        algorithm.set_obj_func(obj_func)
        for metric in performance_metrics:
            metric.set_algo(algorithm)

    if "OPT" in policy:
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        for metric in performance_metrics:
            OPT = metric.compute_OPT()
            fn = results_folder + "performance_metrics_" + str(trial) + ".txt"
            # create an array of OPT with size (iter + 1, 1)
            OPT_arr = np.array([OPT for i in range(num_iter + 1)]).reshape(-1, 1)
            np.savetxt(fn, OPT_arr)
        return 

    x_torch = obj_func.get_x()

    if restart:
        try: 
            inputs = torch.tensor(np.loadtxt(
                results_folder + "inputs/inputs_" + str(trial) + ".txt"
            ))
            obj_vals = torch.tensor(np.loadtxt(
                results_folder + "obj_vals/obj_vals_" + str(trial) + ".txt"
            ))
            runtimes = list(np.atleast_1d(np.loadtxt(
                results_folder + "runtimes/runtimes_" + str(trial) + ".txt"
            )))
            # performance_metrics_vals = list(np.atleast_1d(np.loadtxt(
            #     results_folder + "performance_metrics_" + str(trial) + ".txt"
            # )))
            performance_metrics_arr = np.loadtxt(
                results_folder + "performance_metrics_" + str(trial) + ".txt"
            )
            performance_metrics_vals = []
            if len(performance_metrics_arr.shape) == 1:
                for i in range(performance_metrics_arr.shape[0]):
                    performance_metrics_vals.append(np.array([performance_metrics_arr[i]]))
            else:
                for i in range(performance_metrics_arr.shape[0]):
                    performance_metrics_vals.append(performance_metrics_arr[i])
            iteration = len(runtimes)

            if iteration == num_iter:
                return
            
            t0 = time.time()
            model = fit_model(
                inputs,
                obj_vals,
                model_type=model_type,
                **kwargs,
            )
            t1 = time.time()
            model_training_time = t1 - t0
            
            # available_indices = obj_func.get_idx()
            # cumulative_indices = [] # only used for graphing
            # last_selected_indices = []

            # check if inputs @ inputs.T is positive definite
            try:
                torch.linalg.cholesky(inputs @ inputs.T) # (n, n)
            except:
                pass

        except:
            pass
    else:

        available_indices = obj_func.get_idx()
        # test_indices = sorted(list(np.random.choice(available_indices,size=int(test_ratio * len(available_indices)),replace=False,)))
        # available_indices = sorted(list(set(available_indices) - set(test_indices)))
        # obj_func.update_df(obj_func.df.loc[available_indices])

        cumulative_indices = []
        last_selected_indices = []
        last_selected_indices = list(np.random.choice(available_indices, num_init_points, replace=allow_reselect))
        cumulative_indices += last_selected_indices

        inputs = obj_func.get_x(last_selected_indices)


        # randomly choose initial points from x_torch

        # init_idx = np.random.choice(np.arange(x_torch.shape[0]), num_init_points, replace=False)
        # inputs = x_torch[init_idx]
        # if len(inputs.shape) == 1:
        #     inputs = inputs.unsqueeze(0)
        # obj_vals = obj_func(last_selected_indices)
        obj_vals = obj_func.get_y_from_x(inputs)
        t0 = time.time()
        model = fit_model(
            inputs,
            obj_vals,
            model_type=model_type,
            **kwargs,
        )
        t1 = time.time()
        model_training_time = t1 - t0

        performance_metrics_vals = [
            evaluate_performance(performance_metrics, model, **kwargs)
        ]
        runtimes = []
        iteration = 0

    while iteration < num_iter:
        assert(len(available_indices) > 0 and not allow_reselect)
        iteration += 1
        print("Problem: " + problem)
        print("Sampling policy: " + policy_id)
        print("Trial: " + str(trial))
        print("Iteration: " + str(iteration))

        # Update obj_func, algo
        if not allow_reselect:
            available_indices = sorted(list(set(available_indices) - set(cumulative_indices)))
            obj_func.update(available_indices)
            algorithm.set_obj_func(obj_func)


        # New suggested batch
        t0 = time.time()
        if "random" in policy:
            if eval_all:
                last_selected_indices = list(np.random.choice(obj_func.get_idx(), algorithm.params.k, replace=allow_reselect))
            last_selected_indices = list(np.random.choice(obj_func.get_idx(), batch_size, replace=allow_reselect))
            # x_next = obj_func.get_x(last_selected_indices)
        elif "ps" in policy:
            last_selected_indices = gen_posterior_sampling_batch_discrete(
                model, algorithm, batch_size, eval_all=eval_all,
            ) # a list

        elif "bax" in policy:
            x_batch = obj_func.get_x()
            # x_batch = obj_func.x_to_idx.keys()
            # x_batch = list(x_batch)
            # x_batch = torch.tensor(x_batch)
            acq_func = BAXAcquisitionFunction(
                model=model, 
                algo=algorithm,
                **kwargs, 
            )
            acq_func.initialize()
            x_next, _ = optimize_acqf_discrete(
                acq_function=acq_func, 
                q=batch_size, 
                choices=x_batch, 
                max_batch_size=100, 
            )
            last_selected_indices = obj_func.get_idx_from_x(x_next)
        
        else:
            raise ValueError("Policy not recognized")

        # x_new = x_next
        
        t1 = time.time()
        acquisition_time = t1 - t0
        runtimes.append(acquisition_time + model_training_time)

        # Get obj vals at new batch
        cumulative_indices += last_selected_indices
        new_obj_vals = obj_func(last_selected_indices)
        x_new = obj_func.get_x(last_selected_indices)
        inputs = torch.cat([inputs, x_new])   
        # new_obj_vals = obj_func.get_y_from_x(x_new)
        obj_vals = torch.cat([obj_vals, new_obj_vals])

        # Fit GP model
        t0 = time.time()
        model = fit_model(
            inputs,
            obj_vals,
            model_type=model_type,
            # architecture=architecture,
            **kwargs,
        )
        t1 = time.time()
        model_training_time = t1 - t0

        # Check how good the model's fit is 
        if check_GP_fit:
            x_ = obj_func.get_x()
            y_ = obj_func.get_y()
            post_ = model.posterior(x_)
            mean_ = post_.mean.detach().numpy().flatten()
            std_ = post_.variance.detach().sqrt().numpy().flatten()
            if len(cumulative_indices) > 1:
                sampled_int_idx = obj_func.index_to_int_index(cumulative_indices)
                sampled_mean_ = mean_[sampled_int_idx]
                sampled_y_ = y_[sampled_int_idx]

            # calculate RSS 
            RSS = np.sum((mean_ - y_.numpy()) ** 2)

            # plot scatter of true vs predicted mean
            fig, ax = plt.subplots()
            ax.scatter(y_, mean_, color='b', marker='.', s=20, label='All')
            if len(cumulative_indices) > 1:
                ax.scatter(sampled_y_, sampled_mean_, color='g', marker='.', s=10, label='Sampled')
            # plot y = x line
            ax.plot([min(y_), max(y_)], [min(y_), max(y_)], color='r')
            # ax.set_aspect('equal')
            ax.set_xlabel('True')
            ax.set_ylabel('GP Mean')
            ax.set_title(f'Policy {policy}, Iter {iteration}, RSS: {RSS:.2f}')
            
            ax.legend()

            plots_folder = results_folder + "plots/"
            if not os.path.exists(plots_folder):
                os.makedirs(plots_folder)
            plt.savefig(plots_folder + "trial_" + str(trial) + "_" + str(iteration) + ".png")

        current_performance_metrics = evaluate_performance(performance_metrics, model)

        for i, metric in enumerate(performance_metrics):
            print(metric.name + ": " + str(current_performance_metrics[i]))

        performance_metrics_vals.append(current_performance_metrics)

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
            # 
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


