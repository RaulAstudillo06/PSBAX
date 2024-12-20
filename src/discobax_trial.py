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

from src.acquisition_functions.posterior_sampling import gen_posterior_sampling_batch
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
    # project_path = script_dir[:-11]
    # results_folder = (
    #     project_path + "/experiments/results/" + problem + "/" + policy_id + "/"
    # )
    results_folder = os.path.join(script_dir, "results", problem, policy_id) + "/"

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # save stderr to a file
    try:
        stderr_dir = results_folder + "stderr/"
        os.makedirs(stderr_dir, exist_ok=True)
        sys.stderr = open(stderr_dir + f"stderr_{trial}.txt", "w")
    except:
        print("Not writing to stderr.txt")
        pass

    if update_objective:
        obj_func.initialize(seed=seed, regenerate=True)
        algorithm.set_obj_func(obj_func)
        for metric in performance_metrics:
            metric.set_algo(algorithm)

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
                file_path=results_folder + f"failed/trial{trial}",
                **kwargs,
            )
            t1 = time.time()
            model_training_time = t1 - t0
            
            available_indices = obj_func.get_idx()
            cumulative_indices = obj_func.get_idx_from_x(inputs)
            last_selected_indices = cumulative_indices[-batch_size:]

            # check if inputs @ inputs.T is positive definite
            # try:
            #     torch.linalg.cholesky(inputs @ inputs.T) # (n, n)
            # except:
            #     pass

        except:
            available_indices = obj_func.get_idx()
            cumulative_indices = []
            last_selected_indices = list(np.random.choice(available_indices, num_init_points, replace=allow_reselect))
            cumulative_indices += last_selected_indices

            inputs = obj_func.get_x(last_selected_indices)
            obj_vals = obj_func.get_y_from_x(inputs)
            t0 = time.time()
            model = fit_model(
                inputs,
                obj_vals,
                model_type=model_type,
                file_path=results_folder + f"failed/trial{trial}",
                **kwargs,
            )
            t1 = time.time()
            model_training_time = t1 - t0

            performance_metrics_vals = [
                evaluate_performance(performance_metrics, model, **kwargs)
            ]
            runtimes = []
            iteration = 0
    else:
        available_indices = obj_func.get_idx()
        cumulative_indices = []
        last_selected_indices = list(np.random.choice(available_indices, num_init_points, replace=allow_reselect))
        cumulative_indices += last_selected_indices

        inputs = obj_func.get_x(last_selected_indices)
        obj_vals = obj_func.get_y_from_x(inputs)
        t0 = time.time()
        model = fit_model(
            inputs,
            obj_vals,
            model_type=model_type,
            file_path=results_folder + f"failed/trial{trial}",
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
        assert(len(available_indices) > 0 or not allow_reselect)
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
        elif "ps" in policy:
            last_selected_indices = gen_posterior_sampling_batch(
                model, algorithm, batch_size, eval_all=eval_all, **kwargs
            )

        elif "bax" in policy:
            x_batch = obj_func.get_x()
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
        
        t1 = time.time()
        acquisition_time = t1 - t0
        runtimes.append(acquisition_time + model_training_time)

        # Get obj vals at new batch
        cumulative_indices += last_selected_indices
        new_obj_vals = obj_func(last_selected_indices)
        x_new = obj_func.get_x(last_selected_indices)
        inputs = torch.cat([inputs, x_new])   
        obj_vals = torch.cat([obj_vals, new_obj_vals])

        # Fit GP model
        t0 = time.time()
        model = fit_model(
            inputs,
            obj_vals,
            model_type=model_type,
            # architecture=architecture,
            file_path=results_folder + f"failed/trial{trial}",
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

