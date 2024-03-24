#!/usr/bin/env python3

from typing import Callable, Dict, Optional, List

import numpy as np
import os
import sys
import time
import torch
from botorch.models.model import Model
from torch import Tensor

from src.acquisition_functions.posterior_sampling import gen_posterior_sampling_batch
from src.acquisition_functions.bax_acquisition import BAXAcquisitionFunction
from src.fit_model import fit_model
from src.performance_metrics import evaluate_performance

from src.utils import (
    generate_initial_data,
    generate_random_points,
    get_obj_vals,
    seed_torch,
)


# See experiment_manager.py for parameters
def california_trial(
    problem: str,
    obj_func: Callable,
    algorithm,
    performance_metrics: List,
    input_dim: int,
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
    additional_params: Optional[Dict] = None,
) -> None:
    seed_torch(trial)
    architecture = additional_params.get("architecture", None)


    policy_id = policy + "_" + str(batch_size)  # Append q to policy ID

    # Get script directory
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    project_path = script_dir[:-11]
    results_folder = (
        project_path + "/experiments/results/" + problem + "/" + policy_id + "/"
    )

    # edge_locs = additional_params.get("edge_locs", None)
    edge_positions = additional_params.get("edge_positions", None)
    # edge_locs = list(edge_positions)

    if restart:
        raise NotImplementedError
    else:
        # Initial data
        inputs, obj_vals = generate_initial_data(
            num_init_points=num_init_points,
            input_dim=input_dim,
            obj_func=obj_func,
            noise_type=noise_type,
            noise_level=noise_level,
            seed=trial,
            edge_positions=edge_positions,
        )

        # x_init_args = np.random.choice(range(len(edge_locs)), num_init_points)
        # x_init = [edge_positions[idx] for idx in x_init_args]
        # inputs = torch.tensor(x_init)
        # obj_vals = obj_func(inputs)

        # =============== TODO ===============


        # Fit GP model
        t0 = time.time()
        model = fit_model(
            inputs,
            obj_vals,
            model_type=model_type,
            architecture=architecture,
        )
        t1 = time.time()
        model_training_time = t1 - t0

        # Historical performance metrics
        # performance_metrics_vals = [
        #     compute_performance_metrics(obj_func, model, performance_metrics)
        # ]
        performance_metrics_vals = [
            evaluate_performance(performance_metrics, model)
        ]

        # Historical acquisition runtimes
        runtimes = []

        iteration = 0

    while iteration < num_iter:
        iteration += 1
        print("Problem: " + problem)
        print("Sampling policy: " + policy_id)
        print("Trial: " + str(trial))
        print("Iteration: " + str(iteration))

        # New suggested batch
        t0 = time.time()
        new_batch = get_new_suggested_batch(
            policy=policy,
            model=model,
            algorithm=algorithm,
            batch_size=batch_size,
            input_dim=input_dim,
            policy_params=policy_params,
            model_type=model_type,
        )
        t1 = time.time()
        acquisition_time = t1 - t0
        runtimes.append(acquisition_time + model_training_time)

        # Get obj vals at new batch
        new_obj_vals = get_obj_vals(obj_func, new_batch, noise_type, noise_level)

        # Update training data
        inputs = torch.cat((inputs, new_batch))
        obj_vals = torch.cat([obj_vals, new_obj_vals], 0)

        # Fit GP model
        t0 = time.time()
        model = fit_model(
            inputs,
            obj_vals,
            model_type=model_type,
            architecture=architecture,
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


# Computes new batch to evaluate
def get_new_suggested_batch(
    policy: str,
    model: Model,
    algorithm,
    batch_size,
    input_dim: int,
    model_type: str,
    policy_params: Optional[Dict] = None,
) -> Tensor:
    standard_bounds = torch.tensor(
        [[0.0] * input_dim, [1.0] * input_dim]
    )  # This assumes the input domain has been normalized beforehand
    num_restarts = input_dim * batch_size
    raw_samples = 30 * input_dim * batch_size
    batch_initial_conditions = None

    if "random" in policy:
        return generate_random_points(num_points=1, input_dim=input_dim)
    elif "ps" in policy:
        standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])
        return gen_posterior_sampling_batch(model, algorithm, batch_size)
    elif "bax" in policy:
        x_batch = generate_random_points(
            num_points=500, input_dim=input_dim
        )
        # FIXME 
        acq_func = BAXAcquisitionFunction(model=model, algo=algorithm, )
        acq_func.initialize()
        acq_vals = acq_func(x_batch)
        x_next = x_batch[torch.argsort(acq_vals)[-batch_size:]]

        return x_next

