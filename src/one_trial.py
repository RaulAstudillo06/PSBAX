#!/usr/bin/env python3

from typing import Callable, Dict, Optional

import numpy as np
import os
import sys
import time
import torch
from botorch.acquisition import PosteriorMean
from botorch.models.model import Model
from botorch.sampling import SobolQMCNormalSampler
from torch import Tensor

from src.acquisition_functions.thompson_sampling import gen_thompson_sampling_batch
from src.utils import (
    fit_model,
    generate_initial_data,
    generate_random_queries,
    get_obj_vals,
    optimize_acqf_and_get_suggested_batch,
)

# See experiment_manager.py for parameters
def one_trial(
    problem: str,
    obj_func: Callable,
    algo_exe: Callable,
    input_dim: int,
    noise_type: str,
    noise_level: float,
    policy: str,
    batch_size: int,
    num_init_batches: int,
    num_policy_batches: int,
    trial: int,
    restart: bool,
    model_type: str,
    add_baseline_point: bool,
    ignore_failures: bool,
    policy_params: Optional[Dict] = None,
) -> None:

    policy_id = policy + "_" + str(batch_size)  # Append q to policy ID

    # Get script directory
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    project_path = script_dir[:-11]
    results_folder = (
        project_path + "/experiments/results/" + problem + "/" + policy_id + "/"
    )

    if restart:
        # Check if training data is already available
        try:
            # Current available evaluations
            inputs = np.loadtxt(
                results_folder + "inputs/inputs_" + str(trial) + ".txt"
            )
            inputs = inputs.reshape(
                inputs.shape[0],
                batch_size,
                int(inputs.shape[1] / batch_size),
            )
            inputs = torch.tensor(inputs)
            obj_vals = torch.tensor(
                np.loadtxt(results_folder + "obj_vals/obj_vals_" + str(trial) + ".txt")
            )
            # Historical maximum objective values within inputs
            max_obj_vals_within_inputs = list(
                np.loadtxt(
                    results_folder
                    + "max_obj_vals_within_inputs_"
                    + str(trial)
                    + ".txt"
                )
            )
            # Historical objective values at the maximum of the posterior mean
            obj_vals_at_max_post_mean = list(
                np.loadtxt(
                    results_folder + "obj_vals_at_max_post_mean_" + str(trial) + ".txt"
                )
            )
            # Historical acquisition runtimes
            runtimes = list(
                np.atleast_1d(
                    np.loadtxt(
                        results_folder + "runtimes/runtimes_" + str(trial) + ".txt"
                    )
                )
            )

            # Fit GP model
            t0 = time.time()
            model = fit_model(
                inputs,
                obj_vals,
                model_type=model_type,
                likelihood=noise_type,
            )
            t1 = time.time()
            model_training_time = t1 - t0

            iteration = len(max_obj_vals_within_inputs) - 1
            print("Restarting experiment from available data.")

        except:
            # Initial data
            inputs, obj_vals = generate_initial_data(
                num_batches=num_init_batches,
                batch_size=batch_size,
                input_dim=input_dim,
                obj_func=obj_func,
                noise_type=noise_type,
                noise_level=noise_level,
                add_baseline_point=add_baseline_point,
                seed=trial,
            )

            # Fit GP model
            t0 = time.time()
            model = fit_model(
                inputs,
                obj_vals,
                model_type=model_type,
                likelihood=noise_type,
            )
            t1 = time.time()
            model_training_time = t1 - t0

            # Historical objective values at the maximum of the posterior mean
            obj_val_at_max_post_mean = compute_obj_val_at_max_post_mean(
                obj_func, model, input_dim
            )
            obj_vals_at_max_post_mean = [obj_val_at_max_post_mean]

            # Historical maximum objective values within inputs and runtimes
            max_obj_val_within_inputs = obj_vals.max().item()
            max_obj_vals_within_inputs = [max_obj_val_within_inputs]

            # Historical acquisition runtimes
            runtimes = []

            iteration = 0
    else:
        # Initial data
        inputs, obj_vals = generate_initial_data(
            num_batches=num_init_batches,
            batch_size=batch_size,
            input_dim=input_dim,
            obj_func=obj_func,
            noise_type=noise_type,
            noise_level=noise_level,
            add_baseline_point=add_baseline_point,
            seed=trial,
        )

        # Fit GP model
        t0 = time.time()
        model = fit_model(
            inputs,
            obj_vals,
            model_type=model_type,
            likelihood=noise_type,
        )
        t1 = time.time()
        model_training_time = t1 - t0

        # Historical objective values at the maximum of the posterior mean
        obj_val_at_max_post_mean = compute_obj_val_at_max_post_mean(
            obj_func, model, input_dim
        )
        obj_vals_at_max_post_mean = [obj_val_at_max_post_mean]

        # Historical maximum objective values within inputs and runtimes
        max_obj_val_within_inputs = obj_vals.max().item()
        max_obj_vals_within_inputs = [max_obj_val_within_inputs]

        # Historical acquisition runtimes
        runtimes = []

        iteration = 0

    while iteration < num_policy_batches:
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
            algo_exe=algo_exe,
            batch_size=batch_size,
            input_dim=input_dim,
            policy_params=policy_params,
            noise_level=noise_level,
            model_type=model_type,
        )
        t1 = time.time()
        acquisition_time = t1 - t0
        runtimes.append(acquisition_time + model_training_time)

        # Get obj vals at new batch
        new_obj_vals = get_obj_vals(new_batch, obj_func)

        # Update training data
        inputs = torch.cat((inputs, new_batch))
        obj_vals = torch.cat([obj_vals, new_obj_vals], 0)

        # Fit GP model
        t0 = time.time()
        model = fit_model(
            inputs,
            obj_vals,
            model_type=model_type,
            likelihood=noise_type,
        )
        t1 = time.time()
        model_training_time = t1 - t0

        # Append current objective value at the maximum of the posterior mean
        obj_val_at_max_post_mean = compute_obj_val_at_max_post_mean(
            obj_func, model, input_dim
        )
        obj_vals_at_max_post_mean.append(obj_val_at_max_post_mean)
        print(
            "Objective value at the maximum of the posterior mean: "
            + str(obj_val_at_max_post_mean)
        )

        # Append current max objective val within inputs
        max_obj_val_within_inputs = obj_vals.max().item()
        max_obj_vals_within_inputs.append(max_obj_val_within_inputs)
        print("Max objecive value within inputs: " + str(max_obj_val_within_inputs))

        # Save data
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
            results_folder + "obj_vals_at_max_post_mean_" + str(trial) + ".txt",
            np.atleast_1d(obj_vals_at_max_post_mean),
        )
        np.savetxt(
            results_folder + "max_obj_vals_within_inputs_" + str(trial) + ".txt",
            np.atleast_1d(max_obj_vals_within_inputs),
        )


# Computes new batch to evaluate
def get_new_suggested_batch(
    policy: str,
    model: Model,
    algo_exe:Callable,
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

    if policy == "random":
        return generate_random_queries(
            num_queries=1, batch_size=batch_size, input_dim=input_dim
        )
    elif policy == "ts":
        standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])
        return gen_thompson_sampling_batch(
            model, algo_exe, batch_size, standard_bounds, input_dim, 30 * input_dim
        )

    new_batch = optimize_acqf_and_get_suggested_batch(
        acq_func=acquisition_function,
        bounds=standard_bounds,
        batch_size=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        batch_initial_conditions=batch_initial_conditions,
    )

    new_batch = new_batch.unsqueeze(0)
    return new_batch


# Computes the (true) objective value at the maximizer of the model's posterior mean function
def compute_obj_val_at_max_post_mean(
    obj_func: Callable,
    model: Model,
    input_dim: int,
) -> Tensor:

    standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])
    num_restarts = 6 * input_dim
    raw_samples = 180 * input_dim

    post_mean_func = PosteriorMean(model=model)
    max_post_mean_func = optimize_acqf_and_get_suggested_batch(
        acq_func=post_mean_func,
        bounds=standard_bounds,
        batch_size=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )

    obj_val_at_max_post_mean_func = obj_func(max_post_mean_func).item()
    return obj_val_at_max_post_mean_func
