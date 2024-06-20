#!/usr/bin/env python3

from typing import Callable, Dict, Optional, List

import numpy as np
import os
import sys
import time
import torch
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement
)
from botorch.models.model import Model
from botorch.optim import optimize_acqf_discrete
from botorch.sampling.normal import SobolQMCNormalSampler
from torch import Tensor
import matplotlib.pyplot as plt


from src.acquisition_functions.posterior_sampling import gen_posterior_sampling_batch
from src.acquisition_functions.bax_acquisition import BAXAcquisitionFunction
from src.acquisition_functions.lse import LSE
from src.fit_model import fit_model
from src.performance_metrics import evaluate_performance

from src.utils import (
    generate_initial_data,
    generate_random_points,
    get_obj_vals,
    seed_torch,
    optimize_acqf_and_get_suggested_batch,
    f1_score,
)

# See experiment_manager.py for parameters
def one_trial(
    problem: str,
    obj_func: Callable,
    algorithm,
    performance_metrics: List,
    input_dim: int,
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
    seed_torch(trial)
    policy_id = policy + "_" + str(batch_size)  # Append q to policy ID
    check_GP_fit = kwargs.get("check_GP_fit", False)

    # Get script directory
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    # project_path = script_dir[:-11]
    # results_folder = (
    #     project_path + "/experiments/results/" + problem + "/" + policy_id + "/"
    # )
    results_folder = os.path.join(script_dir, "results", problem, policy_id) + "/"

    

    if restart:
        # Check if training data is already available
        try:
            # Current available evaluations
            inputs = np.loadtxt(results_folder + "inputs/inputs_" + str(trial) + ".txt")
            # inputs = inputs.reshape(
            #     inputs.shape[0],
            #     batch_size,
            #     int(inputs.shape[1] / batch_size),
            # )
            inputs = inputs.reshape(inputs.shape[0], -1)
            inputs = torch.tensor(inputs)
            obj_vals = torch.tensor(
                np.loadtxt(results_folder + "obj_vals/obj_vals_" + str(trial) + ".txt")
            )
            # # Historical maximum performance metrics
            # performance_metrics = torch.tensor(
            #     np.loadtxt(
            #         results_folder
            #         + "performance_metrics_vals/performance_metrics_vals_"
            #         + str(trial)
            #         + ".txt"
            #     )
            # )
            # Historical acquisition runtimes
            runtimes = list(
                np.atleast_1d(
                    np.loadtxt(
                        results_folder + "runtimes/runtimes_" + str(trial) + ".txt"
                    )
                )
            )
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

            # Fit GP model
            t0 = time.time()
            model = fit_model(
                inputs,
                obj_vals,
                model_type=model_type,
                # architecture=architecture,
                file_path=results_folder + f"failed/trial{trial}",
                **kwargs
            )
            t1 = time.time()
            model_training_time = t1 - t0

            # iteration = len(performance_metrics_vals[:, 0]) - 1
            print("Restarting experiment from available data.")

        except:
            # Initial data
            inputs, obj_vals = generate_initial_data(
                num_init_points=num_init_points,
                input_dim=input_dim,
                obj_func=obj_func,
                seed=trial,
                **kwargs,
            )

            # Fit GP model
            t0 = time.time()
            model = fit_model(
                inputs,
                obj_vals,
                model_type=model_type,
                file_path=results_folder + f"failed/trial{trial}",
                # architecture=architecture,
                **kwargs
            )
            t1 = time.time()
            model_training_time = t1 - t0

            if policy == "lse":
                threshold = kwargs.get("threshold", None)
                acq_func = LSE(threshold)
                x_set = kwargs.get("x_set", None)
                acq_func.initialize(x_set)
                kwargs["lse_acq_func"] = acq_func

            # Historical performance metrics
            # performance_metrics_vals = [
            #     compute_performance_metrics(obj_func, model, performance_metrics)
            # ]
            # kwargs["seed"] = trial - 1 # TODO: change to previous trial
            performance_metrics_vals = [
                evaluate_performance(performance_metrics, model, **kwargs)
            ]
            

            # Historical acquisition runtimes
            runtimes = []

            iteration = 0

            
    else:
        # Initial data
        inputs, obj_vals = generate_initial_data(
            num_init_points=num_init_points,
            input_dim=input_dim,
            obj_func=obj_func,
            seed=trial,
            **kwargs,
        )

        # Fit GP model
        t0 = time.time()
        model = fit_model(
            inputs,
            obj_vals,
            model_type=model_type,
            file_path=results_folder + f"failed/trial{trial}",
            # architecture=architecture,
            **kwargs,
        )
        t1 = time.time()
        model_training_time = t1 - t0

        if policy == "lse":
            threshold = kwargs.get("threshold", None)
            acq_func = LSE(threshold)
            x_set = kwargs.get("x_set", None)
            acq_func.initialize(x_set)
            kwargs["lse_acq_func"] = acq_func
            

        performance_metrics_vals = [
            evaluate_performance(performance_metrics, model, **kwargs)
        ]

        # Historical acquisition runtimes
        runtimes = []

        iteration = 0



    while iteration < num_iter:

        # Checking GP fit MSE
        if check_GP_fit:

            edge_coords = kwargs.get("edge_coords", None)
            x_batch = kwargs.get("x_batch", None)
            if edge_coords is not None:
                x_ = torch.tensor(np.array(edge_coords))
            elif x_batch is not None:
                x_ = x_batch
            else:
                x_ = generate_random_points(num_points=1000, input_dim=input_dim, **kwargs)
            y_ = obj_func(x_)
            post_ = model.posterior(x_)
            mean_ = post_.mean.detach().numpy().flatten()
            # std_ = post_.variance.detach().sqrt().numpy().flatten()

            train_x = model.train_inputs[0]
            train_y_standardized = model.train_targets.numpy().flatten()
            train_y_true = obj_func(train_x)
            mean_y_true = train_y_true.mean().item()
            std_y_true = train_y_true.std().item()
            train_y = train_y_standardized * std_y_true + mean_y_true

            post_train = model.posterior(train_x)
            mean_train = post_train.mean.detach().numpy().flatten()
            mean_train = mean_train * std_y_true + mean_y_true
            # std_train_ = post_train_.variance.detach().sqrt().numpy().flatten

            RSS = np.sum((mean_ - y_.numpy()) ** 2)

            fig, ax = plt.subplots()
            ax.scatter(y_, mean_, color='b', marker='.', s=20, label='All')
            ax.scatter(train_y, mean_train, color='g', marker='+', s=20, label='Train')
            
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
            **kwargs,
        )
        t1 = time.time()
        acquisition_time = t1 - t0
        runtimes.append(acquisition_time + model_training_time)

        # Get obj vals at new batch
        new_obj_vals = get_obj_vals(
            obj_func=obj_func, 
            inputs=new_batch, 
            noise_type=kwargs.get("noise_type", "noiseless"),
            noise_level=kwargs.get("noise_level", 0.0),
        )

        # Update training data
        inputs = torch.cat((inputs, new_batch))
        obj_vals = torch.cat([obj_vals, new_obj_vals], 0)

        # Fit GP model
        t0 = time.time()
        model = fit_model(
            inputs,
            obj_vals,
            model_type=model_type,
            file_path=results_folder + f"failed/trial{trial}",
            # architecture=architecture,
            **kwargs,
        )
        t1 = time.time()
        model_training_time = t1 - t0

        # Append current objective value at the maximum of the posterior mean
        # current_performance_metrics = compute_performance_metrics(
        #     obj_func, model, performance_metrics
        # )
        # TODO: is this necessary?
    
        # kwargs["seed"] = trial
        current_performance_metrics = evaluate_performance(performance_metrics, model, **kwargs)

        
            


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
    **kwargs,
) -> Tensor:
    edge_coords = kwargs.get("edge_coords", None) # California
    x_set = kwargs.get("x_set", None) # Level set
    algo_acq = algorithm.get_copy()
    if algo_acq.params.name == "EvolutionStrategies":
        data_x = model.train_inputs[0]
        data_y = model.train_targets
        if algo_acq.params.opt_mode == "min":
            opt_idx = np.argmin(data_y)
        elif algo_acq.params.opt_mode == "max":
            opt_idx = np.argmax(data_y)
        algo_acq.params.init_x = data_x[opt_idx].tolist()
    
    if algo_acq.params.name == "ScalarizedParetoSolver":
        algo_acq.set_model(model)

    if "random" in policy:
        return generate_random_points(num_points=batch_size, input_dim=input_dim, **kwargs)
    elif "ps" in policy:
        return gen_posterior_sampling_batch(model, algo_acq, batch_size, **kwargs)
    elif "qei" in policy:
        acq_func = qNoisyExpectedImprovement(
            model=model,
            X_baseline=model.train_inputs[0],
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])),
        )
        standard_bounds = torch.tensor(
            [[0.0] * input_dim, [1.0] * input_dim]
        )
        x_next = optimize_acqf_and_get_suggested_batch(
            acq_func=acq_func,
            bounds=standard_bounds,
            batch_size=batch_size,
            num_restarts=5 * input_dim * batch_size,
            raw_samples=100 * input_dim * batch_size,
            batch_limit=5,
            init_batch_limit=100,
        )
        return x_next
    elif "qehvi" in policy:
        mean_at_train_inputs = model.posterior(model.train_inputs[0][0]).mean.detach()
        ref_point = mean_at_train_inputs.min(0).values
        # ref_point = torch.tensor(algo_acq.params.ref_point)
        acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=model.train_inputs[0][0],
            prune_baseline=False,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])),
        )
        standard_bounds = torch.tensor(
            [[0.0] * input_dim, [1.0] * input_dim]
        )  # This assumes the input domain has been normalized beforehand
        x_next = optimize_acqf_and_get_suggested_batch(
            acq_func=acq_func,
            bounds=standard_bounds,
            batch_size=batch_size,
            num_restarts=5 * input_dim * batch_size,
            raw_samples=100 * input_dim * batch_size,
            batch_limit=5,
            init_batch_limit=100,
        )
        return x_next
    elif "bax" in policy:
        acq_func = BAXAcquisitionFunction(
            model=model, 
            algo=algo_acq,
            **kwargs, 
        )
        acq_func.initialize(**kwargs)
        continuos_bax_opt = kwargs.get("continuos_bax_opt", False)
        if continuos_bax_opt:
            standard_bounds = torch.tensor(
                [[0.0] * input_dim, [1.0] * input_dim]
            )  # This assumes the input domain has been normalized beforehand
            x_next = optimize_acqf_and_get_suggested_batch(
                acq_func=acq_func,
                bounds=standard_bounds,
                batch_size=1,
                num_restarts=5 * input_dim,
                raw_samples=100 * input_dim,
                batch_limit=5,
                init_batch_limit=100,
            )
        else:
            if edge_coords is not None:
                x_batch = torch.from_numpy(edge_coords) # In BAX, they query all the edge locs as well.
            elif x_set is not None:
                x_batch = torch.from_numpy(x_set)
            elif algo_acq.params.name == "TopK":
                x_batch = np.array(algo_acq.params.x_path)
                x_batch = torch.from_numpy(x_batch)
            elif algo_acq.params.name == "TopKTorch":
                x_batch = algo_acq.params.x_path
            else:
                num_points=kwargs.get("bax_num_cand", 500)
                x_batch = generate_random_points(
                    num_points=num_points, 
                    input_dim=input_dim,
                    **kwargs,
                ) # (N, d)
            x_next, _ = optimize_acqf_discrete(acq_function=acq_func, q=batch_size, choices=x_batch, max_batch_size=100)
        return x_next
    elif "lse" in policy:
        acq_func = kwargs.get("lse_acq_func", None)

        x_next = torch.from_numpy(acq_func.get_next_x(model).reshape(1, -1))

        return x_next

