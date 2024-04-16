#!/usr/bin/env python3

from typing import Callable, Dict, List, Optional

from src.one_trial import one_trial
from .discobax_trial import discobax_trial
# from .california_trial import california_trial


def experiment_manager(
    problem: str,
    algorithm,
    performance_metrics: Dict,
    input_dim: int,
    # noise_type: str,
    # noise_level: float,
    policy: str,
    batch_size: int,
    num_init_points: int,
    num_iter: int,
    first_trial: int,
    last_trial: int,
    restart: bool,
    obj_func = None,
    data_df = None,
    model_type: str = "gp",
    ignore_failures: bool = False,
    policy_params: Optional[Dict] = None,
    save_data: bool = False,
    **kwargs,
) -> None:
    r"""
    Args:
        problem: Problem ID
        obj_func:
        algorithm: Algorithm object
        performance_metrics:
        input_dim: Input dimension
        policy: Acquisition function
        batch_size: Number of points sampled at each iteration
        num_init_points: Number of intial queries (chosen uniformly at random)
        num_iter: Number of queries to be chosen using the acquisition function
        first_trial: First trial to be ran (This function runs all trials between first_trial and last_trial sequentially)
        last_trial: Last trial to be ran
        restart: If true, it will try to restart the experiment from available data
        model_type: Type of model (see utils.py for options)
    """

    additional_params = {}
    for key, value in kwargs.items():
        if key not in experiment_manager.__code__.co_varnames:
            additional_params[key] = value

    for trial in range(first_trial, last_trial + 1):
        if "discobax" in problem:
            discobax_trial(
                problem=problem,
                # df = data_df,
                obj_func=obj_func,
                algorithm=algorithm,
                performance_metrics=performance_metrics,
                # input_dim=input_dim,
                policy=policy,
                policy_params=policy_params,
                batch_size=batch_size,
                num_init_points=num_init_points,
                num_iter=num_iter,
                trial=trial,
                restart=restart,
                model_type=model_type,
                ignore_failures=ignore_failures,
                save_data=save_data,
                **additional_params,
            )
        else:
            one_trial(
                problem=problem,
                obj_func=obj_func,
                algorithm=algorithm,
                performance_metrics=performance_metrics,
                input_dim=input_dim,
                policy=policy,
                policy_params=policy_params,
                batch_size=batch_size,
                num_init_points=num_init_points,
                num_iter=num_iter,
                trial=trial,
                restart=restart,
                model_type=model_type,
                ignore_failures=ignore_failures,
                save_data=save_data,
                **additional_params,
            )
