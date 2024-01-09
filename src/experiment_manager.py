#!/usr/bin/env python3

from typing import Callable, Dict, List, Optional

from src.one_trial import one_trial


def experiment_manager(
    problem: str,
    obj_func: Callable,
    algo_exe: Callable,
    performance_metrics: Dict,
    input_dim: int,
    noise_type: str,
    noise_level: float,
    policy: str,
    batch_size: int,
    num_init_points: int,
    num_iter: int,
    first_trial: int,
    last_trial: int,
    restart: bool,
    model_type: str = "single_task_gp",
    ignore_failures: bool = False,
    policy_params: Optional[Dict] = None,
) -> None:
    r"""
    Args:
        problem: Problem ID
        obj_func:
        algo_exe:
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
    for trial in range(first_trial, last_trial + 1):
        one_trial(
            problem=problem,
            obj_func=obj_func,
            algo_exe=algo_exe,
            performance_metrics=performance_metrics,
            input_dim=input_dim,
            noise_type=noise_type,
            noise_level=noise_level,
            policy=policy,
            policy_params=policy_params,
            batch_size=batch_size,
            num_init_points=num_init_points,
            num_iter=num_iter,
            trial=trial,
            restart=restart,
            model_type=model_type,
            ignore_failures=ignore_failures,
        )
