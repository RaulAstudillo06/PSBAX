#!/usr/bin/env python3
from typing import Callable

import os
import sys
import torch
from botorch.settings import debug
from botorch.test_functions.synthetic import Ackley
from torch import Tensor

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
debug._set_state(False)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
print(script_dir[:-12])
sys.path.append(script_dir[:-12])

from src.experiment_manager import experiment_manager


# Objective function
input_dim = 6


def obj_func(X: Tensor) -> Tensor:
    X_unnorm = (4.0 * X) - 2.0
    ackley = Ackley(dim=input_dim)
    objective_X = -ackley.evaluate_true(X_unnorm)
    return objective_X


# Algorithm executable
def algo_exe(func: Callable) -> Tensor:
    solution_set = torch.zeros(11, input_dim)
    return solution_set


# Policies
policy = "ts"

# Run experiment
if len(sys.argv) == 3:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[2])
elif len(sys.argv) == 2:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[1])

experiment_manager(
    problem="ackley",
    obj_func=obj_func,
    algo_exe=algo_exe,
    input_dim=input_dim,
    policy=policy,
    batch_size=2,
    num_init_batches=2 * (input_dim + 1),
    num_algo_batches=100,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=False,
)
