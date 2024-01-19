#!/usr/bin/env python3

from typing import Callable, Dict

import numpy as np
import torch
from botorch.acquisition.analytic import PosteriorMean
from botorch.models.model import Model
from torch import Tensor

from src.utils import optimize_acqf_and_get_suggested_batch
from .bax.util.graph import area_of_polygons

class PerformanceMetric:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, obj_func: Callable, posterior_mean_func: PosteriorMean) -> Tensor:
        raise NotImplementedError

class ObjValAtMaxPostMean(PerformanceMetric):
    def __init__(self, obj_func: Callable):
        super().__init__("obj_val_at_max_post_mean")
        self.obj_func = obj_func

    def __call__(self, posterior_mean_func: PosteriorMean) -> Tensor:
        input_dim = posterior_mean_func.dim
        obj_val_at_max_post_mean_func = self.compute_obj_val_at_max_post_mean(
            self, posterior_mean_func, input_dim
        )
        return obj_val_at_max_post_mean_func
    
    def compute_obj_val_at_max_post_mean(
        self,
        posterior_mean_func: PosteriorMean,
        input_dim: int,
    ) -> Tensor:
        standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])
        num_restarts = 6 * input_dim
        raw_samples = 180 * input_dim

        max_post_mean_func = optimize_acqf_and_get_suggested_batch(
            acq_func=posterior_mean_func,
            bounds=standard_bounds,
            batch_size=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

        obj_val_at_max_post_mean_func = self.obj_func(max_post_mean_func).item()
        return obj_val_at_max_post_mean_func

class JaccardSimilarity(PerformanceMetric):
    def __init__(self, algo, obj_func):
        super().__init__("Jaccard similarity")
        self.algo = algo # TopK({"x_path": x_path, "k": k}, verbose=False)
        # TODO: x_path is another source of randomness...
        self.obj_func = obj_func
        self.output_gt = None

    def __call__(self, posterior_mean_func: PosteriorMean) -> Tensor:
        if self.output_gt is None:
            _, self.output_gt = self.algo.run_algorithm_on_f(self.obj_func)
        _, output_mf = self.algo.run_algorithm_on_f(posterior_mean_func)
        
        return self.output_dist_fn_jaccard(output_mf, self.output_gt)
    
    @staticmethod
    def output_dist_fn_jaccard(a, b):
        """Output dist_fn based on Jaccard similarity."""
        a_x_tup = set([tuple(x) for x in a.x])
        b_x_tup = set([tuple(x) for x in b.x])

        jac_sim = len(a_x_tup & b_x_tup) / len(a_x_tup | b_x_tup)
        dist = 1 - jac_sim
        return dist


class NormDifference(PerformanceMetric):
    def __init__(self, algo, obj_func):
        super().__init__("Norm difference")
        self.algo = algo # TopK({"x_path": x_path, "k": k}, verbose=False)
        # TODO: x_path is another source of randomness...
        self.obj_func = obj_func
        self.output_gt = None

    def __call__(self, posterior_mean_func: PosteriorMean) -> Tensor:
        if self.output_gt is None:
            _, self.output_gt = self.algo.run_algorithm_on_f(self.obj_func)
        _, output_mf = self.algo.run_algorithm_on_f(posterior_mean_func)
        
        return self.output_dist_fn_norm(output_mf, self.output_gt)
    
    @staticmethod
    def output_dist_fn_norm(a, b):
        """Output dist_fn based on concatenated vector norm."""
        a_list = []
        list(map(a_list.extend, a.x))
        a_list.extend(a.y)
        a_arr = np.array(a_list)

        b_list = []
        list(map(b_list.extend, b.x))
        b_list.extend(b.y)
        b_arr = np.array(b_list)

        return np.linalg.norm(a_arr - b_arr)


class ShortestPathCost(PerformanceMetric):
    def __init__(self, algo):
        super().__init__("Shortest path cost")
        self.algo = algo
    
    def __call__(self, posterior_mean_func: PosteriorMean) -> Tensor:
        _, output_mf = self.algo.run_algorithm_on_f(posterior_mean_func) 
        return output_mf[0] # true cost func is defined in the algo
    

class ShortestPathArea(PerformanceMetric):
    def __init__(self, algo, obj_func):
        super().__init__("Shortest path area")
        self.algo = algo
        self.obj_func = obj_func
        self.output_gt = None

    def __call__(self, posterior_mean_func: PosteriorMean) -> Tensor:
        if self.output_gt is None:
            _, self.output_gt = self.algo.run_algorithm_on_f(self.obj_func)
        _, output_mf = self.algo.run_algorithm_on_f(posterior_mean_func)
        
        return area_of_polygons(self.output_gt[1], output_mf[1])
    


def evaluate_performance(metrics, model) -> Tensor:
    '''
    Args:
        metrics: list of PerformanceMetric
        model: GP model
    '''
    posterior_mean_func = PosteriorMean(model)
    performance_metrics = []
    for metric in metrics:
        performance_metrics.append(metric(posterior_mean_func))
    return np.atleast_1d(performance_metrics)


def compute_performance_metrics(
    obj_func: Callable, model: Model, perf_metrics_comp: Dict
) -> Tensor:
    posterior_mean_func = PosteriorMean(model)
    performance_metrics = []
    for perf_metric in perf_metrics_comp.values():
        performance_metrics.append(perf_metric(obj_func, posterior_mean_func))
    return np.atleast_1d(performance_metrics)


# Computes the (true) objective value at the maximizer of the model's posterior mean function
def compute_obj_val_at_max_post_mean(
    obj_func: Callable,
    posterior_mean_func: PosteriorMean,
    input_dim: int,
) -> Tensor:
    standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])
    num_restarts = 6 * input_dim
    raw_samples = 180 * input_dim

    max_post_mean_func = optimize_acqf_and_get_suggested_batch(
        acq_func=posterior_mean_func,
        bounds=standard_bounds,
        batch_size=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )

    obj_val_at_max_post_mean_func = obj_func(max_post_mean_func).item()
    return obj_val_at_max_post_mean_func

