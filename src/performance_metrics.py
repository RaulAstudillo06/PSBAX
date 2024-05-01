#!/usr/bin/env python3

from typing import Callable, Dict

import numpy as np
import torch
import pandas as pd

from copy import deepcopy
from torch import Tensor
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models import ModelListGP
from botorch.models.model import Model

from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.gp_regression import SingleTaskGP
from botorch.utils.gp_sampling import get_gp_samples
from pymoo.indicators.hv import HV
from botorch.utils.multi_objective import Hypervolume

from src.models.deep_kernel_gp import DKGP
from src.utils import optimize_acqf_and_get_suggested_batch
from .bax.util.graph import area_of_polygons

class PosteriorMeanPerformanceMetric:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, posterior_mean_func: PosteriorMean) -> Tensor:
        raise NotImplementedError

class SamplePerformanceMetric:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, model: Model) -> Tensor:
        raise NotImplementedError

class ObjValAtMaxPostMean(PosteriorMeanPerformanceMetric):
    def __init__(self, obj_func: Callable, input_dim: int):
        super().__init__("obj_val_at_max_post_mean")
        self.obj_func = obj_func
        self.dim = input_dim

    def __call__(self, posterior_mean_func: PosteriorMean) -> Tensor:
        obj_val_at_max_post_mean_func = self.compute_obj_val_at_max_post_mean(
            posterior_mean_func
        )
        return obj_val_at_max_post_mean_func
    
    def compute_obj_val_at_max_post_mean(
        self,
        posterior_mean_func: PosteriorMean,
    ) -> Tensor:
        standard_bounds = torch.tensor([[0.0] * self.dim, [1.0] * self.dim])
        num_restarts = 6 * self.dim
        raw_samples = 180 * self.dim

        max_post_mean_func = optimize_acqf_and_get_suggested_batch(
            acq_func=posterior_mean_func,
            bounds=standard_bounds,
            batch_size=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

        obj_val_at_max_post_mean_func = self.obj_func(max_post_mean_func).item()
        return obj_val_at_max_post_mean_func


class BestValue(PosteriorMeanPerformanceMetric):
    def __init__(self, algo, obj_func, **kwargs):
        super().__init__("Best value")
        self.algo = algo
        self.obj_func = obj_func
        self.optimum = kwargs.pop("optimum", None)
        self.eval_mode = kwargs.pop("eval_mode", "best_value")
        self.opt_mode = algo.params.opt_mode
        self.num_runs = kwargs.pop("num_runs", 1)
        

    def __call__(self, posterior_mean_func: PosteriorMean) -> Tensor:
        # _, output_mf = self.algo.run_algorithm_on_f(posterior_mean_func)
        values = []
        for _ in range(self.num_runs):
            
            output_mf = self.algo.execute(posterior_mean_func)
            f_output = self.obj_func(output_mf)
            if self.opt_mode == "max":
                if self.eval_mode == "regret":
                    # return self.obj_func(self.optimum).item() - torch.max(f_output).item()
                    value = self.obj_func(self.optimum).item() - torch.max(f_output).item()
                elif self.eval_mode == "best_value":
                    # return torch.max(f_output).item()
                    value = torch.max(f_output).item()
            elif self.opt_mode == "min":
                if self.eval_mode == "regret":
                    # return self.obj_func(self.optimum).item() - torch.min(f_output).item()
                    value = self.obj_func(self.optimum).item() - torch.min(f_output).item()
                elif self.eval_mode == "best_value":
                    # return torch.min(f_output).item()
                    value = torch.min(f_output).item()
            values.append(value)
        return np.mean(values)



class JaccardSimilarity(PosteriorMeanPerformanceMetric):
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
        
        return output_dist_fn_jaccard(output_mf, self.output_gt)

class NormDifference(PosteriorMeanPerformanceMetric):
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
        
        return output_dist_fn_norm(output_mf, self.output_gt)


class ShortestPathCost(PosteriorMeanPerformanceMetric):
    def __init__(self, algo):
        super().__init__("Shortest path cost")
        self.algo = algo
    
    def __call__(self, posterior_mean_func: PosteriorMean) -> Tensor:
        _, output_mf = self.algo.run_algorithm_on_f(posterior_mean_func) 
        true_cost = self.algo.true_cost_of_shortest_path.item()
        return true_cost # true cost func is defined in the algo
    

class ShortestPathArea(PosteriorMeanPerformanceMetric):
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
    

class DijkstraBAXMetric(SamplePerformanceMetric):
    def __init__(self, algo, obj_func, **kwargs):
        super().__init__("DijkstraBAXMetric")
        self.algo = algo
        self.obj_func = obj_func
        self.n_samples = kwargs.pop("n_samples", 1)
        self.grid_area = kwargs.pop("total_area", 1)
        self.optimum_cost = kwargs.pop("optimum_cost", 0)
        self.output_gt = None
        self.algo_list = None
        self.output_list = None

    def initialize(self, model):
        data_x = model.train_inputs[0]
        data_y = model.train_targets
        if self.algo.params.opt_mode == "min":
            opt_idx = np.argmin(data_y)
        elif self.algo.params.opt_mode == "max":
            opt_idx = np.argmax(data_y)
        self.algo.params.init_x = data_x[opt_idx].tolist()

        if self.output_gt is None:
            _, self.output_gt = self.algo.run_algorithm_on_f(self.obj_func)

        f_sample_list = []
        for _ in range(self.n_samples):

            if isinstance(model, DKGP):
                aux_model = deepcopy(model)
                inputs = aux_model.train_inputs[0]
                aux_model.train_inputs = (aux_model.embedding(inputs),)
                gp_layer_sample = get_gp_samples(
                    model=aux_model,
                    num_outputs=1,
                    n_samples=1,
                    num_rff_features=1000,
                )
        
                def aux_obj_func_sample_callable(X):
                    return gp_layer_sample.posterior(aux_model.embedding(X)).mean
                
                obj_func_sample = GenericDeterministicModel(f=aux_obj_func_sample_callable)
            elif isinstance(model, SingleTaskGP):
                obj_func_sample = get_gp_samples(
                    model=model,
                    num_outputs=1,
                    n_samples=1,
                    num_rff_features=1000,
                )
                obj_func_sample = PosteriorMean(model=obj_func_sample)
            f_sample_list.append(obj_func_sample)

        algo_list, output_list = self.run_algorithm_on_f_list(f_sample_list)
        self.algo_list = algo_list
        self.output_list = output_list
    
    def run_algorithm_on_f_list(
            self, 
            f_sample_list, 
        ):

        self.algo.initialize()
        algo_list = []
        algo_list = [self.algo.get_copy() for _ in range(self.n_samples)]
        output_list = []
        for f_sample, algo in zip(f_sample_list, algo_list):
            _, output = algo.run_algorithm_on_f(f_sample)
            output_list.append(output)
        
        return algo_list, output_list

    def __call__(self, model) -> Tensor:
        # if self.algo_list is None or self.output_list is None:
        #     self.initialize(model)
        
        assert self.output_list is not None and self.algo_list is not None
        costs = []
        areas = []
        errors = []

        for output, algo in zip(self.output_list, self.algo_list):
            cost = algo.true_cost_of_shortest_path
            area = area_of_polygons(self.output_gt[1], output[1]) / self.grid_area
            error = (cost - self.optimum_cost) * area
            costs.append(cost)
            areas.append(area)
            errors.append(error)
            
        # costs = [algo.true_cost_of_shortest_path.item() for algo in self.algo_list]
        # areas = [area_of_polygons(self.output_gt[1], output[1]) for output in self.output_list]
        
        
        return np.mean(costs), np.mean(areas), np.mean(errors)
        
    

class DiscretePerformanceMetric(PosteriorMeanPerformanceMetric):
    def __init__(self, name, data_df):
        '''
        self.output_gt: list of indices 
        '''
        super().__init__(name)
        self.df = data_df
        
    def __call__(self, posterior_mean_func: PosteriorMean) -> Tensor:
        raise NotImplementedError
    
    def index_to_x(self, idx):
        return self.df.drop(columns=["y"]).loc[idx].values # np.array(len(idx), d)

    def index_to_y(self, idx):
        return self.df.loc[idx, "y"].values # np.array(len(idx),)
    
class DiscreteTopKMetric(DiscretePerformanceMetric):
    def __init__(self, name, algo, data_df):
        super().__init__(name, data_df)
        self.algo = algo.get_copy()
        self.output_gt = None
        self.k = algo.params.k
    
    def __call__(self, posterior_mean_func: PosteriorMean) -> Tensor:
        if self.output_gt is None:
            df_y = self.df["y"]
            self.output_gt = topk_indices(df_y, self.k)
        _, output_mf = self.algo.run_algorithm_on_f(posterior_mean_func)
        # gt_x = [self.index_to_x(idx) for idx in self.output_gt]
        # mf_x = [self.index_to_x(idx) for idx in output_mf]
        if "Norm" in self.name:
            return self.norm_difference(self.output_gt, output_mf)
        elif "Jaccard" in self.name:
            return self.jaccard_similarity(self.output_gt, output_mf)
        else:
            raise NotImplementedError
    
    def jaccard_similarity(self, x1, x2):
        '''
        Args: 
            x1, x2: list of indices
        '''
        x1_set = set(x1)
        x2_set = set(x2)
        return len(x1_set.intersection(x2_set)) / len(x1_set.union(x2_set))
    
    def norm_difference(self, x1, x2):
        '''
        Args: 
            x1, x2: list of indices
        '''
        x1_arr = self.index_to_x(x1)
        x1_y = self.index_to_y(x1)
        x2_arr = self.index_to_x(x2)
        x2_y = self.index_to_y(x2)

        x1_arr = x1_arr.flatten()
        x1_arr = np.append(x1_arr, x1_y)
        x2_arr = x2_arr.flatten()
        x2_arr = np.append(x2_arr, x2_y)

        return np.linalg.norm(x1_arr - x2_arr)
    

class DiscreteDiscoBAXMetric(PosteriorMeanPerformanceMetric):
    def __init__(self, name, obj_func, **kwargs):
        super().__init__(name)
        self.algo = None
        self.obj_func = obj_func
        for (k, v) in kwargs.items():
            setattr(self, k, v)
        self.true_values = None

    # def set_obj_func(self, obj_func):
    #     self.obj_func = obj_func
    
    def set_algo(self, algo):
        self.algo = algo
    
    def __call__(self, posterior_mean_func: PosteriorMean) -> Tensor:
        _, output_mf = self.algo.run_algorithm_on_f(posterior_mean_func)
        return self.evaluate()
    
    def evaluate(self):
        if self.true_values is None:
            x = self.obj_func.get_idx()
            fx = self.obj_func(x).detach().numpy()
            self.true_values = self.obj_func.get_noisy_f_lst(fx) # (n, budget)
        _, idxes = self.algo.get_values_and_selected_indices()
        
        return np.mean(np.max(self.true_values[:, idxes], axis=-1))


    def compute_OPT(self):
        x = self.obj_func.get_idx()
        fx = self.obj_func(x).detach().numpy()
        _, output_mf = self.algo.run_algorithm_on_f(fx)
        # values = obj_func.get_noisy_f_lst(fx) # (n, budget)
        # max_values = np.max(values, axis=1) # (n,)
        return self.evaluate()


class PymooHypervolume(PosteriorMeanPerformanceMetric):
    def __init__(self, algo, obj_func, ref_point, **kwargs):
        super().__init__("Hypervolume")
        self.algo = algo
        self.obj_func = obj_func
        self.ref_point = ref_point
        self.num_runs = kwargs.pop("num_runs", 1)
        self.weight = -1 if self.algo.params.opt_mode == "maximize" else 1
        self.hv = Hypervolume(ref_point=torch.from_numpy(ref_point))
        # self.hv = HV(ref_point=self.weight * self.ref_point)
        self.opt_value = kwargs.pop("opt_value", None)
    
    def __call__(self, posterior_mean_func: PosteriorMean) -> Tensor:
        hvs = []
        # ind = HV(ref_point=self.weight * self.ref_point)
        for _ in range(self.num_runs):
            output_x = self.algo.execute(posterior_mean_func)
            x_torch = torch.tensor(output_x)
            # f_values = self.obj_func(x_torch).detach().numpy()
            # hvs.append(ind(self.weight * f_values))

            f_values = self.obj_func(x_torch).detach()
            hvs.append(self.hv.compute(f_values))
        if self.opt_value is not None:
            return self.weight * (np.mean(hvs) - self.opt_value)
        else:
            return np.mean(hvs)



def evaluate_performance(metrics, model, **kwargs) -> Tensor:
    '''
    Args:
        metrics: list of PerformanceMetric
        model: GP model
    '''

    seed = kwargs.pop("seed", None)
    
    performance_metrics = []
    for metric in metrics:
        if isinstance(metric, SamplePerformanceMetric):
            metric.initialize(model)
            val = metric(model)
        elif isinstance(metric, PosteriorMeanPerformanceMetric):
            # if isinstance(model, SingleTaskGP) or isinstance(model, DKGP):
            #     posterior_mean_func = PosteriorMean(model)
            # else:
            #     # assert(isinstance(model, ModelListGP))
            #     pms = []
            #     for m in model.models:
            #         pms.append(PosteriorMean(m))
            #     def aux_func(x):
            #         # FIXME
            #         return torch.cat([pm(x) for pm in pms], dim=-1)
            #     posterior_mean_func = GenericDeterministicModel(f=aux_func)
            if metric.algo.params.name == "LBFGSB":
                posterior_mean_func = PosteriorMean(model)
            else:
                posterior_mean_func = lambda x : model.posterior(x).mean
            if metric.algo.params.name == "EvolutionStrategies":
                # metric.algo.set_cma_seed(seed)
                data_x = model.train_inputs[0]
                data_y = model.train_targets
                if metric.algo.params.opt_mode == "min":
                    opt_idx = np.argmin(data_y)
                elif metric.algo.params.opt_mode == "max":
                    opt_idx = np.argmax(data_y)
                metric.algo.params.init_x = data_x[opt_idx].tolist()
            val = metric(posterior_mean_func)
        if isinstance(val, tuple):
            performance_metrics.extend(val)
        else:
            performance_metrics.append(val)
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

def output_dist_fn_jaccard(a, b):
    """Output dist_fn based on Jaccard similarity."""
    a_x_tup = set([tuple(x) for x in a.x])
    b_x_tup = set([tuple(x) for x in b.x])

    jac_sim = len(a_x_tup & b_x_tup) / len(a_x_tup | b_x_tup)
    dist = 1 - jac_sim
    return dist

def topk_indices(y: pd.Series, k):
    # check if k is a float
    if isinstance(k, float):
        return list(y.sort_values(ascending=False).index[:int(k * y.shape[0])].values)
    elif isinstance(k, int):
        return list(y.sort_values(ascending=False).index[:k].values)