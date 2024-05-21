import math
import numpy as np
import torch
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.multi_objective.monte_carlo import MultiObjectiveMCAcquisitionFunction
from botorch.acquisition.analytic import PosteriorMean
from botorch.utils.gp_sampling import get_gp_samples
from botorch.models import ModelListGP
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.gp_regression import SingleTaskGP
from botorch.acquisition.objective import (
    ConstrainedMCObjective,
    IdentityMCObjective,
    MCAcquisitionObjective,
    PosteriorTransform,
    ScalarizedPosteriorTransform
)
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
    MCMultiOutputObjective,
)
from botorch.utils.transforms import (
    concatenate_pending_points,
    match_batch_shape,
    t_batch_mode_transform,
)
from copy import deepcopy

from src.models.deep_kernel_gp import DKGP
from src.utils import get_function_samples


tkwargs = {
    "dtype": torch.float64,
    "device": torch.device("cpu"),
}


class BAXAcquisitionFunction(MultiObjectiveMCAcquisitionFunction):
    '''
    Args:
        - model
        - algo
    '''
    def __init__(
            self, 
            model, 
            algo, 
            **kwargs
        ):
        n_obj = len(model.train_targets)
        super().__init__(
            model=model, 
            # posterior_transform=ScalarizedPosteriorTransform(weights=torch.ones(n_obj, **tkwargs)),
            objective=IdentityMCMultiOutputObjective(),
        )
        self.algorithm = algo

        default_params = {
            "batch_size": 1,
            "num_rff_features": 1000,
            "crop": True,
            "acq_str": "exe",
        }

        n_paths_dict = {
            # "Dijkstras": 30,
            # "EvolutionStrategies": 50,
            # "TopK": 100,
            # "SubsetSelect": 20,
            # "NSGA2": 30, 
            # "LBFGSB": 30,
        }
        for (k, v) in default_params.items():
            if k not in kwargs:
                kwargs[k] = v
        # TODO: where is batch_size supposed to be used?
        
        if self.algorithm.params.name in n_paths_dict:
            self.n_samples = kwargs.get("exe_paths", n_paths_dict[self.algorithm.params.name])
        else:
            self.n_samples = kwargs.get("exe_paths", 30)

        for (k, v) in kwargs.items():
            setattr(self, k, v)
    
    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X):
        '''
        Args:
            X: (N, q, dim)
        Returns:
            torch.tensor: (N, 1)
        '''
        # if len(X.shape) == 2:
        #     X = X.unsqueeze(1)
        
        # N, q, d = X.shape

        posterior = self.model.posterior(X)
        mu, cov = posterior.mean, posterior.covariance_matrix
        # mu: (N, q, num_objectives), cov: (N, num_objectives x q, num_objectives x q)

        mu_list = [] # (n_samples, n)
        cov_list = [] # (n_samples, n)

        # data_x = self.model.train_inputs[0] # TODO: is this before or after normalization?  (N, n_dim)
        # data_y = self.model.train_targets # (N, )

        # TODO: check all dimensions
        for comb_model in self.comb_model_list:
            comb_posterior = comb_model.posterior(X)
            comb_mu, comb_cov = comb_posterior.mvn.mean, comb_posterior.mvn.covariance_matrix
            mu_list.append(comb_mu)
            cov_list.append(comb_cov)
        if self.acq_str == "exe":
            acq_vals = self.acq_exe_normal(cov, cov_list)
        elif self.acq_str == "out": # Not implemented
            acq_vals = self.acq_out_normal(mu, cov, mu_list, cov_list)
        # print(f"acq_vals: {acq_vals}")
        return acq_vals
    
    # @staticmethod
    # def normal_entropy(std):
    #     '''
    #     Args:
    #         std: (N, )
    #         std: (N, num_objectives)
    #     '''
    #     # torch.product(std, dim=-1)
    #     std = std.squeeze()
    #     if len(std.shape) > 1:
    #         # 1/2 log(det(Sigma)) = sum(log(std))
    #         log_std = torch.sum(torch.log(std), dim=-1) # (N, )
    #     else:
    #         # log and sqrt are monotonic
    #         log_std = torch.log(std)
        
    #     return 0.5 * torch.log(torch.tensor(2 * math.pi)) + log_std 
    
    @staticmethod
    def entropy(Sigma):
        '''
        Args:
            Sigma: (N, num_objectives x q, num_objectives x q)
        '''
        # H(x) = 1/2 log(det(Sigma)) 

        
        log_det = torch.logdet(Sigma) # (N, )
        return 0.5 * torch.log(torch.tensor(2 * math.pi)) + 0.5 * log_det


        
    def acq_exe_normal(self, cov, cov_list):
        '''
        Args:
            cov: (N, 1)
            cov_list: (n_samples, N, 1)
        Returns:
            torch.tensor: (N, 1)
        '''
        h_post = self.entropy(cov) # (N, ) or (N, num_outputs)

        h_samp_list = []
        for cov in cov_list: # TODO: change name
            h_samp = self.entropy(cov).squeeze() # (N, )
            h_samp_list.append(h_samp)
        
        h_samp = torch.stack(h_samp_list) # (n_samples, N)
        avg_h_samp = torch.mean(h_samp, dim=-2) # (N, )
        acq = h_post - avg_h_samp # (N, )
        return acq


    def fit_with_old_model(self, data_x, data_y):
        '''
        Args:
            data_x: (N, dim)
            data_y: (N, 1)
        Returns:
            botorch.models.model.Model
        '''
        # model = self.model.clone()
        

        if isinstance(self.model, ModelListGP):
            new_models = []
            for i, m in enumerate(self.model.models):
                m = m.condition_on_observations(
                    X=data_x,
                    Y=data_y[:, i].view(-1, 1),
                )
                new_models.append(m)
            new_model = ModelListGP(*new_models)
            return new_model

        model = self.model.condition_on_observations(
            X=data_x,
            Y=data_y,
        )
        return model

    def fit_with_new_model(self, data_x, data_y):
        new_model_mean_module = self.model.mean_module.clone()
        new_model_covar = self.model.covar_module.clone()
        new_model_likelihood = self.model.likelihood.clone()
        new_model_class = type(self.model)
        new_model = new_model_class(
            data_x, 
            data_y, 
            likelihood=new_model_likelihood,
            mean_module=new_model_mean_module, 
            covar_module=new_model_covar
        )
        new_model.load_state_dict(self.model.state_dict())
        return new_model

        
    def initialize(self, **kwargs):
        batch_size = kwargs.pop("batch_size", 1)
        f_sample_list = []
        for _ in range(self.n_samples):
            obj_func_sample = get_function_samples(self.model)
            f_sample_list.append(obj_func_sample)

        exe_path_list, output_list = self.run_algorithm_on_f_list(f_sample_list)
        self.exe_path_list = exe_path_list
        self.output_list = output_list
        self.comb_model_list = []

        for exe_path in self.exe_path_list:
            if self.algorithm.params.name == "Dijkstras" or self.algorithm.params.name == "TopK":
                # comb_data_x = torch.cat([data_x, torch.tensor(np.array(exe_path.x))], dim=-2)
                # comb_data_y = torch.cat([data_y, torch.tensor(np.array(exe_path.y))]) # (N, )
                new_data_x = torch.tensor(np.array(exe_path.x)) # (N, n_dim)
                new_data_y = torch.tensor(np.array(exe_path.y)) # (N, )
            else:
                # comb_data_x = torch.cat([data_x, torch.tensor(exe_path.x)], dim=-2) # (N, n_dim)
                # comb_data_y = torch.cat([data_y, torch.tensor(exe_path.y)]) # (N, )
                new_data_x = torch.tensor(exe_path.x) # (N, n_dim)
                new_data_y = torch.tensor(exe_path.y) # (N, )
            # comb_model = self.model.clone()
            # fit gp model again with the new data
            comb_model = self.fit_with_old_model(
                new_data_x,
                new_data_y,
            ) # FIXME: check if its doing the right thing
            # get posterior mean and cov
            self.comb_model_list.append(comb_model)
    
    def run_algorithm_on_f_list(
            self, 
            f_sample_list, 
        ):

        # if self.algorithm.params.name == "Dijkstras":
        #     exe_path_list = []
        #     output_list = []
        #     for f_sample in f_sample_list:
        #         algo = self.algorithm.initialize()
        #         exe_path, output = algo.run_algorithm_on_f(f_sample)
        #         if self.crop:
        #             exe_path = algo.get_exe_path_crop()

        #         exe_path_list.append(exe_path)
        #         output_list.append(output)
            
        #     return exe_path_list, output_list

        self.algorithm.initialize()
        if (nocopy := getattr(self.algorithm.params, "no_copy", False)):
            exe_path_list = []
            output_list = []
            for f_sample in f_sample_list:
                algo = self.algorithm.get_copy()
                exe_path, output = algo.run_algorithm_on_f(f_sample)
                # algo_list.append(algo)
                exe_path_list.append(exe_path)
                output_list.append(output)
            return exe_path_list, output_list
        

        algo_list = [self.algorithm.get_copy() for _ in range(self.n_samples)]
        if self.algorithm.params.name == "TopK":
            for algo in algo_list:
                algo.initialize()
            x_list = [algo.get_next_x() for algo in algo_list]
            while any(x is not None for x in x_list):
                fx_list = [f(torch.tensor(x).unsqueeze(0)).item() for f, x in zip(f_sample_list, x_list)]
                x_list_new = []
                for algo, x, fx in zip(algo_list, x_list, fx_list):
                    if x is not None:
                        algo.exe_path.x.append(x)
                        algo.exe_path.y.append(fx)
                        x_next = algo.get_next_x()
                    else:
                        x_next = None
                    x_list_new.append(x_next) # the next_x for every x in x_list
                x_list = x_list_new
            self.algo_list = algo_list
            exe_path_list = [algo.exe_path for algo in algo_list]
            output_list = [algo.get_output() for algo in algo_list] # FIXME
            
        else:
            exe_path_list = []
            output_list = []
            for f_sample, algo in zip(f_sample_list, algo_list):
                # algo = self.algorithm.get_copy()
                exe_path, output = algo.run_algorithm_on_f(f_sample)
                # algo_list.append(algo)
                exe_path_list.append(exe_path)
                output_list.append(output)
            

        if self.crop:
            exe_path_list_crop = []
            for algo in algo_list:
                exe_path_crop = algo.get_exe_path_crop()
                exe_path_list_crop.append(exe_path_crop)
            exe_path_list = exe_path_list_crop
        
        return exe_path_list, output_list
