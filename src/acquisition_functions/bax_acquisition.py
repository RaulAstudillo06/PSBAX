import math
import numpy as np
import torch
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.analytic import PosteriorMean
from botorch.utils.gp_sampling import get_gp_samples
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.gp_regression import SingleTaskGP
from copy import deepcopy

from src.models.deep_kernel_gp import DKGP


class BAXAcquisitionFunction(MCAcquisitionFunction):
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
        
        super().__init__(model=model)
        self.algorithm = algo

        default_params = {
            "batch_size": 1,
            "num_rff_features": 500,
            "crop": True,
            "acq_str": "exe",
        }

        n_paths_dict = {
            "Dijkstras": 30,
            "EvolutionStrategies": 50,
            "TopK": 100,
            "SubsetSelect": 20,
        }
        for (k, v) in default_params.items():
            if k not in kwargs:
                kwargs[k] = v
        # TODO: where is batch_size supposed to be used?
        
        self.n_samples = kwargs.get("exe_paths", n_paths_dict[self.algorithm.params.name])

        for (k, v) in kwargs.items():
            setattr(self, k, v)
    
    def forward(self, X):
        '''
        Args:
            X: (N, dim)
        Returns:
            torch.tensor: (N, 1)
        '''
        # if len(X.shape) == 2:
        #     X = X.unsqueeze(0)
        
        N, d = X.shape

        posterior = self.model.posterior(X)
        mu, std = posterior.mvn.mean.detach(), posterior.mvn.stddev.detach()
        # torch.mean(mu): tensor(-1.1588), torch.mean(std): tensor(0.9377)

        mu_list = []
        std_list = []

        data_x = self.model.train_inputs[0] # TODO: is this before or after normalization?  (N, n_dim)
        data_y = self.model.train_targets # (N, )

        # TODO: check all dimensions
        for exe_path in self.exe_path_list:
            if self.algorithm.params.name == "Dijkstras":
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
            # get posterior mean and std
            comb_posterior = comb_model.posterior(X)
            comb_mu, comb_std = comb_posterior.mvn.mean.detach(), comb_posterior.mvn.stddev.detach()
            mu_list.append(comb_mu)
            std_list.append(comb_std)
        if self.acq_str == "exe":
            acq_vals = self.acq_exe_normal(std, std_list)
        elif self.acq_str == "out":
            acq_vals = self.acq_out_normal(mu, std, mu_list, std_list)
        return acq_vals
    
    @staticmethod
    def normal_entropy(std):
        '''
        Args:
            std: (N, )
        '''
        return 0.5 * torch.log(torch.tensor(2 * math.pi)) + torch.log(std) + 0.5
        
    def acq_exe_normal(self, std, std_list):
        '''
        Args:
            std: (N, 1)
            std_list: (n_samples, N, 1)
        Returns:
            torch.tensor: (N, 1)
        '''
        h_post = self.normal_entropy(std) # (N, )

        h_samp_list = []
        for std in std_list:
            h_samp = self.normal_entropy(std).squeeze() # (N, )
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
        # name = self.algorithm.name

        batch_size = kwargs.pop("batch_size", 1)

        f_sample_list = []
        for i in range(self.n_samples):
            # f_sample = get_gp_samples(
            #     model=self.model,
            #     num_outputs=1,  
            #     n_samples=1,
            #     num_rff_features=1000,
            # )
            # f_sample = PosteriorMean(model=f_sample)

            if isinstance(self.model, DKGP):
                aux_model = deepcopy(self.model)
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
            elif isinstance(self.model, SingleTaskGP):
                obj_func_sample = get_gp_samples(
                    model=self.model,
                    num_outputs=1,
                    n_samples=1,
                    num_rff_features=1000,
                )
                obj_func_sample = PosteriorMean(model=obj_func_sample)
            f_sample_list.append(obj_func_sample)

        exe_path_list, output_list = self.run_algorithm_on_f_list(f_sample_list)
        self.exe_path_list = exe_path_list
        self.output_list = output_list
    
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
        algo_list = []
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


            



    

