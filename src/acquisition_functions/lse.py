import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)

class LSE():
    def __init__(self, threshold, beta=3, epsilon=0.01, **kwargs):
        self.threshold = threshold
        self.beta = 3
        self.epsilon = 0.01
        
    def get_next_x(self, model):
        # posterior = model.posterior(torch.from_numpy(self.x_set))
        # mean, var = posterior.mean.detach(), posterior.variance.detach()
        x_next = None
        max_acq_val = -np.inf

        idx_to_del = []
        for i, x in enumerate(self.x_set):
            x_post = model.posterior(torch.from_numpy(x.reshape(1, -1)))
            mean = x_post.mean.detach().numpy().squeeze()
            std = x_post.stddev.detach().numpy().squeeze()
            C = self.x_to_C[tuple(x)]
            Q = np.array([
                mean - self.beta * std,
                mean + self.beta * std,
            ])
            # get the intersection of C and Q
            C = np.array([max(C[0], Q[0]), min(C[1], Q[1])])
            self.x_to_C[tuple(x)] = C
            if C[0] + self.epsilon > self.threshold:
                
                self.x_to_C.pop(tuple(x))
                self.H.append(x)
                if len(self.x_to_C) == 0:
                    return None
                idx_to_del.append(i)
                continue

            if C[1] - self.epsilon < self.threshold:
                self.x_to_C.pop(tuple(x))
                self.L.append(x)
                if len(self.x_to_C) == 0:
                    return None
                idx_to_del.append(i)   
                continue
            acq_val = np.min(
                [C[1] - self.threshold, self.threshold - C[0]]
            )
            if acq_val > max_acq_val:
                max_acq_val = acq_val
                x_next = x
        self.x_set = np.delete(self.x_set, idx_to_del, axis=0)

        return x_next
    
    def initialize(self, x_set):
        self.x_set = x_set
        self.x_to_C = {}
        for x in self.x_set:
            self.x_to_C[tuple(x)] = np.array([-np.inf, np.inf])
        
        self.H = []
        self.L = []