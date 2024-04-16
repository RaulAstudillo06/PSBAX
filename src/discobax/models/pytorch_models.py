import torch
import slingpy as sp
from typing import List
from torch import nn as nn
from torch.nn import functional as F
# from discobax.models import consistent_mc_dropout

torch.set_default_dtype(torch.float64)
from . import consistent_mc_dropout


class MLP(torch.nn.Module, sp.ArgumentDictionary):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size * 2)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.fc3 = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        x = x.float()
        hidden1 = self.fc1(x)
        relu1 = self.relu(hidden1)
        hidden2 = self.fc2(relu1)
        relu2 = self.relu(hidden2)
        output = self.fc3(relu2)
        return output


class BayesianMLP(consistent_mc_dropout.BayesianModule, sp.ArgumentDictionary):
    def __init__(self, input_size: int = 808, hidden_size: int = 32):
        super(BayesianMLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc1_drop = consistent_mc_dropout.ConsistentMCDropout()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)

    def mc_forward_impl(
        self, x: torch.Tensor, return_embedding=False
    ) -> List[torch.Tensor]:
        x = x.float()
        emb = self.fc1(x)
        x = F.relu(self.fc1_drop(emb))
        x = self.fc2(x)
        if not self.training:
            x = x[:, 0]
        if return_embedding:
            return [x, emb]
        else:
            return [x]
