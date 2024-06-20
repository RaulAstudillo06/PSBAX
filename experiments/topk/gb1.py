from __future__ import annotations

import os

import pandas as pd
import torch

from . import DiscreteObjective


Tensor = torch.Tensor  # for convenience of typing annotations
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data')

OPTIMAL_VALUE = 8.761965656


class GB1onehot(DiscreteObjective):
    """GB1 one-hot inputs. Larger values are better.

    Total dataset size: 149361

    Args:
        noise_std: standard deviation of the observation noise
        negate: if True, negate the function
        dtype: dtype for self.X and self._y

    Attributes:
        X: Tensor, shape [149361, 80], type self.dtype, by default on CPU
        _y: Tensor, shape [149361], type self.dtype, by default on CPU
        dtype: torch.dtype
    """
    name = 'gb1onehot'

    # BoTorch API
    dim = 80
    _bounds = [(0, 1)] * 80     # 4 positions, 20 possible amino acids each
    _optimal_value: float
    _optimizers = [(
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # F
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,  # W
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # A
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0   # A
    )]

    def __init__(self, noise_std: float | None = None, negate: bool = False,
                 dtype: torch.dtype = torch.float32):
        self.dtype = dtype

        self.df = pd.read_csv(os.path.join(DATA_DIR, 'GB1_fitness.csv'))
        self.X = torch.load(os.path.join(DATA_DIR, 'GB1_onehot_x_bool.pt')).to(dtype)
        self._y = torch.from_numpy(self.df['fit'].values).to(dtype)

        self._optimal_value = self._y.max().item()
        assert torch.isclose(
            torch.tensor(self._optimal_value).to(dtype),
            torch.tensor(OPTIMAL_VALUE).to(dtype))
        assert (self._y == self._optimal_value).sum() == 1

        super().__init__(noise_std=noise_std, negate=negate)


class GB1tape(DiscreteObjective):
    """GB1 with TAPE embeddings. Larger values are better.

    Total dataset size: 149361

    Args:
        noise_std: standard deviation of the observation noise
        negate: if True, negate the function
        dtype: dtype for self.X and self.y

    Attributes:
        X: Tensor, shape [149361, 2048], type self.dtype, by default on CPU
        y: Tensor, shape [149361], type self.dtype, by default on CPU
        dtype: torch.dtype
    """
    name = 'gb1tape'

    # BoTorch API
    dim = 2048
    _bounds: list[tuple[float, float]]
    _optimal_value: float
    _optimizers: list[tuple[float, ...]]

    def __init__(self, noise_std: float | None = None, negate: bool = False,
                 dtype: torch.dtype = torch.float32):
        self.dtype = dtype

        self.X = torch.load(os.path.join(DATA_DIR, 'GB1_x_filt.pt')).to(dtype)
        self._y = torch.load(os.path.join(DATA_DIR, 'GB1_y_filt.pt')).to(dtype)

        self._bounds = list(zip(
            self.X.min(dim=0)[0].numpy().tolist(),
            self.X.max(dim=0)[0].numpy().tolist()
        ))

        self._optimal_value = self._y.max().item()
        assert torch.isclose(
            torch.tensor(self._optimal_value).to(dtype),
            torch.tensor(OPTIMAL_VALUE).to(dtype))
        assert (self._y == self._optimal_value).sum() == 1

        self._optimizers = [
            tuple(self.X[self._y.argmax()].numpy().tolist())
        ]

        super().__init__(noise_std=noise_std, negate=negate)
