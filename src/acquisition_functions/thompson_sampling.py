#!/usr/bin/env python3

from __future__ import annotations

import torch
from botorch.acquisition.analytic import PosteriorMean
from botorch.utils.gp_sampling import get_gp_samples
from copy import copy

from src.utils import optimize_acqf_and_get_suggested_query


def gen_thompson_sampling_batch(model, batch_size, bounds, num_restarts, raw_samples):
    query = []
    for _ in range(batch_size):
        model_rff_sample = get_gp_samples(
            model=model,
            num_outputs=1,
            n_samples=1,
            num_rff_features=1000,
        )
        acquisition_function = PosteriorMean(
            model=model_rff_sample
        )  # Approximate sample from the GP posterior
        new_x = optimize_acqf_and_get_suggested_query(
            acq_func=acquisition_function,
            bounds=bounds,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            batch_size=1,  # Batching is not supported by RFFs-based sample constructor
            batch_limit=1,
            init_batch_limit=1,
        )
        query.append(new_x.clone())

    query = torch.cat(query, dim=-2)
    query = query.unsqueeze(0)
    return query
