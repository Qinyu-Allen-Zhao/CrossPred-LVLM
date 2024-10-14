import logging
import time

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import scipy as sp
import xarray as xr
import matplotlib.pyplot as plt

from utils.metric import rmse
from method.ptf import PTF

# Set up logging.
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class BPTF(PTF):
    """Bayesian Probabilistic Matrix Factorization model using pymc."""

    def __init__(self, train, dim, alpha=2, std=0.01, bounds=(-10, 10)):
        self.dim = dim
        self.alpha = alpha
        self.beta_0 = 1  # scaling factor for lambdas; unclear on its use
        self.std = np.sqrt(1.0 / alpha)
        self.bounds = bounds
        self.data = train.copy()
        n, m, k = self.data.shape
        nan_mask = np.isnan(self.data)

        # Specify the model.
        logging.info('building the BPMF model')
        with pm.Model(
            coords={
                "users": np.arange(n),
                "movies": np.arange(m),
                "dim_output": np.arange(k),
                "latent_factors": np.arange(dim),
                "obs_id": np.arange(self.data[~nan_mask].shape[0]),
            }
        ) as bpmf:
            # Specify user feature matrix
            chol_u, _, _ = pm.LKJCholeskyCov(
                "chol_u", n=dim, eta=5.0, sd_dist=pm.Exponential.dist(1.0, shape=dim)
            )
            mu_u = pm.MvNormal(
                'mu_u', mu=0, chol=chol_u,
                dims="latent_factors")
            U = pm.MvNormal(
                'U', mu=mu_u, chol=chol_u,
                dims=("users", "latent_factors"))
            
            # Specify item feature matrix
            chol_v, _, _ = pm.LKJCholeskyCov(
                "chol_v", n=dim, eta=5.0, sd_dist=pm.Exponential.dist(1.0, shape=dim)
            )
            mu_v = pm.MvNormal(
                'mu_v', mu=0, chol=chol_v,
                dims="latent_factors")
            V = pm.MvNormal(
                'V', mu=mu_v, chol=chol_v,
                dims=("movies", "latent_factors"))

            W = pm.MvNormal("W", mu=0, tau=np.eye(k), dims=("dim_output"))
            b = pm.MvNormal("b", mu=0, tau=np.eye(k), dims=("dim_output"))
            R_raw = pm.Deterministic("R_raw", pm.math.dot(U, V.T))[:, :, None]

            R = pm.Normal(
                "R",
                mu=(R_raw @ W[None, :] + b[None, :])[~nan_mask],
                tau=self.alpha,
                dims="obs_id",
                observed=self.data[~nan_mask],
            )

        logging.info('done building the BPMF model')
        self.model = bpmf
