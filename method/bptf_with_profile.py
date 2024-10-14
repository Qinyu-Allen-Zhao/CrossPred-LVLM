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


class BCPTF(PTF):
    """Bayesian Probabilistic Matrix Factorization model using pymc."""

    def __init__(self, train, user_profiles, item_profiles, dim, alpha=2, std=0.01, bounds=(-10, 10)):
        self.dim = dim
        self.alpha = alpha
        self.beta_0 = 1  # scaling factor for lambdas; unclear on its use
        self.std = np.sqrt(1.0 / alpha)
        self.bounds = bounds
        self.data = train.copy()
        n, m, k = self.data.shape
        dim_user_feat = user_profiles.shape[1]
        dim_item_feat = item_profiles.shape[1]
        nan_mask = np.isnan(self.data)

        # Specify the model.
        logging.info('building the BPMF model')
        with pm.Model(
            coords={
                "users": np.arange(n),
                "movies": np.arange(m),
                "dim_user_feat": np.arange(dim_user_feat),
                "dim_item_feat": np.arange(dim_item_feat),
                "dim_output": np.arange(k),
                "latent_factors": np.arange(dim),
                "obs_id": np.arange(self.data[~nan_mask].shape[0]),
            }
        ) as bpmf:
            # Specify user feature matrix
            CHOL_U, _, _ = pm.LKJCholeskyCov(
                "CHOL_U", n=dim, eta=5.0, sd_dist=pm.Exponential.dist(1.0, shape=dim)
            )
            MU_U = pm.MvNormal(
                'MU_U', mu=0, chol=CHOL_U,
                dims="latent_factors")
            U = pm.MvNormal(
                'U', mu=MU_U, chol=CHOL_U,
                dims=("users", "latent_factors"))
            
            W_U = pm.MvNormal(
                "W_U",
                mu=0,
                tau=np.eye(dim),
                dims=("dim_user_feat", "latent_factors")
            )
            U_comb = pm.Deterministic("U_comb", U + user_profiles @ W_U)

            # Specify item feature matrix
            CHOL_V, _, _ = pm.LKJCholeskyCov(
                "CHOL_V", n=dim, eta=5.0, sd_dist=pm.Exponential.dist(1.0, shape=dim)
            )
            MU_V = pm.MvNormal(
                'MU_V', mu=0, chol=CHOL_V,
                dims="latent_factors")
            V = pm.MvNormal(
                'V', mu=MU_V, chol=CHOL_V,
                dims=("movies", "latent_factors"))
            
            W_V = pm.MvNormal(
                "W_V",
                mu=0,
                tau=np.eye(dim),
                dims=("dim_item_feat", "latent_factors")
            )
            V_comb = pm.Deterministic("V_comb", V + item_profiles @ W_V)

            W = pm.MvNormal("W", mu=0, tau=np.eye(k), dims=("dim_output"))
            b = pm.MvNormal("b", mu=0, tau=np.eye(k), dims=("dim_output"))
            R_raw = pm.Deterministic("R_raw", U_comb @ V_comb.T)[:, :, None]

            R = pm.Normal(
                "R",
                mu=(R_raw @ W[None, :] + b[None, :])[~nan_mask],
                tau=self.alpha,
                dims="obs_id",
                observed=self.data[~nan_mask],
            )

        logging.info('done building the BPMF model')
        self.model = bpmf
