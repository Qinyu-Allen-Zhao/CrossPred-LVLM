import logging
import time

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import scipy as sp
import xarray as xr
import arviz as az
import matplotlib.pyplot as plt

from utils.metric import rmse
from method.ptf import PTF

# Set up logging.
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class CPTF(PTF):
    """Probabilistic Matrix Factorization model using pymc."""

    def __init__(self, train, user_profiles, item_profiles, dim, alpha=2, std=0.01, bounds=(-10, 10)):
        """Build the Probabilistic Matrix Factorization model using pymc.

        :param np.ndarray train: The training data to use for learning the model.
        :param int dim: Dimensionality of the model; number of latent factors.
        :param int alpha: Fixed precision for the likelihood function.
        :param float std: Amount of noise to use for model initialization.
        :param (tuple of int) bounds: (lower, upper) bound of ratings.
            These bounds will simply be used to cap the estimates produced for R.

        """
        self.dim = dim
        self.alpha = alpha
        self.std = np.sqrt(1.0 / alpha)
        self.bounds = bounds
        self.data = train.copy()
        n, m, k = self.data.shape
        dim_user_feat = user_profiles.shape[1]
        dim_item_feat = item_profiles.shape[1]
        nan_mask = np.isnan(self.data)

        # Specify the model.
        logging.info("building the PMF model")
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
        ) as pmf:
            U = pm.MvNormal(
                "U",
                mu=0,
                tau=np.eye(dim),
                dims=("users", "latent_factors"),
                initval=np.random.standard_normal(size=(n, dim)) * std,
            )
            W_U = pm.MvNormal(
                "W_U",
                mu=0,
                tau=np.eye(dim),
                dims=("dim_user_feat", "latent_factors")
            )
            U_comb = pm.Deterministic("U_comb", U + user_profiles @ W_U)

            V = pm.MvNormal(
                "V",
                mu=0,
                tau=np.eye(dim),
                dims=("movies", "latent_factors"),
                initval=np.random.standard_normal(size=(m, dim)) * std,
            )
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

        logging.info("done building the PMF model")
        self.model = pmf
