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

# Set up logging.
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class CPMF:
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
        n, m = self.data.shape
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

            R = pm.Normal(
                "R",
                mu=(U_comb @ V_comb.T)[~nan_mask],
                tau=self.alpha,
                dims="obs_id",
                observed=self.data[~nan_mask],
            )

        logging.info("done building the PMF model")
        self.model = pmf

    # Draw MCMC samples.
    def draw_samples(self, **kwargs):
        kwargs.setdefault("chains", 1)
        with self.model:
            self.trace = pm.sample(**kwargs)

    def predict(self, U, V):
        """Estimate R from the given values of U and V."""
        R = np.dot(U, V.T)

        return np.array(R)
    
    def running_rmse(self, test_data, train_data, plot=True):
        """Calculate RMSE for each step of the trace to monitor convergence."""
        results = {"per-step-train": [], "running-train": [], "per-step-test": [], "running-test": []}
        R = np.zeros(test_data.shape)
        for cnt in self.trace.posterior.draw.values:
            U_comb = self.trace.posterior["U_comb"].sel(chain=0, draw=cnt)
            V_comb = self.trace.posterior["V_comb"].sel(chain=0, draw=cnt)
            sample_R = self.predict(U_comb, V_comb)
            R += sample_R
            running_R = R / (cnt + 1)
            results["per-step-train"].append(rmse(train_data, sample_R))
            results["running-train"].append(rmse(train_data, running_R))
            results["per-step-test"].append(rmse(test_data, sample_R))
            results["running-test"].append(rmse(test_data, running_R))

        results = pd.DataFrame(results)

        if plot:
            results.plot(
                kind="line",
                grid=False,
                figsize=(15, 7),
                title="Per-step and Running RMSE From Posterior Predictive",
            )

        # Return the final predictions, and the RMSE calculations
        return running_R, results

    def __str__(self):
        return self.name