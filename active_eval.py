import argparse
import numpy as np

import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from method.matrix import MatrixManager
from method.active_selector import RandomSelector, ActiveSelector, WorstSelector
from method.pmf import PMF
from utils.metric import recomm_eval, rmse

parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="random", help="random or active")
parser.add_argument("--seed", type=int, default=0, help="random seed")
args = parser.parse_args()

np.random.seed(args.seed)

if args.method == "random":
    selector = RandomSelector() # Random baseline
elif args.method == "active":
    selector = ActiveSelector() # Active learning
elif args.method == 'worst':
    selector = WorstSelector() # Worst case scenario

# Load data
manager = MatrixManager()
train, test, mu, sigma = manager.load_data_for_pmf(percent_test=0.8)

num_samples = (~np.isnan(train)).sum() + (~np.isnan(test)).sum()

print("Train: ", train.shape)
print("Test: ", test.shape)
print("Num samples: ", num_samples)
print("Num samples in test: ", (~np.isnan(test)).sum())

model = PMF(train, dim=10, alpha=2, std=0.05)
model.draw_samples(draws=100, tune=500)
mcmc_pred, results = model.running_rmse(test, train, plot=False)
original_rmse, original_mae, original_r2 = recomm_eval(mcmc_pred, test, mu, sigma, "PMF Original")
print(f"Original RMSE: {original_rmse:.4f}")
print(f"Original MAE: {original_mae:.4f}")
print(f"Original R2: {original_r2:.4f}")

for select_ratio in [0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]:
    # Select what to evaluate next
    selected = selector.select(model, train, test, mu, sigma, num_samples, select_ratio)
    print("Selected: ", len(selected))

    # Transfer random sample from test set to train set.
    for idx in selected:
        train[idx] = test[idx]  # transfer to train set

    model = PMF(train, dim=10, alpha=2, std=0.05)
    model.draw_samples(draws=100, tune=500)
    mcmc_pred, results = model.running_rmse(test, train, plot=False)
    mcmc_pred[~np.isnan(train)] = train[~np.isnan(train)]  # We know the actual value

    new_rmse, new_mae, new_r2 = recomm_eval(mcmc_pred, test, mu, sigma, "PMF Update")

    print(f"RMSE: {new_rmse:.4f}")
    print(f"Improvement: {original_rmse - new_rmse:.4f}")
    print(f"Improvement (%): {(original_rmse - new_rmse) / original_rmse * 100:.2f}%")

    print(f"MAE: {new_mae:.4f}")
    print(f"Improvement: {original_mae - new_mae:.4f}")
    print(f"Improvement (%): {(original_mae - new_mae) / original_mae * 100:.2f}%")

    print(f"R2: {new_r2:.4f}")
    print(f"Improvement: {new_r2 - original_r2:.4f}")
    print(f"Improvement (%): {(new_r2 - original_r2) / original_r2 * 100:.2f}%")
