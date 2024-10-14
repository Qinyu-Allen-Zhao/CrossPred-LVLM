import argparse
import numpy as np

from method.matrix import MatrixManager
from method.pmf import PMF
from utils.metric import rmse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="random seed")
args = parser.parse_args()

np.random.seed(args.seed)

# Load data
manager = MatrixManager()
train, test, mu, sigma = manager.load_data_for_pmf(percent_test=0.8)

print("Train: ", train.shape)
print("Test: ", test.shape)

model = PMF(train, dim=10, alpha=2, std=0.05)
model.draw_samples(draws=100, tune=500)
mcmc_pred, results = model.running_rmse(test, train, plot=False)
original_rmse = rmse(test, mcmc_pred)
print(f"Original RMSE: {original_rmse:.4f}")

dataset_info = []
for i in range(train.shape[1]):
    train_i, test_i = train.copy(), test.copy()

    # Get all results from one benchmark
    for idx in range(train.shape[0]):
        if np.isnan(train_i[idx, i]):
            train_i[idx, i] = test_i[idx, i]
            # test_i[idx, i] = np.nan

    model = PMF(train_i, dim=10, alpha=2, std=0.05)
    model.draw_samples(draws=100, tune=500)
    mcmc_pred, results = model.running_rmse(test_i, train_i, plot=False)
    mcmc_pred[~np.isnan(train_i)] = train_i[~np.isnan(train_i)]  # We know the actual value

    new_rmse = rmse(test_i, mcmc_pred)

    print(f"RMSE: {new_rmse:.4f}")
    print(f"Improvement: {original_rmse - new_rmse:.4f}")
    print(f"Improvement (%): {(original_rmse - new_rmse) / original_rmse * 100.0:.2f}%")

    dataset_info.append(f"{(original_rmse - new_rmse) / original_rmse * 100.0:.2f}")

print(dataset_info)

model_info = []
for i in range(train.shape[0]):
    train_i, test_i = train.copy(), test.copy()

    # Get all results from one model
    for idx in range(train.shape[1]):
        if np.isnan(train_i[i, idx]):
            train_i[i, idx] = test_i[i, idx]

    model = PMF(train_i, dim=10, alpha=2, std=0.05)
    model.draw_samples(draws=100, tune=500)
    mcmc_pred, results = model.running_rmse(test_i, train_i, plot=False)
    mcmc_pred[~np.isnan(train_i)] = train_i[~np.isnan(train_i)]  # We know the actual value

    new_rmse = rmse(test_i, mcmc_pred)

    print(f"RMSE: {new_rmse:.4f}")
    print(f"Improvement: {original_rmse - new_rmse:.4f}")
    print(f"Improvement (%): {(original_rmse - new_rmse) / original_rmse * 100.0:.2f}%")

    model_info.append(f"{(original_rmse - new_rmse) / original_rmse * 100.0:.2f}")

print(model_info)
