import argparse

import numpy as np

from method.ptf import PTF
from method.pmf import PMF

from method.matrix import MatrixManager
from utils.metric import recomm_eval


def into_one_mat(train, test, mu, sigma, eval_metric):
    acc_subset = [i for i in range(len(eval_metric)) if eval_metric[i] == "acc"]
    bart_subset = [i for i in range(len(eval_metric)) if eval_metric[i] == "BART"]
    
    assert len(acc_subset) + len(bart_subset) == train.shape[1]

    res_om = []
    for mat in [train, test, mu, sigma]:
        r = np.zeros((mat.shape[0], mat.shape[1])) * np.nan
        r[:, acc_subset] = mat[:, acc_subset, 0]
        r[:, bart_subset] = mat[:, bart_subset, 4]
        res_om.append(r)

    return res_om[0], res_om[1], res_om[2], res_om[3], acc_subset, bart_subset


def main(args):
    np.random.seed(args.random_seed)

    # Load data
    manager = MatrixManager()
    train, test, mu, sigma = manager.load_data_for_ptf(percent_test=args.percent_test / 100.0)
    
    # Just validate
    model = PTF(train, dim=args.dim, alpha=args.alpha, std=args.std)
    model.draw_samples(draws=args.draws, tune=500)
    mcmc_pred, results = model.running_rmse(test, train, plot=False)
    _ = recomm_eval(mcmc_pred, test, mu, sigma, args.method + " PTF")

    all_pred = np.zeros_like(test) * np.nan
    for i in range(train.shape[2]):
        train_sep = train[:, :, i]
        test_sep = test[:, :, i]

        model = PMF(train_sep, dim=args.dim, alpha=args.alpha, std=args.std)
        model.draw_samples(draws=args.draws, tune=500)
        mcmc_pred, results = model.running_rmse(test_sep, train_sep, plot=False)
        _ = recomm_eval(mcmc_pred, test_sep, mu[:, :, i], sigma[:, :, i], args.method + f" {i}")

        all_pred[:, :, i] = mcmc_pred

    _ = recomm_eval(all_pred, test, mu, sigma, args.method + " all")

    eval_metric = manager.get_metric_info()
    train_om, test_om, mu_om, sigma_om, acc_subset, bart_subset = into_one_mat(train, test, mu, sigma, eval_metric)

    model = PMF(train_om, dim=args.dim, alpha=args.alpha, std=args.std)
    model.draw_samples(draws=args.draws, tune=500)
    mcmc_pred, results = model.running_rmse(test_om, train_om, plot=False)
    _ = recomm_eval(mcmc_pred, test_om, mu_om, sigma_om, args.method + f" One Mat")

    

    _ = recomm_eval(mcmc_pred[:, acc_subset], test_om[:, acc_subset], mu_om[:, acc_subset], sigma_om[:, acc_subset], args.method + f" One Mat for ACC")
    _ = recomm_eval(mcmc_pred[:, bart_subset], test_om[:, bart_subset], mu_om[:, bart_subset], sigma_om[:, bart_subset], args.method + f" One Mat for BART")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=81021, help="Random seed")
    parser.add_argument("--method", type=str, default="ptf", help="Method to use (ptf, bptf, cptf, bcptf)")
    parser.add_argument("--gt_profile", action="store_true", help="Use ground truth profile")
    parser.add_argument("--dim", type=int, default=10, help="Number of latent factors")
    parser.add_argument("--alpha", type=float, default=2, help="Precision parameter for the likelihood")
    parser.add_argument("--std", type=float, default=0.05, help="Standard deviation of the prior")
    parser.add_argument("--draws", type=int, default=100, help="Number of MCMC draws")
    parser.add_argument("--subset", type=str, default="None", help="Subset of metrics to use")
    parser.add_argument("--percent_test", type=float, default=20, help="Percentage of data to use for testing")
    args = parser.parse_args()

    main(args)