import argparse

import numpy as np

from method.baseline_pmf import UniformRandomBaseline, GlobalMeanBaseline, MeanOfMeansBaseline
from method.pmf import PMF

from method.matrix import MatrixManager
from utils.metric import recomm_eval

BASELINE_METHODS = {
    "ur": UniformRandomBaseline,
    "gm": GlobalMeanBaseline,
    "mom": MeanOfMeansBaseline
}

def main(args):
    np.random.seed(args.random_seed)

    # Load data
    manager = MatrixManager()
    if args.subset == "OneMat":
        train, test, mu, sigma = manager.load_data_for_pmf(percent_test=args.percent_test / 100.0)
        eval_metric = manager.get_metric_info()
        acc_subset = [i for i in range(len(eval_metric)) if eval_metric[i] == "acc"]
        bart_subset = [i for i in range(len(eval_metric)) if eval_metric[i] == "BART"]
    else:
        train, test, mu, sigma = manager.load_data_for_ptf(percent_test=args.percent_test / 100.0,
                                                        subset=eval(args.subset))
        train = train.squeeze()
        test = test.squeeze()
        mu = mu.squeeze()
        sigma = sigma.squeeze()

    # Fit the baseline models
    baselines = {}
    for name in BASELINE_METHODS:
        method = BASELINE_METHODS[name](train)
        baselines[name] = method.rmse(test)
        recomm_eval(method.predicted, test, mu, sigma, str(method))
        if args.subset == "OneMat":
            _ = recomm_eval(method.predicted[:, acc_subset], test[:, acc_subset], mu[:, acc_subset], sigma[:, acc_subset], f"{str(method)} for ACC")
            _ = recomm_eval(method.predicted[:, bart_subset], test[:, bart_subset], mu[:, bart_subset], sigma[:, bart_subset], f"{str(method)} for BART")        
    

    # PMF modeling
    model = PMF(train, dim=args.dim, alpha=args.alpha, std=args.std)

    # MCMC sampling
    model.draw_samples(draws=args.draws, tune=500)
    mcmc_pred, results = model.running_rmse(test, train, plot=False)
    if args.eval_train:
        _ = recomm_eval(mcmc_pred, train, mu, sigma, "PMF on train")

    _ = recomm_eval(mcmc_pred, test, mu, sigma, "PMF")
    if args.subset == "OneMat":
        _ = recomm_eval(mcmc_pred[:, acc_subset], test[:, acc_subset], mu[:, acc_subset], sigma[:, acc_subset], "PMF for ACC")
        _ = recomm_eval(mcmc_pred[:, bart_subset], test[:, bart_subset], mu[:, bart_subset], sigma[:, bart_subset], "PMF for BART")        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=81021, help="Random seed")
    parser.add_argument("--dim", type=int, default=10, help="Number of latent factors")
    parser.add_argument("--alpha", type=float, default=2, help="Precision parameter for the likelihood")
    parser.add_argument("--std", type=float, default=0.05, help="Standard deviation of the prior")
    parser.add_argument("--draws", type=int, default=100, help="Number of MCMC draws")
    parser.add_argument("--subset", type=str, default="None", help="Subset of metrics to use")
    parser.add_argument("--percent_test", type=float, default=20, help="Percentage of data to use for testing")
    parser.add_argument("--eval_train", action="store_true", help="Evaluate on training data")
    args = parser.parse_args()

    main(args)