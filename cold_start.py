import argparse

import joblib
import numpy as np

from method.pmf import PMF
from method.pmf_with_profile import CPMF

from method.matrix import MatrixManager
from utils.metric import recomm_eval

PMF_METHODS = {
    "pmf": PMF,
    "cpmf": CPMF,
}

def main(args):
    np.random.seed(args.random_seed)

    # Load data
    manager = MatrixManager()
    train, test, mu, sigma, new_datasets, new_models, acc_subset = manager.load_data_for_cold_start(percent_test=args.percent_test / 100.0)

    # PMF modeling
    kwargs = {}
    postfix = ""

    if "cpmf" in args.method:
        if args.gt_profile:
            model_profiles = manager.get_model_profiles(model_profile_cont=["gt_cluster"])
            dataset_profiles = manager.get_dataset_profiles(dataset_profile_cont=["gt_cluster"])
            postfix = " GT profile"
        else:
            model_profiles = manager.get_model_profiles(model_profile_cont=["num_params_llm", "vision_encoder", "model_family"])
            dataset_profiles = manager.get_dataset_profiles(dataset_profile_cont=[args.dataset_profile])
            postfix = " our profile"

        dataset_profiles = dataset_profiles[acc_subset, :]
        
        print(model_profiles.mean(axis=0), dataset_profiles.mean(axis=0))
        kwargs["user_profiles"] = model_profiles
        kwargs["item_profiles"] = dataset_profiles
    elif "mean" == args.method:
        pred = np.zeros_like(test)

    if "pmf" in args.method:
        model = PMF_METHODS[args.method](train, dim=args.dim, alpha=args.alpha, std=args.std, **kwargs)
        # MCMC sampling
        model.draw_samples(draws=args.draws, tune=500)
        pred, results = model.running_rmse(test, train, plot=False)

    _ = recomm_eval(pred, test, mu, sigma, args.method)

    # Only new datasets
    test_d = np.zeros_like(test) * np.nan
    test_m = np.zeros_like(test) * np.nan
    test_both = np.zeros_like(test) * np.nan
    for i in range(test.shape[0]):
        for j in range(test.shape[1]):
            if i in new_models and j not in new_datasets:
                test_m[i, j] = test[i, j]
            if i not in new_models and j in new_datasets:
                test_d[i, j] = test[i, j]
            if i in new_models and j in new_datasets:
                test_both[i, j] = test[i, j]
    
    _ = recomm_eval(pred, test_m, mu, sigma, args.method + " New models")
    _ = recomm_eval(pred, test_d, mu, sigma, args.method + " New datasets")
    _ = recomm_eval(pred, test_both, mu, sigma, args.method + " New both")


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
    parser.add_argument("--dataset_profile", type=str, default="hs_cluster", help="Dataset profile to use")
    args = parser.parse_args()

    main(args)