import argparse

import joblib
import numpy as np

from method.baseline import UniformRandomBaseline, GlobalMeanBaseline, MeanOfMeansBaseline, UseProfileBaseline
from method.ptf import PTF
from method.bptf import BPTF
from method.ptf_with_profile import CPTF
from method.bptf_with_profile import BCPTF

from method.matrix import MatrixManager
from utils.metric import recomm_eval

PMF_METHODS = {
    "ptf": PTF,
    "bptf": BPTF,
    "cptf": CPTF,
    "bcptf": BCPTF
}

BASELINE_METHODS = {
    "ur": UniformRandomBaseline,
    "gm": GlobalMeanBaseline,
    "mom": MeanOfMeansBaseline
}

def main(args):
    np.random.seed(args.random_seed)

    # Load data
    manager = MatrixManager()
    train, test, mu, sigma = manager.load_data_for_ptf(percent_test=args.percent_test / 100.0,
                                                       subset=eval(args.subset))

    # Fit the baseline models
    baselines = {}
    for name in BASELINE_METHODS:
        method = BASELINE_METHODS[name](train)
        baselines[name] = method.rmse(test)
        recomm_eval(method.predicted, test, mu, sigma, str(method))

    # PMF modeling
    kwargs = {}
    postfix = ""

    if "cptf" in args.method:
        if args.gt_profile:
            model_profiles = manager.get_model_profiles(model_profile_cont=["gt_cluster"])
            dataset_profiles = manager.get_dataset_profiles(dataset_profile_cont=["gt_cluster"])
            postfix = " GT profile"
        else:
            model_profiles = manager.get_model_profiles(model_profile_cont=["num_params_llm", "vision_encoder", "model_family"])
            dataset_profiles = manager.get_dataset_profiles(dataset_profile_cont=[args.dataset_profile])
            postfix = " our profile"

        print(model_profiles.mean(axis=0), dataset_profiles.mean(axis=0))
        kwargs["user_profiles"] = model_profiles
        kwargs["item_profiles"] = dataset_profiles

    model = PMF_METHODS[args.method](train, dim=args.dim, alpha=args.alpha, std=args.std, **kwargs)

    # MCMC sampling
    model.draw_samples(draws=args.draws, tune=500)
    mcmc_pred, results = model.running_rmse(test, train, plot=False)
    _ = recomm_eval(mcmc_pred, test, mu, sigma, args.method + postfix)


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