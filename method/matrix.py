import json
import pandas as pd
import numpy as np

from method.model_profile import ModelProfile
from method.dataset_profile import DatasetProfile
from utils.config import MODEL_LIST, TASK_INFO, ALL_DATASETS


class MatrixManager:
    def __init__(self):
        with open("./data/result_summary.json", 'r') as f:
            result_summary = json.load(f)

        result_summary = pd.DataFrame(result_summary).T
        # result_summary = result_summary[[col for col in result_summary.columns if "SEED_2" in col]]
        self.result_summary = result_summary

    def get_metric_info(self):
        metric_info = ["acc" if "Whole dataset" not in col else "BART" 
                       for col in self.result_summary.columns]
        return metric_info

    # Define a function for splitting train/test data.
    def split_train_test(self, data, percent_test=0.1):
        """Split the data into train/test sets.
        :param int percent_test: Percentage of data to use for testing. Default 10.
        """
        n, m = data.shape  # # users, # movies
        N = n * m  # # cells in matrix

        # Prepare train/test ndarrays.
        train = data.copy()
        test = np.ones(data.shape) * np.nan

        # Draw random sample of training data to use for testing.
        tosample = np.where(~np.isnan(data))
        train, test, test_size = self.mask_randomly(tosample, train, test, mask_ratio=percent_test)
        train_size = np.sum(~np.isnan(data)) - test_size  # and remainder for training

        # Verify everything worked properly
        assert train_size == N - np.isnan(train).sum()
        assert test_size == N - np.isnan(test).sum()

        # Return train set and test set
        return train, test

    def load_data_for_pmf(self, percent_test=0.2):
        # Train-test split
        train, test = self.split_train_test(self.result_summary.values.copy(), percent_test=percent_test)

        # Normalization
        train, test, mu, sigma = self.normalize_data(train, test)

        self.train = train
        self.test = test
        self.mu = mu
        self.sigma = sigma

        return train, test, mu, sigma
    
    def load_data_for_pmf_acc_only(self, percent_test=0.2):
        data = self.result_summary.values.copy()
        metric_info = self.get_metric_info()
        acc_subset = [i for i in range(len(metric_info)) if metric_info[i] == "acc"]
        data = data[:, acc_subset]

        # Train-test split
        train, test = self.split_train_test(data, percent_test=percent_test)

        # Normalization
        train, test, mu, sigma = self.normalize_data(train, test, separate_acc_bart_normalization=False)

        self.train = train
        self.test = test
        self.mu = mu
        self.sigma = sigma

        return train, test, mu, sigma

    def normalize_data(self, train, test, separate_acc_bart_normalization=True):
        mu, sigma = np.nanmean(train, axis=0, keepdims=True), np.nanstd(train, axis=0, keepdims=True)

        # Few samples to estimate
        num_samples = (~np.isnan(train)).sum(axis=0, keepdims=True)
        mu[num_samples < 5] = np.nan
        sigma[num_samples < 5] = np.nan
        
        if separate_acc_bart_normalization:
            metric_info = self.get_metric_info()
            acc_subset = [i for i in range(len(metric_info)) if metric_info[i] == "acc"]
            bart_subset = [i for i in range(len(metric_info)) if metric_info[i] == "BART"]

            # Use global mean and std to fill in missing values
            acc_mean, acc_std = np.nanmean(train[:, acc_subset]), np.nanstd(train[:, acc_subset])
            bart_mean, bart_std = np.nanmean(train[:, bart_subset]), np.nanstd(train[:, bart_subset])
            global_mean = np.array([[acc_mean if metric_info[i] == "acc" else bart_mean for i in range(len(metric_info))]])
            global_std = np.array([[acc_std if metric_info[i] == "acc" else bart_std for i in range(len(metric_info))]])
        else:
            global_mean = np.nanmean(train, axis=(0,1), keepdims=True)
            global_mean = np.tile(global_mean, (1, train.shape[1]))
            global_std = np.nanstd(train, axis=(0,1), keepdims=True)
            global_std = np.tile(global_std, (1, train.shape[1]))
                              
        mu[np.isnan(mu)] = global_mean[np.isnan(mu)]
        sigma[np.isnan(sigma)] = global_std[np.isnan(sigma)]

        assert np.isnan(mu).sum() + np.isnan(sigma).sum() == 0

        train = (train - mu) / sigma
        test = (test - mu) / sigma
        return train, test, mu, sigma
    
    def mask_randomly(self, tosample, train, test, mask_ratio=0.1):
        # Draw random sample of training data to use for testing
        idx_pairs = list(zip(tosample[0], tosample[1]))  # tuples of row/col index pairs

        masked_size = int(len(idx_pairs) * mask_ratio) # normal mask
        indices = np.arange(len(idx_pairs))  # indices of index pairs
        sample = np.random.choice(indices, replace=False, size=masked_size)

        # Transfer random sample from train set to test set.
        for idx in sample:
            idx_pair = idx_pairs[idx]
            test[idx_pair] = train[idx_pair]  # transfer to test set
            train[idx_pair] = np.nan  # remove from train set

        return train, test, masked_size

    def get_model_profiles(self, model_profile_cont):
        model_profile = ModelProfile(model_profile_cont)

        model_profile_mat = []
        for model_info in MODEL_LIST:
            model_key = model_info['store_model_path']
            model_profile_mat.append(model_profile.get_profile(model_key))

        model_profile_mat = np.array(model_profile_mat)
       
        print("Load model profile: ", model_profile_mat.shape)

        self.model_profile_mat = model_profile_mat

        return model_profile_mat
    
    def get_dataset_profiles(self, dataset_profile_cont):
        dataset_profile = DatasetProfile(dataset_profile_cont)

        dataset_profile_mat = []
        for dataset_key in ALL_DATASETS:
            dataset_profile_mat.append(dataset_profile.get_profile(dataset_key))

        dataset_profile_mat = np.array(dataset_profile_mat)
       
        print("Load dataset profile: ", dataset_profile_mat.shape)

        self.dataset_profile_mat = dataset_profile_mat

        return dataset_profile_mat

    def split_train_test_3d(self, data, percent_test=0.1):
        n, m, k = data.shape  # # users, # movies, # scores
        N = n * m * k  # cells in matrix

        # Prepare train/test ndarrays.
        train = data.copy()
        test = np.ones_like(data) * np.nan

        # Draw random sample of training data to use for testing.
        tosample = np.where(~np.isnan(data))
        # Draw random sample of training data to use for testing
        idx_pairs = list(zip(tosample[0], tosample[1], tosample[2]))

        test_size = int(len(idx_pairs) * percent_test) # normal mask
        indices = np.arange(len(idx_pairs))  # indices of index pairs
        sample = np.random.choice(indices, replace=False, size=test_size)

        # Transfer random sample from train set to test set.
        for idx in sample:
            idx_pair = idx_pairs[idx]
            test[idx_pair] = train[idx_pair]  # transfer to test set
            train[idx_pair] = np.nan  # remove from train set

        train_size = np.sum(~np.isnan(data)) - test_size  # and remainder for training

        # Verify everything worked properly
        assert train_size == N - np.isnan(train).sum()
        assert test_size == N - np.isnan(test).sum()

        # Return train set and test set
        return train, test
    
    def load_data_for_ptf(self, percent_test=0.2, subset=None):
        all_result_mat = self.get_all_result_mat(subset)

        # Train-test split
        train, test = self.split_train_test_3d(all_result_mat, percent_test=percent_test)

        # Normalization
        mu, sigma = np.nanmean(train, axis=0, keepdims=True), np.nanstd(train, axis=0, keepdims=True)
        # Few samples to estimate
        num_samples = (~np.isnan(train)).sum(axis=0, keepdims=True)
        mu[num_samples < 5] = np.nan
        sigma[num_samples < 5] = np.nan

        # Use global mean and std to fill in missing values
        global_mean = np.nanmean(train, axis=(0,1), keepdims=True)
        global_mean = np.tile(global_mean, (1, train.shape[1], 1))
        mu[np.isnan(mu)] = global_mean[np.isnan(mu)]

        global_std = np.nanstd(train, axis=(0,1), keepdims=True)
        global_std = np.tile(global_std, (1, train.shape[1], 1))
        sigma[np.isnan(sigma)] = global_std[np.isnan(sigma)]

        train = (train - mu) / sigma
        test = (test - mu) / sigma

        self.train = train
        self.test = test
        self.mu = mu
        self.sigma = sigma

        return train, test, mu, sigma

    def get_all_result_mat(self, subset=None):
        with open("./data/all_result_summary.json", 'r') as f:
            all_result_summary = json.load(f)

        all_result_mat = np.empty((len(MODEL_LIST), len(ALL_DATASETS), 6))
        for i, model_info in enumerate(MODEL_LIST):
            model_name = model_info['store_model_path']

            task_data_all = all_result_summary[model_name]
            for j, dataset in enumerate(ALL_DATASETS):
                perf = task_data_all[dataset]

                all_result_mat[i, j, 0] = perf.get("acc", np.nan)
                all_result_mat[i, j, 1] = perf.get("prec", np.nan)
                all_result_mat[i, j, 2] = perf.get("rec", np.nan)
                all_result_mat[i, j, 3] = perf.get("f1", np.nan)
                all_result_mat[i, j, 4] = perf.get("bart", np.nan)
                all_result_mat[i, j, 5] = perf.get("bert", np.nan)

        if subset is not None:
            all_result_mat = all_result_mat[:, :, subset]
        return all_result_mat
    
    def load_data_for_cold_start(self, percent_test=0.2):
        data = self.result_summary.values.copy()
        metric_info = self.get_metric_info()
        acc_subset = [i for i in range(len(metric_info)) if metric_info[i] == "acc"]
        data = data[:, acc_subset]

        # Train-test split
        new_datasets = np.random.choice(data.shape[1], int(data.shape[1] * percent_test), replace=False)
        new_models = np.random.choice(data.shape[0], int(data.shape[0] * percent_test), replace=False)
        train, test = data.copy(), np.zeros_like(data) * np.nan
        for i in new_models:
            test[i, :] = data[i, :]
            train[i, :] = np.nan
        for j in new_datasets:
            test[:, j] = data[:, j]
            train[:, j] = np.nan

        # Normalization
        train, test, mu, sigma = self.normalize_data(train, test, separate_acc_bart_normalization=False)

        self.train = train
        self.test = test
        self.mu = mu
        self.sigma = sigma

        return train, test, mu, sigma, new_datasets, new_models, acc_subset