import numpy as np
from sklearn.linear_model import LinearRegression

from method.model_profile import ModelProfile
from method.dataset_profile import DatasetProfile
from utils.config import MODEL_LIST, ALL_DATASETS
from utils.metric import rmse


def split_title(title):
    """Change "BaselineMethod" to "Baseline Method"."""
    words = []
    tmp = [title[0]]
    for c in title[1:]:
        if c.isupper():
            words.append("".join(tmp))
            tmp = [c]
        else:
            tmp.append(c)
    words.append("".join(tmp))
    return " ".join(words)


class Baseline:
    """Calculate baseline predictions."""

    def __init__(self, train_data):
        """Simple heuristic-based transductive learning to fill in missing
        values in data matrix."""
        self.predict(train_data.copy())

    def predict(self, train_data):
        raise NotImplementedError("baseline prediction not implemented for base class")

    def rmse(self, test_data):
        """Calculate root mean squared error for predictions on test data."""
        return rmse(test_data, self.predicted)

    def __str__(self):
        return split_title(self.__class__.__name__)


# Implement the 3 baselines.


class UniformRandomBaseline(Baseline):
    """Fill missing values with uniform random values."""

    def predict(self, train_data):
        data = train_data.copy()
        nan_mask = np.isnan(data)
        masked_train = np.ma.masked_array(data, nan_mask)
        pmin, pmax = masked_train.min(), masked_train.max()
        N = nan_mask.sum()
        data[nan_mask] = np.random.uniform(pmin, pmax, N)
        self.predicted = data


class GlobalMeanBaseline(Baseline):
    """Fill in missing values using the global mean."""

    def predict(self, train_data):
        data = train_data.copy()
        nan_mask = np.isnan(data)
        data[nan_mask] = data[~nan_mask].mean()
        self.predicted = data


class MeanOfMeansBaseline(Baseline):
    """Fill in missing values using mean of user/item/global means."""

    def predict(self, train_data):
        data = train_data.copy()
        nan_mask = np.isnan(data)
        masked_train = np.ma.masked_array(data, nan_mask)
        global_mean = masked_train.mean(axis=(0, 1))
        user_means = masked_train.mean(axis=1)
        item_means = masked_train.mean(axis=0)

        self.predicted = data.copy()
        n, m, _ = data.shape
        for i in range(n):
            for j in range(m):
                if np.ma.isMA(item_means[j]):
                    self.predicted[i, j] = (global_mean + user_means[i]) / 2
                else:
                    self.predicted[i, j] = (global_mean + user_means[i] + item_means[j]) / 3


class UseProfileBaseline(Baseline):
    """Use model profile and dataset profile to predict performance linearly."""
    def __init__(self, train, model_profile_cont=['rand'], dataset_profile_cont=['rand']):
        self.predict(train, model_profile_cont, dataset_profile_cont)
    
    def load_profile(self, model_profile_cont=['rand'], dataset_profile_cont=['rand']):
        model_profile = ModelProfile(model_profile_cont)
        dataset_profile = DatasetProfile(dataset_profile_cont)

        model_profile_mat, dataset_profile_mat = [], []
        for model_info in MODEL_LIST:
            model_key = model_info['store_model_path']
            model_profile_mat.append(model_profile.get_profile(model_key))
        for dataset_key in ALL_DATASETS:
            dataset_profile_mat.append(dataset_profile.get_profile(dataset_key))

        model_profile_mat = np.array(model_profile_mat)
        dataset_profile_mat = np.array(dataset_profile_mat)
        
        print("Load model profile: ", model_profile_mat.shape)
        print("Load data profile: ", dataset_profile_mat.shape)

        self.model_profile_mat = model_profile_mat
        self.dataset_profile_mat = dataset_profile_mat

        return model_profile_mat, dataset_profile_mat

    def est_using_profile(self, train):
        tosample = np.where(~np.isnan(train))
        X, y = [], []
        for i, j in zip(tosample[0], tosample[1]):
            X.append(np.concatenate([self.model_profile_mat[i], self.dataset_profile_mat[j]]))
            y.append(train[i, j])
        X, y = np.array(X), np.array(y)

        reg = LinearRegression().fit(X, y)

        X_test = []
        n, m = len(MODEL_LIST), len(ALL_DATASETS)
        for i in range(n):
            for j in range(m):
                X_test.append(np.concatenate([self.model_profile_mat[i], self.dataset_profile_mat[j]]))
        
        X_test = np.array(X_test)
        y_pred = reg.predict(X_test)
        y_pred = y_pred.reshape((n, m))

        self.reg = reg
        self.predicted = y_pred

        return reg, y_pred

    def predict(self, train_data, model_profile_cont, dataset_profile_cont):
        self.load_profile(model_profile_cont, dataset_profile_cont)
        self.est_using_profile(train_data.copy())
