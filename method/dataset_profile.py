import os
import re
import json

# import torch
import joblib
import pandas as pd
import numpy as np
from glob import glob

from utils.config import ALL_DATASETS


class DatasetProfile:  
    def __init__(self, profile_content=['hs_cluster']):
        with open("data/dataset_profiles.json") as f:
            self.cluster_info = json.load(f)
        self.rep_of_datasets = self.load_profiles(profile_content)
            
    def get_profile(self, dataset_key):
        profile_vector = []
        for _, value in self.rep_of_datasets[dataset_key].items():
            profile_vector.append(value)
        return np.concatenate(profile_vector)
    
    def load_profiles(self, profile_content):
        rep_of_datasets = {}

        for dataset_key in ALL_DATASETS:
            profile = {}
            for method in profile_content:
                if method == "random":
                    profile[method] = self.get_random_rep()
                else:
                    profile[method] = self.cluster_info[dataset_key][method]

            rep_of_datasets[dataset_key] = profile

        return rep_of_datasets

    def get_random_rep(self):
        idx = np.random.randint(0, 5, 1)
        return np.eye(5)[idx].flatten()
    
