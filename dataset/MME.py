import os
import json

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from dataset.base import BaseDataset


class MME(BaseDataset):
    def __init__(self):
        super(MME, self).__init__()
        self.data = load_dataset("lmms-lab/MME")['test']

    def __len__(self):
        return len(self.data)
    
    def get_data(self, chunk_idx):
        data = []
        for idx in tqdm(chunk_idx):
            ins = self.data[idx]
            data.append({
                "images": [ins['image'].convert("RGB")],
                "question": ins['question'],
                'category': ins['category'],
                "label": ins['answer'],
                "eval_method": "true_false_en",
            })

        return data, ['category']

    def sample(self, num_samples):
        data_id = np.arange(len(self))

        if num_samples is not None:
            np.random.seed(0)
            categories = np.array(self.data['category'])
            sampled_dict = []
    
            for category in np.unique(categories):
                idx = data_id[categories == category]
                if len(idx) < 30:
                    continue
                else:
                    # print(f"Category {category} has {len(idx)} samples")
                    sampled_dict.append(np.random.choice(idx, min(len(idx), num_samples), replace=False))
            
            return np.concatenate(sampled_dict)


        return data_id