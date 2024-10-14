import os
import json

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from dataset.base import BaseDataset


class MMBench(BaseDataset):
    def __init__(self, split):
        super(MMBench, self).__init__()
        self.data = load_dataset("lmms-lab/MMBench", split)['dev']
        self.split = split

    def __len__(self):
        return len(self.data)
    
    def get_data(self, chunk_id):
        data = []
        for idx in tqdm(chunk_id):
            ins = self.data[idx]
            data.append({
                "images": [ins['image'].convert("RGB")],
                "question": self.create_options_prompt(ins),
                "label": ins['answer'],
                'category': ins['category'],
                "options": [ins["A"], ins["B"], ins["C"], ins["D"]],
                "eval_method": "multi_choice"
            })

        return data, ['category']
    
    def create_options_prompt(self, ins):
        options_prompt = [f"{ins['question']}\n"]
        options_prompt.append(f"{ins['hint']}\n")
        
        for key in ['A', 'B', 'C', 'D']:
            item = ins[key]
            if item != "nan":
                options_prompt.append(f"{key}. {item}\n")
        if self.split == 'en':
            options_prompt.append("Answer with the option's letter from the given choices directly.")
        else:
            options_prompt.append("请直接使用所提供的选项字母作为答案回答。")

        return "".join(options_prompt)
    
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