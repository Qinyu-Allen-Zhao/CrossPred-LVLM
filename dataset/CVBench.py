import os
import json

import numpy as np
from tqdm import tqdm
from datasets import load_dataset

from dataset.base import BaseDataset


class CVBench(BaseDataset):
    def __init__(self):
        super().__init__()

        self.data = load_dataset("nyu-visionx/CV-Bench")['test']
  
    def __len__(self):
        return len(self.data)

    def get_data(self, chunk_idx):
        data = []
        for idx in tqdm(chunk_idx):
            ins = self.data[idx]
            data.append({
                "images": [ins['image'].convert("RGB")],
                "question": self.create_options_prompt(ins),
                'category': f"{ins['type']} {ins['task']}",
                "label": ins['answer'][1],
                "options": ins['choices'],
                "eval_method": "multi_choice",
            })

        return data, ['category']

    def create_options_prompt(self, ins):
        question = ins['question']
        options_prompt = [question, "\n"]

        for i, opt in enumerate(ins['choices']):
            options_prompt.append(f"{chr(65+i)}. {opt}\n")
        
        options_prompt.append("Answer with the option's letter from the given choices directly.")

        return "".join(options_prompt)

    def sample(self, num_samples):
        data_id = np.arange(len(self))

        if num_samples is not None:
            np.random.seed(0)
            categories = np.array([f"{type_big} {task}" for type_big, task in zip(self.data['type'], self.data['task'])])
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
