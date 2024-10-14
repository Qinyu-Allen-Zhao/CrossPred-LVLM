import re
import ast

from tqdm import tqdm
import numpy as np
from datasets import load_dataset

from dataset.base import BaseDataset


class ScienceQA(BaseDataset):
    def __init__(self):
        super(ScienceQA, self).__init__()
        self.data = load_dataset("lmms-lab/ScienceQA", "ScienceQA-IMG")['test']
         
    def __len__(self):
        return len(self.data)

    def get_data(self, chunk_idx):
        data = []
        for idx in tqdm(chunk_idx):
            ins = self.data[idx]
            data.append({
                "images": [ins['image'].convert("RGB")],
                "question": self.create_options_prompt(ins),
                'category': ins['category'],
                "label": chr(65 + ins['answer']),
                "options": ins['choices'],
                "eval_method": "multi_choice",
            })

        return data, ['category']
    
    def create_options_prompt(self, ins):
        if ins['hint'] is not None and ins['hint'] != "":
            context = f"Context: {ins['hint']}\n"
            options_prompt = [context]
        else:
            options_prompt = []

        question = f"{ins['question']}\n"
        options_prompt.append(question)

        for i, opt in enumerate(ins['choices']):
            options_prompt.append(f"{chr(65+i)}. {opt}\n")
        
        options_prompt.append("Answer with the option's letter from the given choices directly.")
        
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
