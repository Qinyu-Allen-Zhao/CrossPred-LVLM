import os
import json

import numpy as np
from tqdm import tqdm
from datasets import load_dataset

from dataset.base import BaseDataset


class SEEDBench(BaseDataset):
    def __init__(self, version='1'):
        super(SEEDBench, self).__init__()
        if version == '1':
            self.data = load_dataset("lmms-lab/SEED-Bench")['test']
            self.QUE_TYPE_MAP = [
                    "Scene Understanding",
                    "Instance Identity",
                    "Instance Location",
                    "Instance Attributes",
                    "Instances Counting",
                    "Spatial Relation",
                    "Instance Interaction",
                    "Visual Reasoning",
                    "Text Understanding",
                    "Action Recognition",
                    "Action Prediction",
                    "Procedure Understanding"
            ]
        else:
            self.data = load_dataset("lmms-lab/SEED-Bench-2")['test']
            self.QUE_TYPE_MAP = [
                "Scene Understanding",
                "Instance Identity",
                "Instance Attributes",
                "Instance Location",
                "Instances Counting",
                "Spatial Relation",
                "Instance Interaction",
                "Visual Reasoning",
                "Text Understanding",
                "Celebrity Recognition",
                "Landmark Recognition",
                "Chart Understanding",
                "Visual Referring Expression",
                "Science Knowledge",
                "Emotion Recognition",
                "Visual Mathematics",
                "Difference Spotting",
                "Meme Comprehension",
                "Global Video Understanding",
                "Action Recognition",
                "Action Prediction",
                "Procedure Understanding",
                "In-Context Captioning",
                "Interleaved Image-Text Analysis",
                "Text-to-Image Generation",
                "Next Image Prediction",
                "Text-Image Creation"
            ]
        self.TEMP = "%s\nA. %s\nB. %s\nC. %s\nD. %s\nAnswer with the option's letter from the given choices directly."
         
    def __len__(self):
        return len(self.data)

    def get_data(self, chunk_idx):
        data = []
        for idx in tqdm(chunk_idx):
            ins = self.data[idx]
            data.append({
                "images": [img.convert("RGB") for img in ins['image']],
                "question": self.TEMP % (ins['question'], ins['choice_a'], ins['choice_b'], ins['choice_c'], ins['choice_d']),
                'category': self.QUE_TYPE_MAP[ins['question_type_id']-1],
                "label": ins['answer'],
                "options": [ins["choice_a"], ins["choice_b"], ins["choice_c"], ins["choice_d"]],
                "eval_method": "multi_choice",
            })

        return data, ['category']

    def sample(self, num_samples):
        data_id = np.arange(len(self))

        if num_samples is not None:
            np.random.seed(0)
            categories = self.data['question_type_id']
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