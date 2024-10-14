import os
import json
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from dataset.base import BaseDataset

from utils.func import shell_command
from dataset.hemm.prompts.winoground_prompt import WinogroundPrompt


class WinogroundDataset(BaseDataset):
    def __init__(self,
                download_dir="./",
                dataset_dir=None,
                annotation_file=None,
                **kwargs,
                ):
        super().__init__()
        self.dataset_dir = download_dir
        self.prompt = WinogroundPrompt()
        self.hf_auth_token = kwargs["hf_auth_token"]
        self.load()
    
    def load(self):
        self.dataset = load_dataset("facebook/winoground", 
                                    use_auth_token="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", trust_remote_code=True)['test']
        data = []
        for idx in trange(len(self.dataset)):
            instance = self.dataset[idx]

            for pair in ((0,0), (0,1), (1,0), (1,1)):
                image = instance['image_'+str(pair[0])]
                image = image.convert('RGB')
                query = instance['caption_'+str(pair[1])]
                text = self.get_prompt(query)
                label = 'yes' if pair[0] == pair[1] else 'no'
                
                data.append({
                    "images": [image],
                    "question": text,
                    'category': "Whole dataset",
                    "label": label,
                    "eval_method": "hemm",
                })
        self.data = data

    def __len__(self):
        return len(self.data)

    def get_prompt(self, text):
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text

    def get_data(self, chunk_idx):
        data = [self.data[idx] for idx in chunk_idx]

        return data, []