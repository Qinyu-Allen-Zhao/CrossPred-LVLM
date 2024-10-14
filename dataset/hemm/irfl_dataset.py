import os
import json
from typing import Optional, Union, List
from PIL import Image
import requests
import torch
from datasets import load_dataset
import pandas as pd
import ast
from tqdm import tqdm, trange

from dataset.base import BaseDataset

from utils.func import shell_command
from dataset.hemm.prompts.irfl_prompt import IRFLPrompt


class IRFLDataset(BaseDataset):
    def __init__(self, download_dir="./", dataset_dir=None, annotation_file=None, **kwargs):
        super().__init__()
        self.download_dir = os.path.join(download_dir, "IRFL")
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)
        self.prompt = IRFLPrompt()
        self.load()

    def load(self):
        self.IRFL_images = load_dataset("lampent/IRFL", data_files='IRFL_images.zip', cache_dir=self.download_dir)['train']
        self.dataset = load_dataset("lampent/IRFL", "simile-detection-task", cache_dir=self.download_dir)["test"]
        self.dataset = pd.DataFrame(self.dataset)

        data = []
        for idx in trange(len(self.dataset)):
            row = self.dataset.iloc[idx]
            phrase = row['phrase']
            distractors = ast.literal_eval(row['distractors'])
            question = self.get_prompt(phrase)
            answer_image = ast.literal_eval(row['answer'])[0]
            distractors.append(answer_image)

            for i, distractor in enumerate(distractors):
                image_path = self.get_image_path_from_hugginface_cache(distractor)
                img = Image.open(image_path).convert("RGB")
                data.append({
                    "images": [img],
                    "question": question,
                    'category': "Whole dataset",
                    "label": "no" if i < 3 else "yes",
                    "eval_method": "hemm",
                })

        self.data = data

    def get_prompt(self, text) -> str:
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text
    
    def __len__(self):
        return len(self.dataset)

    def get_data(self, chunk_idx):
        data = [self.data[idx] for idx in chunk_idx]

        return data, []

    def get_image_path_from_hugginface_cache(self, image_name):
        # chached_image_path = self.IRFL_images[0]['image'].filename
        # chached_image_name = chached_image_path.split('/')[-1]
        # return chached_image_path.replace(chached_image_name, image_name.split('.')[0] + '.jpeg')
        image = image_name.split('.')[0] + '.jpeg'

        return os.path.join(self.download_dir, "images", image)
        