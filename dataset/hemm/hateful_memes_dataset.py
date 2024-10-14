import os
import json
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm
import subprocess

from dataset.base import BaseDataset

from utils.func import shell_command
from dataset.hemm.prompts.hateful_memes_prompt import HatefulMemesPrompt


class HatefulMemesDataset(BaseDataset):
    def __init__(self,
                download_dir="./",
                dataset_dir='hateful_memes',
                annotation_file='dev.jsonl',
                **kwargs,
                 ):
        super().__init__()
        self.download_dir = download_dir
        self.dataset_dir = os.path.join(download_dir, dataset_dir)
        self.evaluate_path = annotation_file
        self.kaggle_api_path = kwargs["kaggle_api_path"]
        self.prompt = HatefulMemesPrompt()
        self.load()

    def load(self):
        os.environ['KAGGLE_CONFIG_DIR'] = self.kaggle_api_path
        if not os.path.exists(f'{self.download_dir}/facebook-hateful-meme-dataset.zip'):
          shell_command(f'kaggle datasets download -d parthplc/facebook-hateful-meme-dataset -p {self.download_dir}')
        if not os.path.exists(f'{self.download_dir}/hateful_memes'):
          shell_command(f'unzip {self.download_dir}/facebook-hateful-meme-dataset.zip -d {self.download_dir}/hateful_memes/')
    
    def __len__(self):
        label_path = os.path.join(self.dataset_dir, 'data', self.evaluate_path)
        json_list = list(open(label_path, 'r'))
        return len(json_list)

    def get_prompt(self, text) -> str:
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text

    def get_data(self, chunk_idx):
        data = []
        
        label_path = os.path.join(self.dataset_dir, 'data', self.evaluate_path)
        json_list = list(open(label_path, 'r'))
        image_dir = os.path.join(self.dataset_dir, 'data')

        for idx in tqdm(chunk_idx):
            json_obj = json.loads(json_list[idx])
            text = self.get_prompt(json_obj['text'])
            image_path = os.path.join(image_dir, json_obj['img'])
            img = Image.open(image_path).convert("RGB")

            if json_obj["label"]:
                label = "yes"
            else:
                label = "no"

            data.append({
                "images": [img],
                "question": text,
                'category': "Whole dataset",
                "label": label,
                "eval_method": "hemm",
            })

        return data, []
