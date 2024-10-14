import os
import json
from PIL import Image
import requests
import torch
import subprocess
from tqdm import tqdm
import pandas as pd

from dataset.base import BaseDataset

from utils.func import shell_command
from dataset.hemm.prompts.flickr30k_prompt import Flickr30kPrompt


class Flickr30kDataset(BaseDataset):
    def __init__(self,
                download_dir="./",
                dataset_dir="flickr30k_images/flickr30k_images/",
                annotation_file="flickr30k_images/flickr30k_test.json",
                **kwargs,
                 ):
        super().__init__()
        self.download_dir = download_dir
        self.kaggle_api_path = kwargs["kaggle_api_path"]
        self.load()
        self.image_dir = os.path.join(self.download_dir, dataset_dir)
        annotation_file = os.path.join(self.download_dir, annotation_file)
        self.annotation = json.load(open(annotation_file, "r"))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prompt = Flickr30kPrompt()

    def get_prompt(self) -> str:
        prompt_text = self.prompt.format_prompt()
        return prompt_text
    
    def __len__(self,):
        return len(self.annotation)

    def load(self):
        os.environ['KAGGLE_CONFIG_DIR'] = self.kaggle_api_path
        if not os.path.exists(f'{self.download_dir}/flickr-image-dataset.zip'):
            shell_command(f'kaggle datasets download -d hsankesara/flickr-image-dataset -p {self.download_dir}')
        if not os.path.exists(f'{self.download_dir}/flickr30k_images'):
            shell_command(f'unzip {self.download_dir}/flickr-image-dataset.zip -d {self.download_dir}')
            shell_command(f"wget https://huggingface.co/datasets/akshayg08/Flickr30k_test/raw/main/flickr30k_test.json -P {self.download_dir}/flickr30k_images/")

    def get_data(self, chunk_idx):
        data = []
        for idx in tqdm(chunk_idx):
            ann = self.annotation[idx]
            image_path = f"{self.image_dir}/{ann['image'].split('/')[-1]}"
            img = Image.open(image_path).convert("RGB")

            label = ann["caption"][0]
            text = self.get_prompt()

            data.append({
                "images": [img],
                "question": text,
                'category': "Whole dataset",
                "label": label,
                "eval_method": "hemm",
            })

        return data, []

    