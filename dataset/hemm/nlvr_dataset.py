from torch.utils.data import Dataset, DataLoader
import requests
from PIL import Image
import os
import json
from typing import Optional, Union, List
from PIL import Image
import requests
import torch
from datasets import load_dataset
from tqdm import tqdm

from dataset.base import BaseDataset

from utils.func import shell_command
from dataset.hemm.prompts.nlvr_prompt import nlvrprompt


class NLVRDataset(BaseDataset):
    def __init__(self,
                download_dir="./",
                dataset_dir='nlvr/nlvr/dev/',
                annotation_file=None,
                **kwargs,
                ):
        super().__init__()
        self.download_dir = download_dir
        self.dataset_dir = os.path.join(download_dir, dataset_dir)
        self.prompt = nlvrprompt()

        self.load()
        self.image_dir = os.path.join(self.dataset_dir, 'images/')
        with open(os.path.join(self.dataset_dir, 'dev.json'), "r") as f:
            self.sentences = f.readlines()

    def __len__(self):
        return len(self.sentences)

    def load(self):
        if not os.path.exists(f"{self.download_dir}/nlvr/"):
            shell_command(f'git clone https://github.com/lil-lab/nlvr.git {self.download_dir}/nlvr')

    def get_prompt(self, text) -> str:
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text

    def get_data(self, chunk_idx):
        data = []

        for idx in tqdm(chunk_idx):
            line = self.sentences[idx]
            ann = json.loads(line)
            img_path = os.path.join(self.image_dir, f'{ann["directory"]}/dev-{ann["identifier"]}-0.png')
            img = Image.open(img_path).convert('RGB')

            sentence = ann['sentence']
            text = self.get_prompt(sentence)

            label = ann['label']

            data.append({
                    "images": [img],
                    "question": text,
                    'category': "Whole dataset",
                    "label": label.lower(),
                    "eval_method": "hemm",
                })

        return data, []
