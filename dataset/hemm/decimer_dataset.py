import os
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm
import pickle

from dataset.base import BaseDataset

from utils.func import shell_command
from dataset.hemm.prompts.decimer_prompt import DecimerPrompt


class DecimerDataset(BaseDataset):
    def __init__(self,
                download_dir="./",
                dataset_dir='DECIMER/DECIMER_HDM_Dataset_Images/DECIMER_HDM_Dataset_Images',
                annotation_file='DECIMER/DECIMER_HDM_Dataset_SMILES.tsv',
                **kwargs,
                ):
        super().__init__()
        self.download_dir = os.path.join(download_dir, "DECIMER")
        self.kaggle_api_path = kwargs["kaggle_api_path"]
        self.image_dir = os.path.join(download_dir, dataset_dir)
        self.prompt = DecimerPrompt()

        self.load()
        with open(os.path.join(download_dir, annotation_file)) as f:
            self.annotations = f.readlines()
            
        self.annotations = self.annotations[1:]
        self.annotations = self.annotations[-len(self.annotations) // 10:]

    def __len__(self,):
        return len(self.annotations)

    def load(self):
        os.environ['KAGGLE_CONFIG_DIR'] = self.kaggle_api_path
        if not os.path.exists(self.download_dir):
            shell_command(f'mkdir -p {self.download_dir}')
        if not os.path.exists(f'{self.download_dir}/decimer.zip'):
            shell_command(f'kaggle datasets download -d juliajakubowska/decimer -p {self.download_dir}')
            shell_command(f'unzip {self.download_dir}/decimer.zip -d {self.download_dir}')

    def get_prompt(self):
        prompt_text = self.prompt.format_prompt()
        return prompt_text

    def get_data(self, chunk_idx):
        data = []
        for idx in tqdm(chunk_idx):
            row = self.annotations[idx]
            img_id, label = row.strip().split("\t")
            image_path = f"{self.image_dir}/{img_id}.png"
            img = Image.open(image_path).convert("RGB")
            text = self.get_prompt()
            
            data.append({
                "images": [img],
                "question": text,
                'category': "Whole dataset",
                "label": label,
                "eval_method": "hemm",
            })

        return data, []
    