import os
import json
from PIL import Image, ImageFile
import requests
import torch
import subprocess
from tqdm import tqdm
import pandas as pd

from dataset.base import BaseDataset

from utils.func import shell_command
from dataset.hemm.prompts.memotion_prompt import MemotionPrompt


ImageFile.LOAD_TRUNCATED_IMAGES = True

def ref_text(text):
    sents = text.split("\n")
    sents = [sent.strip() for sent in sents]
    return " ".join(sents).strip()

class MemotionDataset(BaseDataset):
    def __init__(self,
                download_dir="./",
                dataset_dir=None,
                annotation_file=None,
                **kwargs,
                ):
        super().__init__()
        self.download_dir = download_dir
        self.data_path = os.path.join(download_dir, 'memotion-dataset-7k/memotion_dataset_7k/labels.xlsx')
        self.image_dir = os.path.join(download_dir, 'memotion-dataset-7k/memotion_dataset_7k/images')
        self.kaggle_api_path = kwargs["kaggle_api_path"]
        self.prompt = MemotionPrompt()
        self.choices = ['funny', 'very_funny', 'not_funny', 'hilarious']
        self.load()

    def __len__(self):
        df = pd.read_excel(self.data_path)
        return len(df)

    def get_prompt(self, caption) -> str:
        prompt_text = self.prompt.format_prompt(caption)
        return prompt_text

    def load(self):
        os.environ['KAGGLE_CONFIG_DIR'] = self.kaggle_api_path
        if not os.path.exists(f'{self.download_dir}/memotion-dataset-7k.zip'):
          shell_command(f'kaggle datasets download -d williamscott701/memotion-dataset-7k -p {self.download_dir}')
        if not os.path.exists(f'{self.download_dir}/memotion-dataset-7k'):
          shell_command(f'unzip {self.download_dir}/memotion-dataset-7k.zip -d {self.download_dir}/memotion-dataset-7k')

    def get_data(self, chunk_idx):
        data = []
        df = pd.read_excel(self.data_path)

        for idx in tqdm(chunk_idx):
            row = df.iloc[idx]
            image_path = os.path.join(self.image_dir, row['image_name'])
            img = Image.open(image_path).convert("RGB")
            caption = row['text_corrected']
            label = row['humour']
            text = self.get_prompt(caption)
            data.append({
                "images": [img],
                "question": text,
                'category': "Whole dataset",
                "label": label,
                "eval_method": "hemm",
            })

        return data, []

