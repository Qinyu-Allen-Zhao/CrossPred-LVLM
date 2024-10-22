import os
import json
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import pandas as pd

from dataset.base import BaseDataset

from utils.func import shell_command
from dataset.hemm.prompts.ucmerced_prompt import UCMercedPrompt


class UCMercedDataset(BaseDataset):
    def __init__(self,
                download_dir="./",
                dataset_dir=None,
                annotation_file=None,
                **kwargs,
                ):
        super().__init__()
        self.dataset_dir = download_dir
        self.kaggle_api_path = kwargs["kaggle_api_path"]
        self.prompt = UCMercedPrompt()
        self.load()

    def load(self):
        os.environ['KAGGLE_CONFIG_DIR'] = self.kaggle_api_path
        if not os.path.exists(f'{self.dataset_dir}/landuse-scene-classification.zip'):
            shell_command(f'kaggle datasets download -d apollo2506/landuse-scene-classification -p {self.dataset_dir}')
        if not os.path.exists(f'{self.dataset_dir}/ucmercedimages'):
            shell_command(f'unzip {self.dataset_dir}/landuse-scene-classification.zip -d {self.dataset_dir}/ucmercedimages/')

    def get_prompt(self):
        prompt_text = self.prompt.format_prompt()
        return prompt_text
    
    def __len__(self):
        csv_path = f'{self.dataset_dir}/ucmercedimages/validation.csv'
        df = pd.read_csv(csv_path)
        return len(df)

    def get_data(self, chunk_idx):
        csv_path = f'{self.dataset_dir}/ucmercedimages/validation.csv'
        df = pd.read_csv(csv_path)
        images_dir = f'{self.dataset_dir}/ucmercedimages/images_train_test_val/validation'
    
        data = []
        for idx in tqdm(chunk_idx):
            row = df.iloc[idx]
            image_path = os.path.join(images_dir, row['Filename'])
            img = Image.open(image_path).convert("RGB")
            ground_truth_answer = row['ClassName']
            text = self.get_prompt()

            data.append({
                "images": [img],
                "question": text,
                'category': "Whole dataset",
                "label": ground_truth_answer,
                "eval_method": "hemm",
            })

        return data, []