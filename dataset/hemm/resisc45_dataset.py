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
from dataset.hemm.prompts.resisc45_prompt import Resisc45Prompt


class Resisc45Dataset(BaseDataset):
    def __init__(self,
                download_dir="./",
                dataset_dir=None,
                annotation_file=None,
                **kwargs,
                ):
        super().__init__()
        self.dataset_dir = download_dir
        self.kaggle_api_path = kwargs["kaggle_api_path"]
        self.prompt = Resisc45Prompt()
        self.load()

    def load(self):
        os.environ['KAGGLE_CONFIG_DIR'] = self.kaggle_api_path
        if not os.path.exists(f'{self.dataset_dir}/nwpu-data-set.zip'):
            shell_command(f'kaggle datasets download -d happyyang/nwpu-data-set -p {self.dataset_dir}')
        if not os.path.exists(f'{self.dataset_dir}/resisc45'):
            shell_command(f'unzip {self.dataset_dir}/nwpu-data-set.zip -d {self.dataset_dir}/resisc45')
        
        self.images_dir = f'{self.dataset_dir}/resisc45/NWPU Data Set/NWPU-RESISC45/NWPU-RESISC45'
        classes = os.listdir(self.images_dir)

        images_list = []
        ground_truth_list = []
        for image_class in classes:
            x = os.listdir(os.path.join(self.images_dir, image_class))
            images_list.extend(x)
            ground_truth_list.extend([image_class for i in range(len(x))])

        data_list = []
        for x, y in zip(images_list, ground_truth_list):
            data_list.append((x, y))

        self.dataset = sorted(data_list)

    def get_prompt(self):
        prompt_text = self.prompt.format_prompt()
        return prompt_text
    
    def __len__(self):
        return len(self.dataset)

    def get_data(self, chunk_idx):
        data = []
        for idx in tqdm(chunk_idx):
            instance = self.dataset[idx]
            image_path = os.path.join(self.images_dir, instance[1], instance[0])
            img = Image.open(image_path).convert('RGB')
            ground_truth_answer = instance[1]
            text = self.get_prompt()

            data.append({
                "images": [img],
                "question": text,
                'category': "Whole dataset",
                "label": ground_truth_answer,
                "eval_method": "hemm",
            })

        return data, []
