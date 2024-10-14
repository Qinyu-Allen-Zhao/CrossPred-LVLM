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
from dataset.hemm.prompts.screen2words_prompt import Screen2WordsPrompt


class Screen2WordsDataset(BaseDataset):
    def __init__(self,
                download_dir="./",
                dataset_dir=None,
                annotation_file=None,
                **kwargs,
                ):
        super().__init__()
        self.dataset_dir = download_dir
        self.kaggle_api_path = kwargs["kaggle_api_path"]
        self.prompt = Screen2WordsPrompt()
        self.images_dir = os.path.join(self.dataset_dir, 'screen2wordsimages/unique_uis/combined')
        self.csv_path = os.path.join(self.dataset_dir, 'screen2words/screen_summaries.csv')
        self.test_file = os.path.join(self.dataset_dir, 'screen2words/split/test_screens.txt')
        self.load()

    def __len__(self):
        data_file = open(self.test_file, 'r')
        return len(data_file.readlines())        

    def load(self):
        os.environ['KAGGLE_CONFIG_DIR'] = self.kaggle_api_path
        if not os.path.exists(f'{self.dataset_dir}/rico-dataset.zip'):
            shell_command(f'kaggle datasets download -d onurgunes1993/rico-dataset -p {self.dataset_dir}')
        if not os.path.exists(f'{self.dataset_dir}/screen2wordsimages'):
            shell_command(f'unzip {self.dataset_dir}/rico-dataset.zip -d {self.dataset_dir}/screen2wordsimages')
        if not os.path.exists(f'{self.dataset_dir}/screen2words'):
            shell_command(f'git clone https://github.com/google-research-datasets/screen2words {self.dataset_dir}/screen2words')

    def get_prompt(self):
        prompt_text = self.prompt.format_prompt()
        return prompt_text

    def get_ground_truth(self, filename):
        ground_truth = list(self.dataset[self.dataset["screenId"] == int(filename)]["summary"])[-1]
        return ground_truth

    def get_data(self, chunk_idx):
        data = []
        self.dataset = pd.read_csv(self.csv_path)
        data_file = open(self.test_file, 'r')
        data_lines = data_file.readlines()

        for idx in tqdm(chunk_idx):
            line = data_lines[idx]
            file_name = line.strip()
            image_path = os.path.join(self.images_dir, file_name + '.jpg')
            img = Image.open(image_path).convert('RGB')
            ground_truth_answer = self.get_ground_truth(file_name)
            text = self.get_prompt()

            data.append({
                "images": [img],
                "question": text,
                'category': "Whole dataset",
                "label": ground_truth_answer,
                "eval_method": "hemm",
            })

        return data, []
        