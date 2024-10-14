import os
import json
from typing import Optional, Union, List
from PIL import Image
import requests
import torch
import pandas as pd
import subprocess
from tqdm import tqdm, trange

from dataset.base import BaseDataset

from utils.func import shell_command
from dataset.hemm.prompts.newyorkercartoon_prompt import NewYorkerCartoonPrompt


class NewYorkerCartoonDataset(BaseDataset):
    def __init__(self,
                download_dir="./",
                dataset_dir='caption-contest-data/',
                annotation_file=None,
                **kwargs, 
                ):
        super().__init__()
        self.download_dir = download_dir
        self.dataset_dir = os.path.join(download_dir, dataset_dir)
        self.prompt = NewYorkerCartoonPrompt()

        self.image_dir = os.path.join(self.dataset_dir, 'cartoons')
        self.caption_dir = os.path.join(self.dataset_dir, 'summaries')
        self.csv_path_suffix_1 = 'LilUCB'
        self.csv_path_suffix_2 = 'lil-KLUCB'

        self.load()

    def load(self):
        if not os.path.exists(f"{self.download_dir}/caption-contest-data"):
            shell_command(f'git clone https://github.com/nextml/caption-contest-data {os.path.join(self.download_dir, "caption-contest-data")}')

        image_list = sorted(os.listdir(self.image_dir))
        data = []
        
        for idx in trange(len(image_list)):
            img_file = image_list[idx]
            img_id = img_file.split('.jpg')[0]
            if img_id == "890":
                # Broken image
                continue

            img_path = os.path.join(self.image_dir, img_file)
            img = Image.open(img_path).convert('RGB')

            if os.path.exists(os.path.join(self.caption_dir, img_id+'.csv')):
                df = pd.read_csv(os.path.join(self.caption_dir, img_id+ '.csv'))
            elif os.path.exists(os.path.join(self.caption_dir, img_id+"_"+self.csv_path_suffix_1+'.csv')):
                df = pd.read_csv(os.path.join(self.caption_dir, img_id+"_"+self.csv_path_suffix_1+'.csv'))
            elif os.path.exists(os.path.join(self.caption_dir, img_id+"_"+self.csv_path_suffix_2+'.csv')):
                df = pd.read_csv(os.path.join(self.caption_dir, img_id+"_"+self.csv_path_suffix_2+'.csv'))
            
            captions = []
            captions.append(df.iloc[0]['caption'])

            for i in range(1, 5):
                captions.append(df.iloc[-1*i]['caption'])
            
            for i in range(len(captions)):
                text = self.get_prompt(captions[i])
                data.append({
                    "images": [img],
                    "question": text,
                    'category': "Whole dataset",
                    "label": "yes" if i == 0 else "no",
                    "eval_method": "hemm",
                })
        self.data = data


    def get_prompt(self, text) -> str:
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text
    
    def __len__(self):
        return len(os.listdir(self.image_dir))

    def get_data(self, chunk_idx):
        data = [self.data[idx] for idx in chunk_idx]

        return data, []
