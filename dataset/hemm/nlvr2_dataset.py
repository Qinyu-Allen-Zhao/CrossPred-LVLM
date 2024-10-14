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
from dataset.hemm.prompts.nlvr2prompt import NLVR2prompt


class NLVR2Dataset(BaseDataset):
    def __init__(self, download_dir="./", dataset_dir=None, annotation_file="nlvr/nlvr2/data/dev_100.json", **kwargs):
        self.download_dir = download_dir
        self.load()
        with open(os.path.join(download_dir, annotation_file)) as annotation_file:
            self.data_json = json.load(annotation_file)
        self.prompt = NLVR2prompt()

    def __len__(self):
        return len(self.data_json)
    
    def get_prompt(self, text):
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text

    def load(self):
        if not os.path.exists(f"{self.download_dir}/nlvr/"):
            shell_command(f'git clone https://github.com/lil-lab/nlvr.git {self.download_dir}/nlvr/')

    def get_data(self, chunk_idx):
        data = []
        for idx in tqdm(chunk_idx):
            left_img_path = self.data_json[idx]['left_img_path']
            right_img_path = self.data_json[idx]['right_img_path']
            label = self.data_json[idx]['label']
            label = label.lower()
            sentence = self.data_json[idx]['sentence']

            img_1 = Image.open(left_img_path)
            img_2 = Image.open(right_img_path)
            size1 = img_1.size
            size2 = img_2.size
            avg_size = ((size1[0]+size2[0])//2,(size1[1]+size2[1])//2)
            img_1 = img_1.resize(avg_size)
            img_2 = img_2.resize(avg_size)
            image = Image.new('RGB',(2*avg_size[0],avg_size[1]), (255,255,255))
            image.paste(img_1,(0,0))
            image.paste(img_2,(avg_size[0],0))

            text = self.get_prompt(sentence)
            
            data.append({
                "images": [image],
                "question": text,
                'category': "Whole dataset",
                "label": label,
                "eval_method": "hemm",
            })

        return data, []
    