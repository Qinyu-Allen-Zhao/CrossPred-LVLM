import os
import json
from typing import Optional, Union, List
from PIL import Image
import requests
import torch
from datasets import load_dataset
import subprocess
from tqdm import tqdm

from dataset.base import BaseDataset

from utils.func import shell_command
from dataset.hemm.prompts.memecaps_prompt import MemeCapsPrompt


class MemeCapsDataset(BaseDataset):
    def __init__(self,
                download_dir="./",
                images='memecap_images/memes',
                annotation_path='memes-test.json',
                **kwargs):

        self.download_dir = download_dir
        self.annotation_path = os.path.join(download_dir, annotation_path)
        self.images = os.path.join(download_dir, images)
        self.prompt = MemeCapsPrompt()
        self.load()

    def get_prompt(self, title, image_description) -> str:
        prompt_text = self.prompt.format_prompt(title, image_description)
        return prompt_text
    
    def __len__(self):
        annotation_file = json.load(open(self.annotation_path))
        return len(annotation_file)

    def load(self):
        if not os.path.exists(f'{self.download_dir}/memes.zip'):
            shell_command(f'gdown https://drive.google.com/uc?id=1o1IB6am0HdYS58CEOmmxra3WjJkrn-M1 -O {self.download_dir}')
        if not os.path.exists(f'{self.download_dir}/memes-test.json'):
            shell_command(f'wget https://raw.githubusercontent.com/eujhwang/meme-cap/main/data/memes-test.json -P {self.download_dir}')
        if not os.path.exists(f'{self.download_dir}/memecap_images/'):
            shell_command(f'unzip {self.download_dir}/memes.zip -d {self.download_dir}/memecap_images/')
        
    def get_data(self, chunk_idx):
        data = []
        annotation_file = json.load(open(self.annotation_path))
        
        for idx in tqdm(chunk_idx):
            data_dict = annotation_file[idx]
            image_path = f"{self.images}/{data_dict['img_fname'].strip()}"
            img = Image.open(image_path).convert("RGB")
            
            image_desc = data_dict["img_captions"][0]
            title = data_dict["title"]
            gt_caption = data_dict["meme_captions"][0]
            text = self.get_prompt(title, image_desc)
            
            data.append({
                "images": [img],
                "question": text,
                'category': "Whole dataset",
                "label": gt_caption,
                "eval_method": "hemm",
            })

        return data, []
