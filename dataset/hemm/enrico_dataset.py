import os
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm

from dataset.base import BaseDataset

from utils.func import shell_command
from dataset.hemm.prompts.enrico_prompt import EnricoPrompt

class EnricoDataset(BaseDataset):
    def __init__(self,
                download_dir="./",
                dataset_dir='enrico/screenshots',
                annotation_file="enrico/design_topics.csv",
                **kwargs, 
                 ):
        super().__init__()
        self.download_dir = os.path.join(download_dir, "enrico")
        self.image_dir = os.path.join(download_dir, dataset_dir)
        self.prompt = EnricoPrompt()
        self.load()
        with open(os.path.join(download_dir, annotation_file)) as f:
            self.annotations = f.readlines()
        
        self.annotations = self.annotations[1:]
        self.annotations = self.annotations[-len(self.annotations) // 10:]
     
    def __len__(self,):
        return len(self.annotations)

    def load(self):
        if not os.path.exists(self.download_dir):
            shell_command(f'mkdir -p {self.download_dir}')
            shell_command(f'wget http://userinterfaces.aalto.fi/enrico/resources/screenshots.zip -P {self.download_dir}')
            shell_command(f'unzip {self.download_dir}/screenshots.zip -d {self.download_dir}')
            shell_command(f"wget https://raw.githubusercontent.com/luileito/enrico/master/design_topics.csv -P {self.download_dir}")

    def get_prompt(self):
        prompt_text = self.prompt.format_prompt()
        return prompt_text

    def get_data(self, chunk_idx):
        data = []
        for idx in tqdm(chunk_idx):
            row = self.annotations[idx]
            img_id, label = row.strip().split(",")
            image_path = f"{self.image_dir}/{img_id}.jpg"
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
    