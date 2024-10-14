import os
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm
import subprocess
from glob import glob

from dataset.base import BaseDataset

from utils.func import shell_command
from dataset.hemm.prompts.inat_prompt import INATPrompt


class INATDataset(BaseDataset):
    def __init__(self,
                download_dir="./",
                dataset_dir='inat/val',
                annotation_file=None,
                **kwargs,
                ):
        super().__init__()
        self.download_dir = download_dir
        self.image_dir = os.path.join(download_dir, dataset_dir)
        self.kaggle_api_path = kwargs["kaggle_api_path"]
        self.prompt = INATPrompt()
        self.load()
    
    def load(self):
        if not os.path.exists(f"{self.download_dir}/inat/"):
            shell_command(f"mkdir -p {self.download_dir}/inat")
            shell_command(f"wget  https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz -P {self.download_dir}/inat/")
            shell_command(f"tar -xvf {self.download_dir}/inat/val.tar.gz -C {self.download_dir}/inat/")

    def __len__(self):
        all_images = glob(f"{self.image_dir}/*/*.jpg")
        return len(all_images)

    def get_prompt(self) -> str:
        prompt_text = self.prompt.format_prompt()
        return prompt_text

    def get_data(self, chunk_idx):
        data = []
    
        all_images = sorted(glob(f"{self.image_dir}/*/*.jpg"))

        for idx in tqdm(chunk_idx):
            image_path = all_images[idx]
            img = Image.open(image_path).convert("RGB")

            text = self.get_prompt()
            label = " ".join(all_images[idx].split("/")[-2].split("_")[-2:])
            
            data.append({
                "images": [img],
                "question": text,
                'category': "Whole dataset",
                "label": label,
                "eval_method": "hemm",
            })

        return data, []

    