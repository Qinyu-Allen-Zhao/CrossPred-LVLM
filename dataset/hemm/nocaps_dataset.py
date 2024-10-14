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
from dataset.hemm.prompts.nocaps_prompt import NoCapsPrompt
from huggingface_hub import snapshot_download

class NoCapsDataset(BaseDataset):
    def __init__(self,
                    download_dir="./", 
                    dataset_dir="nocaps_images/",
                    annotation_file='nocaps_images/nocaps_val_4500_captions.json',
                    **kwargs):
        super().__init__()
        self.annotation_file = os.path.join(download_dir, annotation_file)
        self.dataset_dir = os.path.join(download_dir, dataset_dir)
        self.prompt = NoCapsPrompt()
        self.load()

    def get_prompt(self) -> str:
        prompt_text = self.prompt.format_prompt()
        return prompt_text
    
    def __len__(self):
        json_file = json.load(open(self.annotation_file, 'r'))
        return len(json_file["images"])

    def load(self):
        if not os.path.exists(self.dataset_dir):
            shell_command(f"mkdir -p {self.dataset_dir}")
            snapshot_download(repo_id="akshayg08/NocapsTest", repo_type="dataset", local_dir=self.dataset_dir)
            shell_command(f"wget https://s3.amazonaws.com/nocaps/nocaps_val_4500_captions.json -P {self.dataset_dir}")
    
    def get_data(self, chunk_idx):
        data = []
        json_file = json.load(open(self.annotation_file, 'r'))
        
        for idx in tqdm(chunk_idx):
            image_dict = json_file['images'][idx]
            fn = image_dict["file_name"]
            captions = []
            for ann in json_file["annotations"][10*idx: 10*(idx + 1)]:
                captions.append(ann["caption"])
            
            image_path = os.path.join(self.dataset_dir, fn)
            img = Image.open(image_path).convert('RGB')

            text = self.get_prompt()

            data.append({
                "images": [img],
                "question": text,
                'category': "Whole dataset",
                "label": captions,
                "eval_method": "hemm",
            })

        return data, []
    