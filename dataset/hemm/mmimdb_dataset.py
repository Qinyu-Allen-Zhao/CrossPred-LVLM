import os
import json
from glob import glob
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm

from dataset.base import BaseDataset

from utils.func import shell_command
from dataset.hemm.prompts.mmimdb_prompt import MMIMDBPrompt
from huggingface_hub import snapshot_download

class MMIMDBDataset(BaseDataset):
    def __init__(self,
                download_dir="./",
                dataset_dir='mmimdb/',
                annotation_file="mmimdb/split.json",
                **kwargs,
                ):
        super().__init__()
        self.download_dir = download_dir
        self.kaggle_api_path = kwargs["kaggle_api_path"]
        self.dataset_dir = os.path.join(download_dir, dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        
        self.prompt = MMIMDBPrompt()
        self.annotation_file = os.path.join(download_dir, annotation_file)
        self.load()

    def load(self):
        os.environ['KAGGLE_CONFIG_DIR'] = self.kaggle_api_path
        if not os.path.exists(f"{self.download_dir}/mmimdb"):
            shell_command(f"mkdir -p {self.download_dir}/mmimdb")
            snapshot_download(repo_id="akshayg08/mmimdb_test", repo_type="dataset", local_dir=f"{self.download_dir}/mmimdb/")
            
    def __len__(self):
        ann_files = json.load(open(self.annotation_file))["test"]
        return len(ann_files)

    def get_prompt(self, text):
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text

    def get_data(self, chunk_idx):
        data = []
        ann_files = json.load(open(self.annotation_file))["test"]
        
        for idx in tqdm(chunk_idx):
            ann_id = ann_files[idx].strip()
            image_path = f"{self.image_dir}/{ann_id}.jpeg"
            img = Image.open(image_path).convert('RGB')
            
            ann = json.load(open(f"{self.dataset_dir}/annotations/{ann_id}.json"))
            text = self.get_prompt(ann["plot"][0])
            label = ", ".join(ann["genres"])

            data.append({
                "images": [img],
                "question": text,
                'category': "Whole dataset",
                "label": label,
                "eval_method": "hemm",
            })

        return data, []
