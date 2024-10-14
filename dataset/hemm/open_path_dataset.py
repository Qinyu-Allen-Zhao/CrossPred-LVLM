import os
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm

from dataset.base import BaseDataset

from utils.func import shell_command
from dataset.hemm.prompts.plip_kather_prompt import PlipKatherPrompt
from huggingface_hub import snapshot_download


class OpenPathDataset(BaseDataset):
    def __init__(self,
                download_dir="./",
                dataset_dir='open_path/',
                annotation_file="open_path/Kather_test/Kather_test.csv",
                **kwargs,
                ):
        super().__init__()
        self.download_dir = download_dir
        self.load()
        self.image_dir = os.path.join(download_dir, dataset_dir)
        self.prompt = PlipKatherPrompt()
        with open(os.path.join(download_dir, annotation_file)) as f:
            self.annotation = f.readlines()
        self.annotation = self.annotation[1:]

    def load(self):
        if not os.path.exists(f"{self.download_dir}/open_path"):
            shell_command(f"mkdir -p {self.download_dir}/open_path")
            snapshot_download(repo_id="akshayg08/OpenPath", repo_type="dataset", local_dir=f"{self.download_dir}/open_path/")

    def __len__(self):
        return len(self.annotation)

    def get_prompt(self):
        prompt_text = self.prompt.format_prompt()
        return prompt_text

    def get_data(self, chunk_idx):
        data = []
        for idx in tqdm(chunk_idx):
            row = self.annotation[idx]
            _, fn, lb, caption = row.strip().split(",")
            label = " ".join(caption.split()[5:])[:-1]
            image_path = f"{self.image_dir}/{lb}/{fn}"
            img = Image.open(image_path).convert('RGB')
            text = self.get_prompt()
            data.append({
                "images": [img],
                "question": text,
                'category': "Whole dataset",
                "label": label,
                "eval_method": "hemm",
            })

        return data, []