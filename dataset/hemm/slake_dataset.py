import os
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm
import json

from dataset.base import BaseDataset

from utils.func import shell_command
from dataset.hemm.prompts.slake_prompt import SlakePrompt

class SlakeDataset(BaseDataset):
    def __init__(self,
                download_dir="./",
                dataset_dir='Slake1.0/imgs',
                annotation_file="Slake1.0/test.json",
                **kwargs,
                ):
        super().__init__()
        self.download_dir = download_dir
        self.load()
        self.image_dir = os.path.join(download_dir, dataset_dir)
        self.prompt = SlakePrompt()
        all_annotation = json.load(open(os.path.join(download_dir, annotation_file)))
        self.annotation = []
        for ann in all_annotation:
            if ann["q_lang"] == "en" and ann["answer_type"] == "CLOSED":
                self.annotation.append(ann)

    def load(self):
        if not os.path.exists(f"{self.download_dir}/Slake1.0"):
            shell_command(f"mkdir -p {self.download_dir}/Slake1.0")
            shell_command(f"wget https://huggingface.co/datasets/BoKelvin/SLAKE/raw/main/test.json -P {self.download_dir}/Slake1.0/")
            shell_command(f"wget https://huggingface.co/datasets/BoKelvin/SLAKE/resolve/main/imgs.zip -P {self.download_dir}/Slake1.0/")
            shell_command(f"unzip {self.download_dir}/Slake1.0/imgs.zip -d {self.download_dir}/Slake1.0/")

    def __len__(self):
        return len(self.annotation)

    def get_prompt(self, text):
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text

    def get_data(self, chunk_idx):
        data = []
        for idx in tqdm(chunk_idx):
            row = self.annotation[idx]
            label = row["answer"]
            question = row["question"]
            image_path = f"{self.image_dir}/{row['img_name']}"
            img = Image.open(image_path).convert("RGB")
            text = self.get_prompt(question)

            data.append({
                "images": [img],
                "question": text,
                'category': "Whole dataset",
                "label": label,
                "eval_method": "hemm",
            })

        return data, []
    