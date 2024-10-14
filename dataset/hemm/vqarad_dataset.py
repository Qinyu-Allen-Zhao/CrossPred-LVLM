import os
import json
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from dataset.base import BaseDataset

from utils.func import shell_command
from dataset.hemm.prompts.vqarad_prompt import VQARADPrompt


class VQARADDataset(BaseDataset):
    def __init__(self,
                download_dir="./",
                dataset_dir=None,
                annotation_file=None,
                **kwargs,
                ):
        super().__init__()
        self.dataset_dir = download_dir
        self.prompt = VQARADPrompt()
        self.load()

    def load(self):
        self.dataset = load_dataset("flaviagiammarino/vqa-rad", cache_dir=self.dataset_dir)
        self.dataset = self.dataset['test']

    def __len__(self):
        return len(self.dataset)

    def get_prompt(self, text):
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text

    def get_data(self, chunk_idx):
        data = []
        for idx in tqdm(chunk_idx):
            data_dict = self.dataset[idx]
            image = data_dict['image']
            img = image.convert("RGB")

            ground_truth_answer = data_dict['answer']

            question = data_dict['question']
            text = self.get_prompt(question)

            data.append({
                    "images": [img],
                    "question": text,
                    'category': "Whole dataset",
                    "label": ground_truth_answer,
                    "eval_method": "hemm",
                })

        return data, []