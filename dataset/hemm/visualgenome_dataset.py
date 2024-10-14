import os
import json
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm, trange
import pandas as pd
import requests

from dataset.base import BaseDataset

from utils.func import shell_command
from dataset.hemm.prompts.visualgenome_prompt import visualgenomeprompt


class VisualGenome(BaseDataset):
    def __init__(self,
                download_dir="./",
                dataset_dir=None,
                annotation_file="visual_genome/question_answers.json",
                **kwargs,
                ):
        super().__init__()
        self.download_dir = download_dir
        self.prompt = visualgenomeprompt()
        self.questions_json_path = os.path.join(download_dir, annotation_file)
        self.load()
    
    def load(self):
        if not os.path.exists(f"{self.download_dir}/visual_genome"):
            shell_command(f"mkdir -p {self.download_dir}/visual_genome")
            shell_command(f"wget https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/question_answers.json.zip -P {self.download_dir}/visual_genome/")
            shell_command(f"unzip {self.download_dir}/visual_genome/question_answers.json.zip -d {self.download_dir}/visual_genome/")

        data = []
        f = open(self.questions_json_path)
        data_vqa = json.load(f)

        for idx in range(100):  # Use the first 100 images
            temp_dict = data_vqa[idx]
            img_id = temp_dict['id']
            qas = temp_dict['qas']
            if idx == 0:
                url = f"https://cs.stanford.edu/people/rak248/VG_100K_2/{img_id}.jpg"
            else:
                url = f"https://cs.stanford.edu/people/rak248/VG_100K/{img_id}.jpg"

            for j in range(len(qas)):
                question = qas[j]['question']
                question_pmt = self.get_prompt(question)
                
                data.append({
                "url": url,
                "question": question_pmt,
                "label": qas[j]['answer'],
            })

        self.data = data

    def get_prompt(self, text):
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text
    
    def __len__(self):
        return len(self.data)

    def get_data(self, chunk_idx):
        data =[]
        for idx in tqdm(chunk_idx):
            ins = self.data[idx]
            url = ins['url']
            image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
            image_b = image.resize((640,480))

            data.append({
                "images": [image_b],
                "question": ins['question'],
                'category': "Whole dataset",
                "label": ins['label'],
                "eval_method": "hemm",
            })

        return data, []
