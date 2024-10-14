import os
import json
from typing import Optional, Union, List
from PIL import Image
import requests
import torch
from datasets import load_dataset
import subprocess
from tqdm import tqdm
import pickle

from dataset.base import BaseDataset

from utils.func import shell_command
from dataset.hemm.prompts.pathvqa_prompt import PathVQAPrompt


class PathVQADataset(BaseDataset):
    def __init__(self, download_dir="./", dataset_dir=None, annotation_file=None, **kwargs):
        self.download_dir = download_dir
        self.load()
        self.prompt = PathVQAPrompt()

    def get_prompt(self, text) -> str:
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text

    def load(self):
        if not os.path.exists(f'{self.download_dir}/Backup'):
            shell_command(f'gdown --no-check-certificate --folder https://drive.google.com/drive/folders/1G2C2_FUCyYQKCkSeCRRiTTsLDvOAjFj5 -O {self.download_dir}')
        if not os.path.exists(f'{self.download_dir}/pathvqa_images'):
            shell_command(f'unzip {self.download_dir}/Backup/pvqa.zip -d {self.download_dir}/pathvqa_images/')
    
    def __len__(self):
        annotation_path = os.path.join(self.download_dir, 'pathvqa_images','pvqa','qas','test','test_qa.pkl')
        annotation_file = pickle.load(open(annotation_path, 'rb'))

        return len(annotation_file)
        
    def get_data(self, chunk_idx):
        data = []
        images_dir = os.path.join(self.download_dir, 'pathvqa_images','pvqa','images','test')
        annotation_path = os.path.join(self.download_dir, 'pathvqa_images','pvqa','qas','test','test_qa.pkl')
        annotation_file = pickle.load(open(annotation_path, 'rb'))
        
        for idx in tqdm(chunk_idx):
            data_dict = annotation_file[idx]
            image_path = os.path.join(images_dir, data_dict['image'] + '.jpg')
            question = data_dict['question']
            ground_truth_answer = data_dict["answer"]
            img = Image.open(image_path).convert('RGB')

            text = self.get_prompt(question)
            data.append({
                "images": [img],
                "question": text,
                'category': "Whole dataset",
                "label": ground_truth_answer,
                "eval_method": "hemm",
            })

        return data, []
