import os
import json
from PIL import Image
import requests
import torch
import subprocess
from tqdm import tqdm
import pandas as pd

from dataset.base import BaseDataset

from utils.func import shell_command
from dataset.hemm.prompts.face_emotion_prompt import FaceEmotionPrompt


class FaceEmotionDataset(BaseDataset):
    def __init__(self,
                download_dir="./",
                dataset_dir='face_emotion',
                annotation_file=None,
                **kwargs,
                ):
        super().__init__()
        self.download_dir = download_dir
        self.data_path = os.path.join(download_dir, "face_emotion")
        self.kaggle_api_path = kwargs["kaggle_api_path"]
        self.prompt = FaceEmotionPrompt()
        self.choices = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.load()

    def get_prompt(self) -> str:
        prompt_text = self.prompt.format_prompt()
        return prompt_text
    
    def __len__(self,):
        data_dict = {}
        for fol in os.listdir(self.data_path):
            for img in os.listdir(os.path.join(self.data_path, fol)):
                data_dict[img] = fol

        return len(data_dict)
        
    def load(self):
        os.environ['KAGGLE_CONFIG_DIR'] = self.kaggle_api_path
        if not os.path.exists(f'{self.download_dir}/fer2013.zip'):
            shell_command(f'kaggle datasets download -d msambare/fer2013 -p {self.download_dir}')
        if not os.path.exists(self.data_path):
            shell_command(f'unzip {self.download_dir}/fer2013.zip -d {self.download_dir}')
            shell_command(f'mv {self.download_dir}/test {self.download_dir}/face_emotion')

    def get_data(self, chunk_idx):
        data_dict = {}
        for fol in os.listdir(self.data_path):
            for img in os.listdir(os.path.join(self.data_path, fol)):
                data_dict[img] = fol
        
        data_dict_list = sorted(list(data_dict.items()))
        
        data = []
        for idx in tqdm(chunk_idx):
            row = data_dict_list[idx]
            img, label = row
            image_path = os.path.join(self.data_path, label, img)
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
