import os
import json
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from dataset.base import BaseDataset

from utils.func import shell_command
from dataset.hemm.prompts.gqa_prompt import GQAPrompt


class GQADataset(BaseDataset):
    def __init__(self,
                  download_dir="./",
                  dataset_dir=None,
                  annotation_file=None,
                  **kwargs,
                  ):
        super().__init__()
        self.dataset_dir = download_dir 
        self.prompt = GQAPrompt()
        self.load()

    def __len__(self):       
       question_file = json.load(open(os.path.join(f'{self.dataset_dir}/gqa_questions', 'testdev_all_questions.json'), 'r'))
       return len(question_file)

    def load(self):
      if not os.path.exists(f'{self.dataset_dir}/sceneGraphs.zip'):
        shell_command(f'wget https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip -P {self.dataset_dir}')
      if not os.path.exists(f'{self.dataset_dir}/questions1.2.zip'):
        shell_command(f'wget https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip -P {self.dataset_dir}')
      if not os.path.exists(f'{self.dataset_dir}/images.zip'):
        shell_command(f'wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip -P {self.dataset_dir}')
      if not os.path.exists(f'{self.dataset_dir}/gqa_images'):
        os.makedirs(f'{self.dataset_dir}/gqa_images/')
        shell_command(f'unzip {self.dataset_dir}/images.zip -d {self.dataset_dir}/gqa_images')
      if not os.path.exists(f'{self.dataset_dir}/gqa_scene_graphs'):
        os.makedirs(f'{self.dataset_dir}/gqa_scene_graphs/')
        shell_command(f'unzip {self.dataset_dir}/sceneGraphs.zip -d {self.dataset_dir}/gqa_scene_graphs')
      if not os.path.exists(f'{self.dataset_dir}/gqa_questions'):
        os.makedirs(f'{self.dataset_dir}/gqa_questions/')
        shell_command(f'unzip {self.dataset_dir}/questions1.2.zip -d {self.dataset_dir}/gqa_questions')

    def get_prompt(self, text):
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text

    def get_data(self, chunk_idx):
        image_dir = f'{self.dataset_dir}/gqa_images/images'
        question_file = json.load(open(os.path.join(f'{self.dataset_dir}/gqa_questions', 'testdev_all_questions.json'), 'r'))

        indexes = list(question_file.keys())
        indexes = [indexes[i] for i in chunk_idx]

        data = []
        for idx in tqdm(indexes):
            question = question_file[idx]['question']
            image_path = os.path.join(image_dir, question_file[idx]['imageId']+'.jpg')
            img = Image.open(image_path).convert("RGB")

            label = question_file[idx]['answer']
            text = self.get_prompt(question)

            data.append({
                "images": [img],
                "question": text,
                'category': "Whole dataset",
                "label": label,
                "eval_method": "hemm",
            })

        return data, []
    