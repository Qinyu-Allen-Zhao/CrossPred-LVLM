import os
import numpy as np
import json
from typing import Optional, Union, List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataset.base import BaseDataset

from utils.func import shell_command
from dataset.hemm.prompts.okqvqa_prompt import OKVQAPrompt


class OKVQADataset(BaseDataset):
    def __init__(self,
                download_dir="./",
                dataset_dir='okvqa/',
                annotation_file=None,
                **kwargs,
                ):
        super().__init__()
        self.download_dir = download_dir
        self.dataset_dir = os.path.join(download_dir, dataset_dir)
        self.prompt = OKVQAPrompt()
        self.load()

    def __len__(self):
        annotation_file = os.path.join(self.dataset_dir, 'mscoco_val2014_annotations.json')
        annotations = json.load(open(annotation_file, "r"))
        return len(annotations["annotations"])

    def load(self):
        if not os.path.exists(f"{self.download_dir}/okvqa"):
            shell_command(f'mkdir -p {self.download_dir}/okvqa')
        if not os.path.exists(f"{self.download_dir}/okvqa/val2014.zip"):
            shell_command(f'wget http://images.cocodataset.org/zips/val2014.zip -P {self.download_dir}/okvqa/')
        if not os.path.exists(f'{self.download_dir}/okvqa/mscoco_val2014_annotations.json.zip'):
            shell_command(f'wget https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip -P {self.download_dir}/okvqa/')
        if not os.path.exists(f'{self.download_dir}/okvqa/OpenEnded_mscoco_val2014_questions.json.zip'):
            shell_command(f'wget https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip -P {self.download_dir}/okvqa/')
        if not os.path.exists(f'{self.download_dir}/okvqa/mscoco_val2014_annotations.json'):
            shell_command(f'unzip {self.download_dir}/okvqa/mscoco_val2014_annotations.json.zip -d {self.download_dir}/okvqa/')
        if not os.path.exists(f'{self.download_dir}/okvqa/OpenEnded_mscoco_val2014_questions.json'):
            shell_command(f'unzip {self.download_dir}/okvqa/OpenEnded_mscoco_val2014_questions.json.zip -d {self.download_dir}/okvqa')
        if not os.path.exists(f'{self.download_dir}/okvqa/val2014'):
            shell_command(f'unzip {self.download_dir}/okvqa/val2014.zip -d {self.download_dir}/okvqa/')

    def get_prompt(self, question):
        return self.prompt.format_prompt(question)

    def get_data(self, chunk_idx):
        data = []

        image_dir = os.path.join(self.dataset_dir, 'val2014')
        annotation_file = os.path.join(self.dataset_dir, 'mscoco_val2014_annotations.json')
        question_file = os.path.join(self.dataset_dir, 'OpenEnded_mscoco_val2014_questions.json')
        annotations = json.load(open(annotation_file, "r"))
        questions = json.load(open(question_file, "r"))

        qid_to_q = {}
        for ques in questions["questions"]:
            qid_to_q[ques["question_id"]] = ques["question"]
        
        for idx in tqdm(chunk_idx):
            ann = annotations["annotations"][idx]
            image_path = os.path.join(image_dir, f"COCO_val2014_{ann['image_id']:0>12d}.jpg")
            img = Image.open(image_path).convert('RGB')
    
            qs = qid_to_q[ann["question_id"]]
            text = self.get_prompt(qs)
    
            ground_truth_answer = ann['answers']
            multiple_gts = []
            for gt_ans in ground_truth_answer:
                multiple_gts.append(gt_ans["answer"])
            
            multiple_gts = list(set(multiple_gts))
    
            data.append({
                "images": [img],
                "question": text,
                'category': "Whole dataset",
                "label": multiple_gts,
                "eval_method": "hemm",
            })

        return data, []
