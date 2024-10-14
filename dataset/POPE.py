import os
import json

import numpy as np
from PIL import Image

from dataset.base import BaseDataset
from utils.func import read_jsonl

class POPEDataset(BaseDataset):
    def __init__(self, data_root="/data/coco/"):
        super(POPEDataset, self).__init__()
        self.ann_path = "./data/pope/"
        self.img_root = os.path.join(data_root, f"val2014")

        data = []
        cats = ["adversarial", "popular", "random"]
        for category in cats:
            ann = read_jsonl(os.path.join(self.ann_path, f"coco_pope_{category}.json"))
            data_cat = [
                {
                    "images": [Image.open(os.path.join(self.img_root, ins['image'])).convert("RGB")],
                    "question": f"{ins['text']}\nAnswer the question using a single word or phrase.",
                    "label": ins['label'],
                    "question_id": ins["question_id"],
                    "category": category,
                    "eval_method": "true_false_en",
                }
                for ins in ann
            ]

            data += data_cat
        
        self.data = data

    def __len__(self):
        return len(self.data)
         
    def get_data(self, chunk_id):
        data = [self.data[idx] for idx in chunk_id]
            
        return data, ["question_id", "category"]