import re
import ast

from tqdm import tqdm
import numpy as np
from datasets import load_dataset

from dataset.base import BaseDataset


class MMMU(BaseDataset):
    def __init__(self):
        super(MMMU, self).__init__()
        self.data = load_dataset("lmms-lab/MMMU")['validation']
         
    def __len__(self):
        return len(self.data)

    def get_data(self, chunk_idx):
        data = []
        for idx in tqdm(chunk_idx):
            ins = self.data[idx]
            data.append({
                "images": self.create_image_list(ins),
                "question": self.create_options_prompt(ins),
                'category': self.extract_subset_name(ins['id']),
                "label": ins['answer'],
                "options": ins['options'],
                "eval_method": "multi_choice" if ins['question_type'] == 'multiple-choice' else "fill_blank_en",
            })

        return data, ['category']
    
    def create_options_prompt(self, ins):
        question = f"{ins['question']}"
        for image_id in [1, 2, 3, 4, 5, 6, 7]:
            question = question.replace(f"<image {image_id}>", "")
        question = question.strip() + "\n"

        options_prompt = [question]
        options = ast.literal_eval(ins['options'])
        for i, opt in enumerate(options):
            options_prompt.append(f"{chr(65+i)}. {opt}\n")
        
        if ins['question_type'] == 'multiple-choice':
            options_prompt.append("Answer with the option's letter from the given choices directly.")
        else:
            options_prompt.append("Answer the question using a single word or phrase.")

        return "".join(options_prompt)
    
    def create_image_list(self, ins):
        images = []
        for image_id in [1, 2, 3, 4, 5, 6, 7]:
            image = ins[f"image_{image_id}"]
            if image is not None:
                images.append(image.convert("RGB"))
        return images
    
    def extract_subset_name(self, input_string):
        # Define a regex pattern to match "validation_" at the beginning and "_<number>" at the end
        split = input_string.split("_")[0]
        pattern = re.compile(rf"^{split}_(.+?)_\d+$")
        match = pattern.search(input_string)
        if match:
            return match.group(1)
        else:
            raise ValueError(f'No match found in "{input_string}"')

    def sample(self, num_samples):
        data_id = np.arange(len(self))

        if num_samples is not None:
            np.random.seed(0)
            categories = np.array([self.extract_subset_name(idx) for idx in self.data['id']])
            sampled_dict = []
    
            for category in np.unique(categories):
                idx = data_id[categories == category]
                if len(idx) < 30:
                    continue
                else:
                    # print(f"Category {category} has {len(idx)} samples")
                    sampled_dict.append(np.random.choice(idx, min(len(idx), num_samples), replace=False))
            
            return np.concatenate(sampled_dict)


        return data_id