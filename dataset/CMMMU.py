import re
import ast

from tqdm import tqdm
import numpy as np
from datasets import load_dataset

from dataset.base import BaseDataset


PROMPT = {
    "task_instructions": [
        "请回答以下多项选择题，并选出正确选项。这些题目可能包括单选和多选题型。如果所提供的信息不足以确定一个明确的答案，那么请根据可用的数据和你的判断来选择最可能正确的选项。",
        "请回答以下判断题，并根据题目描述和所给的信息来判断问题中陈述的对错。如果信息不完整或不足以作出绝对判断，请运用你的逻辑推理和现有信息来做出最可能的判断。",
        "请回答以下填空题，并根据题目的要求和所提供的信息来给出最恰当的答案。如果信息不足以确切回答，那么请依据现有的数据和你的推理能力来填写最合理的答案。",
    ],
    "multi_choice_example_format": ["问题：{}\n选项：\n{}\n正确答案：\n"],
    "T/F_example_format": ["问题：{}\n正确答案：\n"],
    "short_ans_example_format": ["问题：{}\n正确答案：\n"],
}

TYPE_MAP = {
    "选择": "multi_choice",
    "判断": "true_false_cn",
    "填空": "fill_blank_cn",
}


class CMMMU(BaseDataset):
    def __init__(self):
        super(CMMMU, self).__init__()
        self.data = load_dataset("lmms-lab/CMMMU")['val']
         
    def __len__(self):
        return len(self.data)

    def get_data(self, chunk_idx):
        data = []
        for idx in tqdm(chunk_idx):
            ins = self.data[idx]
            prompt = self.construct_prompt(ins)
            data.append({
                "images": self.create_image_list(ins, prompt),
                "question": prompt,
                'category': ins['category'],
                "label": ins['answer'],
                "options": [ins["option1"], ins["option2"], ins["option3"], ins["option4"]],
                "eval_method": TYPE_MAP[ins["type"]],
            })

        return data, ['category']
    
    def construct_prompt(self, ins):
        question = ins["question"]
        task_instructions = PROMPT["task_instructions"]

        if ins["type"] == "选择":
            formatted_options = ""
            start_chr = "A"
            for i in range(1, 5):
                formatted_options += f"({start_chr}) {ins[f'option{i}']}\n"
                start_chr = chr(ord(start_chr) + 1)

            current_example_template = PROMPT["multi_choice_example_format"][0]
            current_example = current_example_template.format(question, formatted_options)
            final_input_prompt = task_instructions[0] + "\n\n" + current_example

        elif ins["type"] == "判断":
            current_example_template = PROMPT["T/F_example_format"][0]
            current_example = current_example_template.format(question)
            final_input_prompt = task_instructions[1] + "\n\n" + current_example

        else:  # For fill in the blanks questions.
            current_example_template = PROMPT["short_ans_example_format"][0]
            current_example = current_example_template.format(question)
            final_input_prompt = task_instructions[2] + "\n\n" + current_example

        for i in range(1, 6):
            final_input_prompt = final_input_prompt.replace(f'<img="{ins[f"image_{i}_filename"]}">', f"<图片 {i}>")

        return final_input_prompt

    def create_image_list(self, ins, prompt):
        image_tokens = re.findall(r"<图片 \d+>", prompt)
        # Remove <> and  swap space as _
        image_tokens = [image_token.strip("<>").replace(" ", "_").replace("图片", "image") for image_token in image_tokens]
        images = [ins[image_token].convert("RGB") for image_token in image_tokens]
        return images
    
    def sample(self, num_samples):
        data_id = np.arange(len(self))

        if num_samples is not None:
            np.random.seed(0)
            categories = np.array(self.data['category'])
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