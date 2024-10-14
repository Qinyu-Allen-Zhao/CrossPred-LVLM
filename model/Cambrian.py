import sys
sys.path.append("/data/qinyu/research/cambrian")
import argparse
import os
import json
import random
import re

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import shortuuid
from PIL import Image
import math

from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.conversation import conv_templates, SeparatorStyle
from cambrian.model.builder import load_pretrained_model
from cambrian.utils import disable_torch_init
from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from model.base import LargeMultimodalModel

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


CONV_MODE_MAP = {
    "cambrian-phi3-3b": "phi3",
    "cambrian-8b": "llama_3",
    "cambrian-34b": "chatml_direct",
    "cambrian-13b": "vicuna_v1"
}


class Cambrian(LargeMultimodalModel):
    def __init__(self, args):
        super().__init__()
        model_name = get_model_name_from_path(args.model_path)
        print(model_name)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(args.model_path, None, model_name)

        self.temperature = args.temperature
        self.top_p = 0.9 #  args.top_p
        self.num_beams = args.num_beams
        
        self.conv_mode = CONV_MODE_MAP[model_name]
        print('Initialization Finished')

    def process(self, images, prompt, model_config):
        qs = prompt

        if model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        conv_prompt = conv.get_prompt()
        
        # image_tensor = [self.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
        #                     for image in images]
        # image_tensor = [_image.unsqueeze(0).to(dtype=torch.float16, device=self.device) for _image in image_tensor]
        # image_tensor = torch.cat(image_tensor, dim=0)
        image_sizes = [images[idx].size for idx in range(len(images))]


        image = images[0]
        image_sizes = [image.size]
        image_tensor = process_images([image], self.image_processor, model_config)

        input_ids = tokenizer_image_token(conv_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        return input_ids, image_tensor, image_sizes
    
    def forward_with_probs(self, images, prompt):
        input_ids, image_tensor, image_sizes = self.process(images, prompt, self.model.config)
        input_ids = input_ids.to(device='cuda', non_blocking=True)

        pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        attention_masks = input_ids.ne(pad_token_ids).to(self.device)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_masks,
                pad_token_id=pad_token_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=512,
                use_cache=True,

                return_dict_in_generate=True,
                output_hidden_states=True,
                output_scores=True)

        logits = torch.cat(outputs['scores'], dim=0).cpu().numpy()
        probs = [nn.functional.softmax(next_token_scores, dim=-1) for next_token_scores in outputs['scores']]
        probs = torch.cat(probs).cpu().numpy()
        output_ids = outputs["sequences"][0][-len(probs):]
    
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        output_ids = output_ids.cpu().numpy()

        hidden_states_all_layers = outputs['hidden_states'][0]
        hidden_states = hidden_states_all_layers[-1][0][[-1]]   # last layer, batch size=1, last token
        
        return response, output_ids, logits, probs, hidden_states