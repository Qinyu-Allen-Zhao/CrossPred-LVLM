from typing import Optional, Union

import torch
from torch import nn
from PIL import Image
import re

from model.base import LargeMultimodalModel
from tqdm import tqdm
from transformers import Blip2ForConditionalGeneration, Blip2Processor


class BLIP2(LargeMultimodalModel):
    def __init__(self, args):
        super().__init__()
        self.model = Blip2ForConditionalGeneration.from_pretrained(args.model_path, device_map=self.device)
        self.processor = Blip2Processor.from_pretrained(args.model_path)
        self.tokenizer = self.processor.tokenizer
        self.model_path = args.model_path
        self.encoder_decoder_arch = True if "flan" in args.model_path else False

        self.model.eval()
        self.model.tie_weights()

        self.model.to(self.device)

        self.temperature = args.temperature
        self.top_p = None
        self.num_beams = args.num_beams

    def forward_with_probs(self, images, prompt):
        if "opt" in self.model_path:
            prompt = f"Question: {prompt} Answer:"

        inputs = self.processor(images=images[0], text=prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            do_sample=True if self.temperature > 0 else False,
            temperature=self.temperature,
            top_p=self.top_p,
            num_beams=self.num_beams,
            max_new_tokens=512,

            return_dict_in_generate=True,
            output_hidden_states=True,
            output_scores=True
        )

        logits = torch.cat(outputs['scores'], dim=0).cpu().numpy()
        probs = [nn.functional.softmax(next_token_scores, dim=-1) for next_token_scores in outputs['scores']]
        probs = torch.cat(probs).cpu().numpy()
        output_ids = outputs["sequences"][0][-len(probs):]
    
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        output_ids = output_ids.cpu().numpy()

        hidden_states_all_layers = outputs['decoder_hidden_states'][0] if self.encoder_decoder_arch else outputs['hidden_states'][0]
        hidden_states = hidden_states_all_layers[-1][0][[-1]]   # last layer, batch size=1, last token

        return response, output_ids, logits, probs, hidden_states

