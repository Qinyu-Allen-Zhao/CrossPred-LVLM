import sys
sys.path.append("/data/qinyu/research/cobra")

import requests
import torch

import cv2
from PIL import Image
from torch import nn
import numpy as np
from pathlib import Path

from cobra import load

from model.base import LargeMultimodalModel

class Cobra(LargeMultimodalModel):
    def __init__(self):
        super(Cobra, self).__init__()

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # In case your GPU does not support bf16
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        # Load a pretrained VLM (either local path, or ID to auto-download from the HF Hub) 
        model_id = "cobra+3b"
        self.vlm = load(model_id)
        self.vlm.to(device, dtype=dtype)

    def _basic_forward(self, image, prompt, return_dict=False):
        pil_image = Image.fromarray(image.astype(np.uint8))

        # Build prompt
        prompt_builder = self.vlm.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=prompt)
        prompt_text = prompt_builder.get_prompt()

        # Generate
        _, outputs = self.vlm.generate(
            pil_image,
            prompt_text,
            cg=True,
            do_sample=True,
            temperature=0.4,
            max_new_tokens=512,
            return_dict_in_generate=return_dict,
            output_scores=return_dict,
        )

        return outputs

    def forward_with_probs(self, image, prompt):
        pil_image = Image.fromarray(image.astype(np.uint8))

        # Build prompt
        prompt_builder = self.vlm.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=prompt)
        prompt_text = prompt_builder.get_prompt()

        # Generate
        generated_text, outputs = self.vlm.generate(
            pil_image,
            prompt_text,
            cg=True,
            do_sample=True,
            temperature=0.4,
            max_new_tokens=10,
            return_dict_in_generate=True,
            output_scores=True,
        )
        
        logits = torch.cat(outputs['scores'], dim=0).float().cpu().numpy()
        probs = [nn.functional.softmax(next_token_scores, dim=-1) for next_token_scores in outputs['scores']]
        probs = torch.cat(probs).float().cpu().numpy()

        output_ids = outputs["sequences"][0][-len(probs):]
        output_ids = output_ids.cpu().numpy()

        return generated_text, output_ids, logits, probs
    

    def forward_pairs(self, image, prompt, labels):
        pil_image = Image.fromarray(image.astype(np.uint8))

        hidden_states, responses = [], []
        for label in labels:
            # Build prompt
            prompt_builder = self.vlm.get_prompt_builder()
            prompt_builder.add_turn(role="human", message=prompt)
            prompt_builder.add_turn(role="gpt", message=label)

            prompt_text = prompt_builder.get_prompt()

            # Generate
            generated_text, outputs = self.vlm.generate(
                pil_image,
                prompt_text,
                cg=True,
                do_sample=True,
                temperature=0.4,
                max_new_tokens=512,
                return_dict_in_generate=True,
                output_scores=True,
            )

            responses.append(generated_text)
            hs = outputs['hidden_states'][0][0][-1].float().detach().squeeze().cpu().numpy()
            hidden_states.append(hs)
        
        return responses, hidden_states