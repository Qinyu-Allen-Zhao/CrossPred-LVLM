import sys
sys.path.append("/data/qinyu/research/LLaMA-Adapter/llama_adapter_v2_multimodal7b")

import torch
from PIL import Image

import llama
from model.base import LargeMultimodalModel

class LLaMA_Adapter(LargeMultimodalModel):
    def __init__(self, args):
        super(LLaMA_Adapter, self).__init__()
        llama_dir = "/data/qinyu/model/llama-v1/"

        # choose from BIAS-7B, LORA-BIAS-7B, LORA-BIAS-7B-v21
        self.model, self.preprocess = llama.load(args.model_path, llama_dir, llama_type="7B", device="cuda")
        self.model.eval()
        
        self.temperature = args.temperature
        self.top_p = None
        
    def forward_with_probs(self, images, prompt):
        prompt = llama.format_prompt(prompt)
        image = images[0]
        img = self.preprocess(image).unsqueeze(0).to("cuda")

        response, output_ids, logits, all_hidden_states = self.model.generate(img, [prompt],
                temperature=self.temperature,
                top_p=self.top_p)
        
        hidden_states = all_hidden_states[0][0][[-1]]

        return response[0], output_ids, logits, None, hidden_states
    