import torch
import torch.nn as nn
from model.base import LargeMultimodalModel
from transformers import FuyuProcessor, FuyuForCausalLM


class Fuyu(LargeMultimodalModel):
    def __init__(self, args):
        super().__init__()
        # An example of model_path: "adept/fuyu-8b"
        self.processor = FuyuProcessor.from_pretrained(args.model_path)
        self.model = FuyuForCausalLM.from_pretrained(args.model_path, device_map=self.device)

    def forward_with_probs(self, images, prompt):
        raw_image = images[0]

        inputs = self.processor(text=prompt, images=raw_image, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=100,
            pad_token_id=self.processor.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_scores=True
        )

        logits = torch.cat(outputs['scores'], dim=0).cpu().numpy()
        probs = [nn.functional.softmax(next_token_scores, dim=-1) for next_token_scores in outputs['scores']]
        probs = torch.cat(probs).cpu().numpy()
        output_ids = outputs["sequences"][0][-len(probs):]

        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        output_ids = output_ids.cpu().numpy()

        hidden_states_all_layers = outputs['hidden_states'][0]
        hidden_states = hidden_states_all_layers[-1][0][[-1]]   # last layer, batch size=1, last token

        return response, output_ids, logits, probs, hidden_states

