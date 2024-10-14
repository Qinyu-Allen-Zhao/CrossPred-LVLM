import torch
import torch.nn as nn

from model.base import LargeMultimodalModel
from transformers import AutoProcessor, AutoModelForVision2Seq

def ref_text(text):
    sents = text.split("\n")
    sents = [sent.strip() for sent in sents]
    return " ".join(sents).strip()

class Kosmos2(LargeMultimodalModel):
    def __init__(self, args):
        super().__init__()
        self.model = AutoModelForVision2Seq.from_pretrained(args.model_path, trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    def forward_with_probs(self, images, prompt):
        raw_image = images[0]
        prompt = "<grounding> " + prompt

        inputs = self.processor(text=prompt, images=raw_image, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
                        pixel_values=inputs["pixel_values"],
                        input_ids=inputs["input_ids"][:, :-1],
                        attention_mask=inputs["attention_mask"][:, :-1],
                        img_features=None,
                        img_attn_mask=inputs["img_attn_mask"][:, :-1],
                        use_cache=True,
                        max_new_tokens=500,

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

