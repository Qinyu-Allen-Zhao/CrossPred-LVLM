import torch
from prismatic import load

from model.base import LargeMultimodalModel


class PrismaticVLM(LargeMultimodalModel):
    def __init__(self, args):
        super().__init__()
        # Load a pretrained VLM (either local path, or ID to auto-download from the HF Hub) 
        self.vlm = load(args.model_path)
        self.vlm.to(self.device, dtype=torch.bfloat16)
        self.eos_token_id = self.vlm.llm_backbone.tokenizer("<s></s>", truncation=True, return_tensors="pt").input_ids.cuda()[0][[-1]]

    def forward_with_probs(self, images, prompt):
        image = images[0]

        # Build prompt
        prompt_builder = self.vlm.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=prompt)
        prompt_text = prompt_builder.get_prompt()

        # Generate!
        
        generated_text = self.vlm.generate(
            image,
            prompt_text,
            do_sample=False,
            max_new_tokens=512,
            min_length=1,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.eos_token_id
        )
        generated_text = generated_text.replace("</s>", "")

        return generated_text, None, None, None, None
    