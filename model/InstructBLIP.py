import torch
from torch import nn
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

from utils.process import process_images
from model.base import LargeMultimodalModel


class InstructBLIP(LargeMultimodalModel):
    """
    InstructBLIP Model
    """

    def __init__(self, args):
        super().__init__()
        self.model = InstructBlipForConditionalGeneration.from_pretrained(args.model_path, device_map=self.device)
        self.image_processor = InstructBlipProcessor.from_pretrained(args.model_path)
        self.tokenizer = self.image_processor.tokenizer
        self.config = self.model.config

        self.encoder_decoder_arch = True if "flan" in args.model_path else False

        self.model.eval()
        self.model.tie_weights()

        self.model.to(self.device)

        self.temperature = args.temperature
        self.top_p = None
        self.num_beams = args.num_beams

    def tok_encode(self, string, left_truncate_len=None, add_special_tokens=None):
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def forward_with_probs(self, images, prompt):
        context = prompt

        # The transformer implementation can't handle multi images for blip
        # Concat it into one image
        # if len(images) > 1:
        #     images = [process_images(images)]

        images = [images[0]]
        inputs = self.image_processor(images=images, text=context, return_tensors="pt", truncation=True).to(self.device)

        outputs = self.model.generate(
            **inputs,
            do_sample=True if self.temperature > 0 else False,
            temperature=self.temperature,
            top_p=self.top_p,
            num_beams=self.num_beams,
            max_new_tokens=2048,

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
