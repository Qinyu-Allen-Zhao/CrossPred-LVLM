import sys
sys.path.append("/data/qinyu/research/mPLUG-Owl/mPLUG-Owl2")

import torch
from torch import nn
from PIL import Image
from transformers import TextStreamer

from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from model.base import LargeMultimodalModel

class mPLUG_Owl(LargeMultimodalModel):
    def __init__(self, args):
        super(mPLUG_Owl, self).__init__()
        model_path = args.model_path
        model_name = get_model_name_from_path(model_path)
        
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, 
                                                                                                   load_4bit=False, device="cuda", 
                                                                                                   fp16=True
                                                                                                   )
        self.temperature = args.temperature
        self.top_p = 0.9 # args.top_p
        self.num_beams = args.num_beams
        
    def _basic_forward(self, image, prompt, return_dict=False):
        max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
        image = image.resize((max_edge, max_edge))
        
        image_tensor = process_images([image], self.image_processor)
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        conv = conv_templates["mplug_owl2"].copy()

        inp = DEFAULT_IMAGE_TOKEN + prompt
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                images=image_tensor,
                streamer=streamer,
                use_cache=True,
                max_new_tokens=512,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=[stopping_criteria],
                
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                
                return_dict_in_generate=return_dict,
                output_hidden_states=return_dict,
                output_scores=return_dict
            )

        return outputs
    
    
    def forward_with_probs(self, images, prompt):
        image = images[0]
        outputs = self._basic_forward(image, prompt, return_dict=True)
        
        logits = torch.cat(outputs['scores'], dim=0).cpu().numpy()
        probs = [nn.functional.softmax(next_token_scores, dim=-1) for next_token_scores in outputs['scores']]
        probs = torch.cat(probs).cpu().numpy()
        output_ids = outputs["sequences"][0][-len(probs):]
    
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        output_ids = output_ids.cpu().numpy()

        hidden_states_all_layers = outputs['hidden_states'][0]
        hidden_states = hidden_states_all_layers[-1][0][[-1]]   # last layer, batch size=1, last token
        
        return response, output_ids, logits, probs, hidden_states