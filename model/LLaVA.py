import torch
from torch import nn
import numpy as np
from io import BytesIO
from baukit import Trace, TraceDict
from transformers import TextStreamer
from transformers.generation import BeamSearchDecoderOnlyOutput

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from model.base import LargeMultimodalModel

class LLaVA(LargeMultimodalModel):
    def __init__(self, args):
        super(LLaVA, self).__init__()
        load_8bit = False
        load_4bit = False
        
        # Load Model
        disable_torch_init()

        model_name = get_model_name_from_path(args.model_path)
        if "finetune-lora" in args.model_path:
            model_base = "liuhaotian/llava-v1.5-7b"
        elif "lora" in args.model_path:
            model_base = "lmsys/vicuna-7b-v1.5"
        else:
            model_base = None
        print(model_base)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(args.model_path, model_base, model_name, load_8bit, load_4bit)
        print(model_name)

        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
        self.conv_mode = conv_mode
        
        self.temperature = args.temperature
        self.top_p = None
        self.num_beams = args.num_beams

    
    def refresh_chat(self):
        self.conv = conv_templates[self.conv_mode].copy()
        self.roles = self.conv.roles
    
    def _basic_forward(self, images, prompt, return_dict=False):
        self.refresh_chat()
        
        image_tensor = [self.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
                        for image in images]
        image_tensor = [_image.unsqueeze(0).to(dtype=torch.float16, device=self.device) for _image in image_tensor]
        image_tensor = torch.cat(image_tensor, dim=0)

        image_sizes = [images[idx].size for idx in range(len(images))]

        # message
        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        self.conv.append_message(self.conv.roles[0], inp)
        self.conv.append_message(self.conv.roles[1], None)

        conv_prompt = self.conv.get_prompt()

        input_ids = tokenizer_image_token(conv_prompt, self.tokenizer, 
                                          IMAGE_TOKEN_INDEX, 
                                          return_tensors='pt').unsqueeze(0).cuda()
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                images=image_tensor,
                pad_token_id=pad_token_ids,
                image_sizes=image_sizes,
                
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                
                max_new_tokens=2048,
                streamer=streamer,
                use_cache=True,
                
                return_dict_in_generate=return_dict,
                output_hidden_states=return_dict,
                output_scores=return_dict)
            
        return outputs

    
    def forward_with_probs(self, images, prompt):
        outputs = self._basic_forward(images, prompt, return_dict=True)
        
        logits = torch.cat(outputs['scores'], dim=0).cpu().numpy()
        probs = [nn.functional.softmax(next_token_scores, dim=-1) for next_token_scores in outputs['scores']]
        probs = torch.cat(probs).cpu().numpy()
        output_ids = outputs["sequences"][0][-len(probs):]

        response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        output_ids = output_ids.cpu().numpy()

        hidden_states_all_layers = outputs['hidden_states'][0]
        hidden_states = hidden_states_all_layers[-1][0][[-1]]   # last layer, batch size=1, last token
        
        return response, output_ids, logits, probs, hidden_states
    