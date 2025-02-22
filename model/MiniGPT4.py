import sys
sys.path.append("/data/qinyu/research/MiniGPT-4")
import argparse

import torch
from torch import nn
from transformers import StoppingCriteriaList

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from utils.process import process_images
from model.base import LargeMultimodalModel


class Args(argparse.Namespace):
    cfg_path = ""
    options = []
    
    def __init__(self, cfg_path='/data/qinyu/research/MiniGPT-4/eval_configs/minigpt4_llama2_eval.yaml'):
        self.cfg_path = cfg_path

    
class MiniGPT4(LargeMultimodalModel):
    def __init__(self, args):
        super(MiniGPT4, self).__init__()
        
        minigpt_args = Args(args.model_path)
        cfg = Config(minigpt_args)

        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        self.model = model_cls.from_config(model_config).to('cuda')

        conv_dict = {'pretrain_vicuna0_7b': CONV_VISION_Vicuna0,
                     'pretrain_vicuna0_13b': CONV_VISION_Vicuna0,
                     'pretrain_llama2': CONV_VISION_LLama2}

        self.CONV_VISION = conv_dict[model_config.model_type]
        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        self.vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

        self.temperature = args.temperature
        self.top_p = 0.9 #  args.top_p
        self.num_beams = args.num_beams
        
        print('Initialization Finished')
    
    def refresh_chat(self):
        stop_words_ids = [[835], [2277, 29937]]
        stop_words_ids = [torch.tensor(ids).to(self.device) for ids in stop_words_ids]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        self.chat = Chat(self.model, self.vis_processor, 
                    device=self.device, 
                    stopping_criteria=stopping_criteria)
        self.chat_state = self.CONV_VISION.copy()
        
    def forward_with_probs(self, images, prompt):
        # if len(images) > 1:
        #     image = process_images(images)
        # else:
        #     image = images[0]
        image = images[0]
        self.refresh_chat()
        
        img_list = []
        llm_message = self.chat.upload_img(image, self.chat_state, img_list)
        self.chat.encode_img(img_list)

        self.chat.ask(prompt, self.chat_state)
        llm_message, outputs = self.chat.answer(conv=self.chat_state,
                                       img_list=img_list,
                                       do_sample=False,
                                       temperature=self.temperature,
                                       top_p=self.top_p,
                                       num_beams=self.num_beams,
                                       max_new_tokens=300,
                                                
                                       return_dict_in_generate=True,
                                       output_hidden_states=True,
                                       output_scores=True
                               )
        
        logits = torch.cat(outputs['scores'], dim=0).cpu().numpy()
        probs = [nn.functional.softmax(next_token_scores, dim=-1) for next_token_scores in outputs['scores']]
        probs = torch.cat(probs).cpu().numpy()
        output_ids = outputs["sequences"][0][-len(probs):]
        output_ids = output_ids.cpu().numpy()

        hidden_states_all_layers = outputs['hidden_states'][0]
        hidden_states = hidden_states_all_layers[-1][0][[-1]]   # last layer, batch size=1, last token
        
        return llm_message, output_ids, logits, probs, hidden_states