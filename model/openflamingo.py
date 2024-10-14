import abc
import torch
from typing import Optional, Union, List
from PIL import Image
from huggingface_hub import hf_hub_download
import torch
import re
import torch.nn as nn
from tqdm import tqdm

from model.base import LargeMultimodalModel
from model.open_flamingo.open_flamingo import create_model_and_transforms
from model.open_flamingo.open_flamingo.eval.utils import unwrap_model, get_autocast, get_cast_dtype

def ref_text(text):
    sents = text.split("\n")
    sents = [sent.strip() for sent in sents]
    return " ".join(sents).strip()

class OpenFlamingo(LargeMultimodalModel):
    def __init__(self, args):
        super(OpenFlamingo, self).__init__()
        
        encoder_path, x_atten_int = {
            "openflamingo/OpenFlamingo-3B-vitl-mpt1b": ("anas-awadalla/mpt-1b-redpajama-200b", 1),
            "openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct": ("anas-awadalla/mpt-1b-redpajama-200b-dolly", 1),
            "openflamingo/OpenFlamingo-4B-vitl-rpj3b": ("togethercomputer/RedPajama-INCITE-Base-3B-v1", 2),
            "openflamingo/OpenFlamingo-4B-vitl-rpj3b-langinstruct": ("togethercomputer/RedPajama-INCITE-Instruct-3B-v1", 2),
            "openflamingo/OpenFlamingo-9B-vitl-mpt7b": ("anas-awadalla/mpt-7b", 4),
        }[args.model_path]
        
        self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path=encoder_path,
            tokenizer_path=encoder_path,
            cross_attn_every_n_layers=x_atten_int,
            cache_dir="/home/users/u7212335/.cache",
        )

        checkpoint_path = hf_hub_download(args.model_path, "checkpoint.pt")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        
        if "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
            ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}

        self.model.load_state_dict(ckpt, strict=False)
        self.tokenizer.padding_side = "left"
        self.model = self.model.to(self.device)
    
    def forward_with_probs(self, images, prompt):        
        vision_x = [self.image_processor(image).unsqueeze(0) for image in images]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0).to(self.device)

        prefix = "".join(["<image>"] * len(images))
        if prompt.strip().endswith(".") or prompt.strip().endswith("。"):
            prompt = prompt.replace("Answer with the option's letter from the given choices directly.", "")
            prompt = prompt.replace("Answer the question using a single word or phrase.", "")
            prompt = prompt.replace("请直接使用所提供的选项字母作为答案回答。", "")

            lang = [prefix + str(prompt) + "\nAnswer: "]
        else:
            lang = [prefix + str(prompt)]
            
        lang_x = self.tokenizer(lang, return_tensors="pt").to(self.device)
        # print(lang)
        # print(vision_x.shape)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.model.generate(
                vision_x=vision_x,
                lang_x=lang_x["input_ids"],
                attention_mask=lang_x["attention_mask"],
                max_new_tokens=50,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,

                return_dict_in_generate=True,
                output_hidden_states=True,
                output_scores=True,
            )

        logits = torch.cat(outputs['scores'], dim=0).float().cpu().numpy()
        probs = [nn.functional.softmax(next_token_scores, dim=-1) for next_token_scores in outputs['scores']]
        probs = torch.cat(probs).float().cpu().numpy()
        output_ids = outputs["sequences"][0][-len(probs):]
    
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        output_ids = output_ids.cpu().numpy()

        hidden_states_all_layers = outputs['hidden_states'][0]
        hidden_states = hidden_states_all_layers[-1][0][[-1]].float()   # last layer, batch size=1, last token

        return response, output_ids, logits, probs, hidden_states


    # def answer_extractor(self, text, dataset_key):
    #     if dataset_key == 'hateful_memes' or dataset_key =='newyorkercartoon' or dataset_key =='irfl':
    #         text = text[:3]
    #         text = text.lower().strip()
    #         text = ''.join(filter(str.isalpha, text.lower()))
    #         return text
    #     elif dataset_key == 'memotion' or dataset_key == 'face_emotion'  or dataset_key == 'scienceqa' or dataset_key == 'vcr':
    #         match = re.search(r"\b\d\b", text)
    #         if match:
    #             first_number = int(match.group())
    #             return first_number
    #         else:
    #             return None
            