import torch
from torch import nn
import uuid
import os

from transformers import TextStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.base import LargeMultimodalModel


class InternLM_XComposer(LargeMultimodalModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map=self.device, trust_remote_code=True).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        self.model.tokenizer = self.tokenizer
        self.model_path = args.model_path

        self.model.tie_weights()
        self.model.to(self.device)

        self.temperature = args.temperature
        self.top_p = None
        self.num_beams = args.num_beams
        self.use_cache = True
        self.max_new_tokens = 4096

    def forward_with_probs(self, images, prompt):

        if "xcomposer2d5" in self.model_path:
            visual_path = []
            for i, visual in enumerate(images):
                name = uuid.uuid4().hex.upper()[0:6]
                visual.save(f"/tmp/{name}.png")
                visual_path.append(f"/tmp/{name}.png")
                prompt = f"Image {i} <ImageHere>; " + prompt
        else:
            name = uuid.uuid4().hex.upper()[0:6]
            images[0].save(f"/tmp/{name}.png")
            visual_path = f"/tmp/{name}.png"
            prompt = "<ImageHere>" + prompt

        with torch.cuda.amp.autocast():
            if "xcomposer2" in self.model_path:
                response, _ = self.model.chat(self.tokenizer, query=prompt, 
                                              image=visual_path, history=[], do_sample=False)
            else:
                response = self.model.generate(
                    prompt, visual_path, do_sample=False
                )
        
        if "xcomposer2d5" in self.model_path:
            for visual_path in visual_path:
                os.remove(visual_path)
        else:
            os.remove(visual_path)

        return response, None, None, None, None


    # def forward_with_probs(self, images, prompt):
    #     visual_paths = []
    #     for visual in images:
    #         name = uuid.uuid4().hex.upper()[0:6]
    #         visual.save(f"/tmp/{name}.png")
    #         visual_paths.append(f"/tmp/{name}.png")

    #     # name = uuid.uuid4().hex.upper()[0:6]
    #     # images[0].save(f"/tmp/{name}.png")
    #     # visual_path = f"/tmp/{name}.png"

    #     with torch.cuda.amp.autocast():
    #         response = self.model.generate(
    #             prompt, visual_paths, do_sample=False
    #         )

    #     # remove visuals from tmp
    #     for visual_path in visual_paths:
    #         try:
    #             os.remove(visual_path)
    #         except:
    #             pass

    #     return response, None, None, None, None

    # def message_to_prompt_embs(self, images, prompt):
    #     img_embeds = []
    #     prompt_segs = [f'<|User|>: {prompt}', self.model.eoh + ' <|Bot|>: ']
    #     for image in images:
    #         image = self.model.vis_processor(image).unsqueeze(0).to(self.device)
    #         img_embeds.append(self.model.encode_img(image))

    #     prompt_seg_tokens = [
    #         self.model.tokenizer(seg, return_tensors='pt', add_special_tokens=(i == 0)).to(self.device).input_ids.long()
    #         for i, seg in enumerate(prompt_segs)
    #     ]
    #     prompt_seg_embs = [self.model.internlm_model.model.embed_tokens(seg) for seg in prompt_seg_tokens]

    #     prompt_embs = torch.cat([prompt_seg_embs[0], img_embeds, prompt_seg_embs[1]], dim=1)
    #     return prompt_embs