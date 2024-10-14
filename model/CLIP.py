# This model is only used for encoding

import torch
import clip
import numpy as np
from FlagEmbedding import BGEM3FlagModel

from model.base import LargeMultimodalModel


class CLIP(LargeMultimodalModel):
    def __init__(self, args):
        super().__init__()
        # Load a pretrained VLM (either local path, or ID to auto-download from the HF Hub) 
        self.model, self.preprocess = clip.load(args.model_path, device=self.device)
        self.lm_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)


    def forward_with_probs(self, images, prompt):
        image = images[0]

        image = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.lm_model.encode([prompt], batch_size=1, max_length=2048)['dense_vecs']

            hs = np.concatenate((image_features.cpu().numpy(), text_features), axis=1)

        print(hs.shape)
        
        # response, output_ids, logits, probs, hidden_states
        return "", [], torch.Tensor([[0]]), [], hs
    