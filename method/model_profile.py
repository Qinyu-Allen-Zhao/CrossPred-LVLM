import json
import numpy as np

MODEL_MAP = {
    'BLIP': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Cambrian': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Fuyu': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'GPT4': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Gemini': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'InternLM': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'LLaMA-Adapter': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'LLaVA': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'MiniGPT4': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'OpenFlamingo': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'Qwen': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'mPLUG-Owl': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'prism': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
}

ENC_MAP = {
    'CLIP': [1, 0, 0, 0, 0, 0],
    'ConvNeXt': [0, 1, 0, 0, 0, 0],
    'DINOv2': [0, 0, 1, 0, 0, 0],
    'NFNet': [0, 0, 0, 1, 0, 0],
    'SigLIP': [0, 0, 0, 0, 1, 0],
    'ViT': [0, 0, 0, 0, 0, 1],
}


class ModelProfile:
    def __init__(self, profile_content):
        self.model_profile_map = {}
        self.profile_content = profile_content
        self.load_profiles()

    def load_profiles(self):
        with open("data/model_profiles.json", 'r') as f:
            self.model_profile_map = json.load(f)
    
    def get_profile(self, model_key):
        profile = self.model_profile_map[model_key]
        
        profile_vector = [] # list(np.random.randn(5))
        for key in self.profile_content:
            if key == "random":
                idx = np.random.randint(0, 5, 1)
                value = np.eye(5)[idx].flatten().tolist()
            else:
                value = profile[key]

            if key == "vision_encoder":
                if value is None or len(value) == 0:
                    value = [0, 0, 0, 0, 0, 0]
                else:
                    enc_one_hot = np.array([ENC_MAP[enc] for enc in value]).reshape([-1, 6])
                    value = np.sum(enc_one_hot, axis=0)
            elif key == "model_family":
                value = MODEL_MAP[value]
            elif key == "num_params_llm":
                value = [value] if value is not None else [0]
            elif key == "gt_cluster" or key == "random":
                pass
            else:
                value = [value]

            profile_vector.extend(value)

        return np.array(profile_vector)