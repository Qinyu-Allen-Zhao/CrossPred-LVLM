import os
import time
import base64
import requests
import uuid

from openai import OpenAI, BadRequestError
from model.base import LargeMultimodalModel

def is_image_too_big(image_path):
    file_size = os.path.getsize(image_path)
    file_size_mb = file_size / (1024 * 1024)
    return file_size_mb > 10

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class GPTClient(LargeMultimodalModel):
    def __init__(self, args):
        super().__init__()
        self.model = args.model_path
    
    def forward_with_probs(self, images, prompt):
        client = OpenAI(api_key="sk-proj-b6xqhfX4pceCh77xGUlAT3BlbkFJhh7TNEQiUK2EHFlTvfVX")
        
        visual_paths = []
        for visual in images:
            name = uuid.uuid4().hex.upper()[0:6]
            visual.save(f"/tmp/{name}.png")
            while is_image_too_big(f"/tmp/{name}.png"):
                visual.thumbnail((visual.width // 2, visual.height // 2))
                print("Image is too big, shrink it ... ", visual.size)
                visual.save(f"/tmp/{name}.png")
            visual_paths.append(f"/tmp/{name}.png")

        # Getting the base64 string
        base64_images = [encode_image(p) for p in visual_paths]
        
        PROMPT_MESSAGES = [
            {
                "role": "user",
                "content": [
                    prompt,
                    *map(lambda x: {"image": x}, base64_images),
                ],
            },
        ]
        params = {
            "model": self.model,
            "messages": PROMPT_MESSAGES,
            "max_tokens": 1024,
        }

        while 1:
            try:
                result = client.chat.completions.create(**params)
                response = result.choices[0].message.content
                return response, None, None, None, None
            except Exception as e:
                print(e, flush=True)
                print('Timeout error, retrying...')
                time.sleep(5)

        # remove visuals from tmp
        for visual_path in visual_paths:
            try:
                os.remove(visual_path)
            except:
                pass