import os
import time

import google.generativeai as genai

from model.base import LargeMultimodalModel


class GeminiClient(LargeMultimodalModel):
    def __init__(self, args):
        super().__init__()
        genai.configure(api_key="AIzaSyD2WUIuDIVxRquEtXitjDy0UlTDngyCDkQ")
        self.model = genai.GenerativeModel(model_name=args.model_path)

    def forward_with_probs(self, images, prompt):
        # while 1:
            try:
                response = self.model.generate_content([prompt] + images,safety_settings=[
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}
                ])
                # print(response)
                # print(response.text)
                return response.text, None, None, None, None
            except Exception as e:
                print(response)
                return "<BLOCKED>", None, None, None, None
