import numpy as np

from model.base import LargeMultimodalModel

class Random(LargeMultimodalModel):
    def __init__(self, device):
        super(Random, self).__init__(device)
    
    def forward(self, image, prompt):
        return np.random.choice(["A", "B", "C", "D", "E"])