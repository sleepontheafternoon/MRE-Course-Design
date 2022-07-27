import torch.nn as nn
from transformers import BeitFeatureExtractor, BeitModel
import torch

class BEiTEncoder(nn.Module):
    def __init__(self, beit_path):
        super().__init__()
        self.feature_extractor = BeitFeatureExtractor.from_pretrained(beit_path)
        self.model = BeitModel.from_pretrained(beit_path)
       
    def forward(self, images):
        inputs = self.feature_extractor(images, return_tensors="pt")  # inputs.pixel_values64 3 244 244
        feat_img = self.model(inputs.pixel_values.to(torch.device("cuda:0")), return_dict=True)  #self.model.device
        return feat_img.last_hidden_state  # 64 197 168