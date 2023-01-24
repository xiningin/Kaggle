import torch.nn.functional as F
import torch.nn as nn
import torch
from transformers import AutoModel

class MarkdownModel(nn.Module):
    def __init__(self , model_path):
        super(MarkdownModel , self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.top = nn.Linear(769 , 1)
    def forward(self , ids , mask , markdown_rate):
        # markdown_rate: 32 , 1
        x = self.model(ids , mask)[0] # dim: torch.Size([32, 512, 768])
        x = torch.cat((x[: , 0 , :] , markdown_rate) , 1) # dim: torch.Size([32, 769])
        x = self.top(x) # dim: torch.Size([32, 1])
        return x