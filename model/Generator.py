# Edited by: daehuiKim
import torch.nn as nn
import torch.nn.functional as F

"""
Generator class
params explanation
d_model => dimension of model
vocab => vocabulary size

function explanation
forward => return normalized input
"""
class Generator(nn.Module):
    def __init__(self,d_model, vocab) -> None:
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model,vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)