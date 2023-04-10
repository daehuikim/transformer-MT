# Edited by: daehuiKim
import torch.nn as nn
import copy
import torch

"""
LayerNorm class
params explanation
features => input size
eps => epsilon for numerical stability

function expl
forward => return normalized input
"""
class LayerNorm(nn.Module):

    def __init__(self, features, eps = 1e-6) -> None:
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features)) #for efficiency we assume it as 1
        self.beta = nn.Parameter(torch.zeros(features)) #for efficiency we assume it as 0
        self.eps = eps

    #Normalization eps:small value 
    def forward(self,x):
        mean = x.mean(-1, keepdim=True) 
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

"""
SublayerConnection class
params explanation
size => input size
dropout => dropout rate

function explanation
forward => return normalized input
"""
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout) -> None:
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

"""
params explanation
module => EncoderLayer
N => number of layers
"""
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


"""
return subsequent masking
"""
def subsequent_mask(size):
    shape = (1,size,size)
    mask = torch.triu(torch.ones(shape), diagonal=1).type(torch.uint8)
    return mask == 0

