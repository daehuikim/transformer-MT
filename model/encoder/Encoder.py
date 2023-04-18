# Edited by: daehuiKim
import torch.nn as nn
from Utils import clones, LayerNorm
"""
Encoder class
params explanation
layer => EncoderLayer
N => number of layers

function explanation
forward => return normalized input
"""
class Encoder(nn.Module):

    def __init__(self, layer, N) -> None:
        super(Encoder,self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)
    