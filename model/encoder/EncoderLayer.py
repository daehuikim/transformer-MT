import torch.nn as nn
from Utils import clones, SublayerConnection

"""
params explanation
size => dimension of model
self_attention => self attention
feed_forward => feed forward
dropout => dropout rate

function explanation
forward => return normalized input
"""
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attention, feed_forward, dropout) -> None:
        super(EncoderLayer, self).__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout),2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)