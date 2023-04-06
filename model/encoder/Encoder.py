import torch.nn as nn
import Utils,LayerNorm

class Encoder(nn.Module):

    def __init__(self, layer, N) -> None:
        super(Encoder,self).__init__()
        self.layer = Utils.clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)