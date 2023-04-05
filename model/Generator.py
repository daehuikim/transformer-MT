# Edited by: daehuiKim
import torch.nn as nn
import torch.nn.functional as F

#generator for output
class Generator(nn.Mudule):
    """
    Params explanation
    d_model => attention output generated by EncoderDecoder
    vocab => output sentences
    """
    def __init__(self,d_model, vocab) -> None:
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model,vocab)

    """
    Params explanation
    x => Linear transformation of attention
    last dimension of proj will be calculated
    """
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)