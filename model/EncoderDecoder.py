# Edited by: daehuiKim
import torch.nn as nn
import torch

# Most High level architecture
class EncoderDecoder(nn.Module):
    """
    Params explanation
    encoder,decoder,generator => part in fig
    src_embed, tgt_embed = > input of encoder(embeded), output of encoder(embeded)

    function explanation
    encode => return context vector
    decode => return output
    forward => return output
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator) -> None:
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    """
    Params explanation
    src => trainning data
    src_mask = > pad masking for self-attention
    """
    def encode(self, src, src_mask):
        return self.encoder(
            self.src_embed(src),
            src_mask
            )
    
    """
    Params explanation
    ##! check !## memory => context vector from encoder
    src_mask => pad masking for self-attention
    tgt => output
    tgt_mask => subsequent masking for generator 
    """
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(
            self.tgt_embed(tgt),
            memory,
            src_mask,
            tgt_mask
            )

    """
    Params explanation
    src => input
    tgt => output
    src_mask => pad masking for self-attention
    tgt_mask => subsequent masking for generator 
    """
    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(
            self.encode(src, src_mask),
                           src_mask,
                           tgt,
                           tgt_mask
                           )

"""
return subsequent masking
"""
def subsequent_mask(size):
    shape = (1,size,size)
    mask = torch.triu(torch.ones(shape), diagonal=1).type(torch.uint8)
    return mask == 0
