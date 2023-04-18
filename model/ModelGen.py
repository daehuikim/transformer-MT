from . import MultiHeadedAttention,PositionalEncoding,PositionwiseFeedForward,EncoderDecoder,Generator,Embeddings
from .encoder import Encoder, EncoderLayer
from .decoder import Decoder, DecoderLayer
import torch.nn as nn
import copy


def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention.MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward.PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding.PositionalEncoding(d_model, dropout)
    model = EncoderDecoder.EncoderDecoder(
        Encoder.Encoder(EncoderLayer.EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder.Decoder(DecoderLayer.DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings.Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings.Embeddings(d_model, tgt_vocab), c(position)),
        Generator.Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model