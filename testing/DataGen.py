import os
from os.path import exists
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import torch
import spacy

def load_tokenizers():
    
    languageDirection = 0
    if languageDirection == 0:
        source = "en_core_web_sm"
        target = "de_core_news_sm"
    elif languageDirection==1:
        source = "de_core_news_sm"
        target = "en_core_web_sm"

    try:
        src = spacy.load(source)
    except IOError:
        os.system("python -m spacy download" + source)
        src = spacy.load(source)

    try:
        tgt = spacy.load(target)
    except IOError:
        os.system("python -m spacy download" + target)
        tgt = spacy.load(target)

    return src, tgt

def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])

def build_vocabulary(src, tgt):
    languageDirection = 0
    if languageDirection == 0:
        language_pair=("en", "de")
    elif languageDirection==1:
        language_pair=("de", "en")
    
    def tokenize_src(text):
        return tokenize(text, src)

    def tokenize_tgt(text):
        return tokenize(text, tgt)

    print("Building Source Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair)
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_src, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building Target Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair)
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_tgt, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])
    return vocab_src, vocab_tgt


def load_vocab(src, tgt):
    if not exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(src, tgt)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt