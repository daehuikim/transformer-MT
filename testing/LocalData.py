from torchtext.vocab import build_vocab_from_iterator
import torch
from os.path import exists

def load_tokenizers(source_file,target_file):
    def tokenize_text(file_path):
        with open(file_path, 'r') as file:
            text = file.read()
        return text.split()  # Simple tokenization function that splits text on whitespace

    src_tokenizer = tokenize_text(source_file)
    tgt_tokenizer = tokenize_text(target_file)

    return src_tokenizer, tgt_tokenizer

def tokenize(text):
    return text.split()

def yield_tokens(data_iter, index, src_tokenizer, tgt_tokenizer):
    for data in data_iter:
        yield src_tokenizer(data[index][0]), tgt_tokenizer(data[index][1])

def build_vocabulary(src_tokenizer, tgt_tokenizer):
    train_data = [
        ("prediction.txt", "grth.txt"),
        # 추가적인 훈련 데이터 파일 경로를 추가할 수 있습니다.
    ]

    print("Building Source Vocabulary ...")
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train_data, index=0, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building Target Vocabulary ...")
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train_data, index=1, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])
    return vocab_src, vocab_tgt

def load_vocab(src,tgt):
    if not exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(src, tgt)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("Load Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt
