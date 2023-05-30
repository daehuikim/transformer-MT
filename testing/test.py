import torch
from . import Iterators, DataGen
import os,sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from trainning import Train,Batch
from model import ModelGen
from torchtext.data.utils import get_tokenizer
import torch.nn as nn
from transformers import MarianMTModel, MarianTokenizer


def check_outputs(
    valid_dataloader,
    model,
    vocab_src,
    vocab_tgt,
    n_examples=15,
    pad_idx=2,
    eos_string="</s>",
):
    results = [()] * n_examples
    for idx in range(n_examples):
        print("\nExample %d ========\n" % idx)
        b = next(iter(valid_dataloader))
        rb = Batch.Batch(b[0], b[1], pad_idx)
        Train.greedy_decode(model, rb.src, rb.src_mask, 64, 0)[0]
        
        
        src_tokens = [
            vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx
        ]
        tgt_tokens = [
            vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx
        ]

        
        print(
            "Source Text (Input)        : "
            + " ".join(src_tokens).replace("\n", "")
        )
        print(
            "Target Text (Ground Truth) : "
            + " ".join(tgt_tokens).replace("\n", "")
        )
        model_out = Train.greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
        model_txt = (
            " ".join(
                [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
            ).split(eos_string, 1)[0]
            + eos_string
        )
        print("Model Output               : " + model_txt.replace("\n", ""))
        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)
        
    return results


def run_model_example(n_examples=5):
    src, tgt = DataGen.load_tokenizers()
    vocab_src, vocab_tgt = DataGen.load_vocab(src, tgt)

    print("Preparing Data ...")
    _, valid_dataloader = Iterators.create_dataloaders(
        torch.device("cpu"),
        vocab_src,
        vocab_tgt,
        src,
        tgt,
        batch_size=1,
        is_distributed=False,
    )

    print("Loading Trained Model ...")

    model = ModelGen.make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(
        torch.load("en_to_de_model_final.pt", map_location=torch.device("cpu"))
    )

    print("Checking Model Outputs:")
    example_data = check_outputs(
        valid_dataloader, model, vocab_src, vocab_tgt, n_examples=n_examples
    )
    return model, example_data


def translate(input_file, output_file, vocab_file, model_file):
    # Load vocabulary
    src, tgt = DataGen.load_tokenizers()
    vocab_src, vocab_tgt = DataGen.load_vocab(src, tgt)
    
    # Load model
    model = torch.load(model_file)
    model = MyModel()
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Set tokenizer
    tokenizer = get_tokenizer("basic_english")

    # Read input text
    with open(input_file, "r", encoding="utf-8") as f:
        input_text = f.read().strip()

    # Tokenize input text
    input_tokens = tokenizer(input_text)
    
    # Convert tokens to indices
    input_indices = [vocab_src[token] for token in input_tokens]

    # Convert indices to tensor
    input_tensor = torch.LongTensor(input_indices).unsqueeze(0).to(device)

    # Perform translation
    with torch.no_grad():
        model.eval()
        output_tensor = model.forward(input_tensor)

    # Convert tensor back to indices
    output_indices = output_tensor.squeeze(0).argmax(dim=-1).cpu().numpy().tolist()

    # Convert indices to tokens
    output_tokens = [vocab_tgt[index] for index in output_indices]

    # Join tokens to form translation text
    output_text = " ".join(output_tokens)

    # Write translation text to output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_text)
    


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        x = self.fc(x)
        return x

def generate_inference(input_file,output_file,model_name):
    model= MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    with open (input_file, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    # GPU를 사용할 수 있는 경우에는 GPU로 모델을 이동
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    translated_sentences = []
    for sentence in sentences:
        # 문장을 토큰화 및 인코딩
        encoded = tokenizer(sentence.strip(), padding="longest", truncation=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        # 번역 수행
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)

        # 번역 결과 디코딩
        translated_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translated_sentences.append(translated_sentence)

    # 번역 결과를 파일에 저장
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(translated_sentences))