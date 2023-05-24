from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import jiwer

# 데이터셋 로드
dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean")

# 모델과 전처리기 불러오기
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

# 입력 오디오 전처리 함수
def preprocess_audio(audio):
    input_values = processor(audio["array"],sampling_rate = audio["sampling_rate"], return_tensors="pt", padding=True).input_values
    return input_values

# 추론 함수
def transcribe_audio(audio):
    input_values = preprocess_audio(audio)
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

pred_f = open("./prediction.txt", "w")
gth_f = open("./ground_truth.txt","w")
wer_total = 0
cer_total = 0
num_examples = 0

# 각 오디오에 대해 추론 실행
for data in dataset["validation"]:
    
    ground_truth = data["text"].lower()
    gth_f.write(ground_truth + "\n")
    print("ground truth:",ground_truth)
    transcription = transcribe_audio(data["audio"])
    pred_f.write(transcription + "\n")
    print("prediction:",transcription)
    
    wer = jiwer.wer(ground_truth,transcription)
    cer = jiwer.cer(ground_truth,transcription)
    wer_total += wer
    cer_total += cer
    num_examples +=1

average_wer = wer_total / num_examples
average_cer = cer_total / num_examples

print("Average WER:", average_wer)
print("Average CER:", average_cer)
    
    
    
pred_f.close()
gth_f.close()

