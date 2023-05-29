import os
import re
import yaml
import torchaudio
from pathlib import Path
from datasets import DatasetDict, Dataset
import torch

def load_wav_dataset(directory, chunk_duration = 60):
    audio_paths = list(Path(directory+"/wav").rglob('*.wav'))
    yaml_path = os.path.join(directory, 'txt/tst-COMMON.yaml')
    i=0
    data = {
        "wave_info":[]
    }
    # Load YAML file
    with open(yaml_path, 'r') as yaml_file:
        yaml_data = yaml.load(yaml_file, Loader=yaml.FullLoader)

    #load source utterances
    for audio_path in audio_paths:
        audio_filename = audio_path.stem+".wav"
        waveform, sample_rate = torchaudio.load(audio_path)
        total_duration = waveform.shape[1] / sample_rate
        chunk_samples = int(sample_rate * chunk_duration)
        yaml_info = [info for info in yaml_data if info["wav"] == audio_filename]
        for info in yaml_info:
            duration = info["duration"]
            start = int(info["offset"])
            end = start + int(duration)
            
            if start >= total_duration:
                break
            if end > total_duration:
                end = waveform.shape[1]
                
                
            chunk = waveform[:, int(start * sample_rate):int(end * sample_rate)] 
            if chunk.numel() == 0:  # 빈 배열인 경우 0으로만 채워진 배열로 대체
                chunk = torch.zeros_like(waveform[:, :chunk_samples])
            data_element = {
                "array": chunk,
                "sample_rate": 16000,
                "duration": info["duration"],
                "offset": info["offset"],
                "rW": info["rW"],
                "uW": info["uW"],
                "speaker_id": info["speaker_id"],
                "file_name": audio_filename
            }
            data["wave_info"].append(data_element)

    dataset = Dataset.from_dict(data)
    dataset_dict = DatasetDict({"validation": dataset})
    return dataset_dict

def preprocess_text(text):
    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\n ]', '', text.lower())
    return text

def preprocess_dataset(dataset_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Preprocess the text files
    text_dir = os.path.join(dataset_dir, 'txt')
    output_text_dir = os.path.join(output_dir, 'txt')
    os.makedirs(output_text_dir, exist_ok=True)

    transcript_de_path = os.path.join(text_dir, 'tst-COMMON.de')
    transcript_en_path = os.path.join(text_dir, 'tst-COMMON.en')

    with open(transcript_de_path, 'r') as transcript_de_file, open(transcript_en_path, 'r') as transcript_en_file:
        transcript_de = transcript_de_file.read().strip()
        transcript_en = transcript_en_file.read().strip()

        preprocessed_transcript_de = preprocess_text(transcript_de)
        preprocessed_transcript_en = preprocess_text(transcript_en)

        output_transcript_de_path = os.path.join(output_text_dir, 'tst-COMMON.de')
        output_transcript_en_path = os.path.join(output_text_dir, 'tst-COMMON.en')

        with open(output_transcript_de_path, 'w') as output_de_file, open(output_transcript_en_path, 'w') as output_en_file:
            output_de_file.write(preprocessed_transcript_de)
            output_en_file.write(preprocessed_transcript_en)

    # Copy the audio files
    speech_data=load_wav_dataset(dataset_dir)
    return speech_data


#Preprocess the dataset
dataset_dir = './tst-COMMON-copy'
output_dir = './ST-output'
preprocess_dataset(dataset_dir, output_dir)
