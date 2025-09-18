import soundfile as sf
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL_DIR = "./checkpoints/whisper-vn"

processor = WhisperProcessor.from_pretrained(MODEL_DIR)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR).to("cuda")

def transcribe(path):
    audio, sr = sf.read(path)
    inputs = processor.feature_extractor(audio, sampling_rate=sr, return_tensors="pt").input_features.to("cuda")
    generated_ids = model.generate(inputs)
    return processor.decode(generated_ids[0])

if __name__ == "__main__":
    print(transcribe("sample.wav"))
