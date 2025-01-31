import os
import torch
import pandas as pd
import torchaudio
from datasets import Dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments

# Paths
WAV_INPUT_DIR = "./split_audio_wav"  # Folder containing WAV clips
OUTPUT_CSV = "transcriptions_whisper.csv"  # CSV file for transcriptions
MODEL_DIR = "/media/rf/T9/Paddlespeech_backup/my_whisper_model/output_dir"  # Path to the fine-tuned Whisper model

# Load Whisper model & processor
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = WhisperProcessor.from_pretrained(MODEL_DIR)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)

# Force transcription in English
forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(language="en", task="transcribe")

# Collect all WAV files
wav_files = [os.path.join(WAV_INPUT_DIR, f) for f in os.listdir(WAV_INPUT_DIR) if f.endswith(".wav")]

# Create a dataset from the WAV files
df = pd.DataFrame({"Path": wav_files})
dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column("Path", Audio(sampling_rate=16000))

# Function to prepare dataset for Whisper
def prepare_dataset(examples):
    audio = examples["Path"]
    examples["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=16000
    ).input_features[0]
    return examples

# Process dataset
dataset = dataset.map(prepare_dataset, num_proc=1)

# Transcribe audio clips
print("Transcribing audio segments...")
transcriptions = []

for example in dataset:
    input_features = torch.tensor(example["input_features"]).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    
    transcription = processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    transcriptions.append(transcription)

# Save transcriptions to CSV
df["Transcription"] = transcriptions
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print(f"Transcriptions saved to {OUTPUT_CSV}")

