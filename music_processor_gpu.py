import os
import torch
import torchaudio
import ffmpeg
import pandas as pd
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from dask import delayed, compute

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Wav2Vec2 model & processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
model.eval()

SUPPORTED_FORMATS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')

def convert_to_wav(input_path):
    output_path = input_path + ".temp.wav"
    try:
        ffmpeg.input(input_path).output(output_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000').run(quiet=True, overwrite_output=True)
        return output_path
    except Exception as e:
        print(f"[FFMPEG ERROR] {input_path} - {e}")
        return None

def extract_embedding(audio_path):
    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
        input_values = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_values.to(device)
        with torch.no_grad():
            outputs = model(input_values)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding
    except Exception as e:
        print(f"[EXTRACT ERROR] {audio_path} - {e}")
        return None

@delayed
def process_file(file_path):
    if not file_path.lower().endswith(SUPPORTED_FORMATS):
        return None

    wav_path = file_path
    if not file_path.endswith(".wav"):
        wav_path = convert_to_wav(file_path)
        if not wav_path:
            return None

    embedding = extract_embedding(wav_path)
    if wav_path != file_path:
        os.remove(wav_path)  # clean up

    if embedding is not None:
        return {"file": os.path.basename(file_path), **{f"emb_{i}": x for i, x in enumerate(embedding)}}
    return None

def collect_all_files(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(SUPPORTED_FORMATS):
                file_paths.append(os.path.join(root, file))
    return file_paths

def process_directory(directory, output_csv="music_embeddings.csv"):
    all_files = collect_all_files(directory)
    print(f"Total files to process: {len(all_files)}")
    tasks = [process_file(f) for f in all_files]
    results = compute(*tasks, scheduler="processes")
    results = [r for r in results if r is not None]
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"✅ Features saved to {output_csv}")

if __name__ == "__main__":
    input_dir = "./musics/"
    process_directory(input_dir)
