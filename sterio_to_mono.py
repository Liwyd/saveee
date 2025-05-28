import os
import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from tqdm import tqdm
import logging

# تنظیم لاگ فقط برای خطاها
logging.basicConfig(level=logging.ERROR)

# استفاده از GPU اگر موجود بود
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# بارگذاری مدل و پردازشگر
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
model.eval()

music_folder = "./musics"
output_folder = "./embeddings"
os.makedirs(output_folder, exist_ok=True)

# تنظیمات batch
batch_size = 16

# لیست همه فایل‌های wav
wav_files = [f for f in os.listdir(music_folder) if f.endswith(".wav")]

# تقسیم‌بندی فایل‌ها به batch
for i in tqdm(range(0, len(wav_files), batch_size), desc="Extracting embeddings"):
    batch_files = wav_files[i:i + batch_size]
    waveforms = []
    sample_rates = []
    valid_filenames = []

    for filename in batch_files:
        filepath = os.path.join(music_folder, filename)
        try:
            waveform, sr = torchaudio.load(filepath)

            # تبدیل به mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # resample اگر لازم بود
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform)
                sr = 16000

            waveforms.append(waveform)
            sample_rates.append(sr)
            valid_filenames.append(filename)

        except Exception as e:
            logging.error(f"Loading failed for {filepath}: {e}")

    if not waveforms:
        continue

    try:
        # پردازش batch با padding
        inputs = processor(
            [w.squeeze(0) for w in waveforms],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        input_values = inputs.input_values.to(device)

        # گرفتن embedding
        with torch.no_grad():
            outputs = model(input_values)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu()

        # ذخیره embeddings هر فایل
        for filename, embedding in zip(valid_filenames, embeddings):
            out_path = os.path.join(output_folder, filename.replace(".wav", ".pt"))
            torch.save(embedding, out_path)

    except Exception as e:
        logging.error(f"Batch processing failed at index {i}: {e}")
