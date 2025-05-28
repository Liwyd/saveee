import os
import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from tqdm import tqdm
import logging

# تنظیمات لاگ برای خطاها
logging.basicConfig(level=logging.ERROR)

# بارگذاری مدل و پردازشگر
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()

# استفاده از CUDA اگر در دسترس بود
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

music_folder = "./musics"
output_folder = "./embeddings"
os.makedirs(output_folder, exist_ok=True)

for filename in tqdm(os.listdir(music_folder)):
    if not filename.endswith(".wav"):
        continue

    filepath = os.path.join(music_folder, filename)
    try:
        # بارگذاری فایل صوتی
        waveform, sample_rate = torchaudio.load(filepath)

        # تبدیل به mono اگر stereo بود
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample به 16000Hz در صورت نیاز
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # آماده‌سازی برای مدل
        input_values = processor(waveform.squeeze(0), sampling_rate=sample_rate, return_tensors="pt").input_values
        input_values = input_values.to(device)

        # گرفتن embedding
        with torch.no_grad():
            outputs = model(input_values)
            embedding = outputs.last_hidden_state.mean(dim=1).cpu()  # شکل: [1, hidden_dim]

        # ذخیره به صورت tensor
        output_path = os.path.join(output_folder, filename.replace(".wav", ".pt"))
        torch.save(embedding, output_path)

    except Exception as e:
        logging.error(f"Embedding extraction failed for {filepath}: {e}")
