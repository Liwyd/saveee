import os
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import logging

# لاگ سطح خطا
logging.basicConfig(level=logging.ERROR)

# تنظیمات اولیه
music_folder = "./musics"
output_folder = "./embeddings"
os.makedirs(output_folder, exist_ok=True)

# بارگذاری مدل و پردازشگر
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()

# استفاده از GPU در صورت وجود
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# پردازش فایل (در thread)
def process_file(filename):
    if not filename.endswith(".wav"):
        return

    filepath = os.path.join(music_folder, filename)
    try:
        waveform, sample_rate = torchaudio.load(filepath)

        # اگر stereo بود -> mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # اگر نرخ نمونه‌برداری درست نبود -> resample
        if sample_rate != 16000:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        return filename, waveform, sample_rate

    except Exception as e:
        logging.error(f"Loading failed for {filepath}: {e}")
        return None

# فایل‌ها
all_files = [f for f in os.listdir(music_folder) if f.endswith(".wav")]

# موازی‌سازی I/O و پردازش اولیه
waveform_data = []
with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    results = list(tqdm(executor.map(process_file, all_files), total=len(all_files)))

for item in results:
    if item:
        waveform_data.append(item)

# پردازش batch در GPU
BATCH_SIZE = 16  # با توجه به VRAM می‌تونی بیشتر یا کمترش کنی

for i in tqdm(range(0, len(waveform_data), BATCH_SIZE), desc="Extracting embeddings"):
    batch = waveform_data[i:i + BATCH_SIZE]
    if not batch:
        continue

    filenames, waveforms, sample_rates = zip(*batch)

    input_values = []
    for waveform, sr in zip(waveforms, sample_rates):
        inputs = processor(waveform.squeeze(0), sampling_rate=sr, return_tensors="pt").input_values
        input_values.append(inputs)

    # batch کردن
    input_values = torch.cat(input_values, dim=0).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        outputs = model(input_values)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu()

    # ذخیره هر embedding به فایل جدا
    for emb, fname in zip(embeddings, filenames):
        out_path = os.path.join(output_folder, fname.replace(".wav", ".pt"))
        torch.save(emb, out_path)
