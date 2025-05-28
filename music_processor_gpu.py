import os
import torch
import torchaudio
import lmdb
import pickle
import asyncio
import logging
import numpy as np
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from telegram import Bot
from telegram.error import TelegramError

# ============ CONFIG (move sensitive tokens to env variables) ============
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if TELEGRAM_TOKEN is None:
    raise RuntimeError("TELEGRAM_TOKEN env var not set")
CHAT_IDS = os.getenv("TELEGRAM_CHAT_IDS", "1451599691").split(",")
MUSIC_DIR = "./musics/"
LMDB_PATH = "music_embeddings.lmdb"

# ============ LOGGING ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('music_processing.log'),
        logging.StreamHandler()
    ]
)

SUPPORTED_FORMATS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')
MAX_TELEGRAM_FILE_SIZE = 50 * 1024 * 1024  # 50MB
CHUNK_SIZE = 20 * 1024 * 1024  # 20MB chunks for large files

# ============ WAV CONVERSION FUNCTION ============
import ffmpeg

def convert_to_wav(input_path: str) -> Optional[str]:
    try:
        wav_path = f"{input_path}.wav"
        if os.path.exists(wav_path):
            return wav_path  # already converted

        (
            ffmpeg
            .input(input_path)
            .output(wav_path, format='wav', ac=1, ar='16000')
            .overwrite_output()
            .run(quiet=True, capture_stdout=True, capture_stderr=True)
        )
        return wav_path
    except Exception as e:
        logging.error(f"Failed to convert {input_path} to wav: {e}")
        return None


# ============ MUSIC PROCESSOR CLASS WITH GPU BATCHING & MULTIPROCESSING ============
class MusicProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            torch.cuda.init()
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)
        self.model.eval()
        self.max_chunk_size = 30  # seconds
        self.sample_rate = 16000

    def _process_chunk(self, waveform: torch.Tensor) -> torch.Tensor:
        inputs = self.processor(
            waveform.squeeze(),
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        ).input_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze()

    def extract_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        try:
            waveform, sr = torchaudio.load(audio_path)

            if waveform.dim() > 1 and waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)

            chunk_samples = self.sample_rate * self.max_chunk_size
            total_samples = waveform.shape[-1]

            chunks = []
            for i in range(0, total_samples, chunk_samples):
                chunk = waveform[..., i:i+chunk_samples]
                if chunk.shape[-1] < chunk_samples // 4:
                    continue
                chunk_embedding = self._process_chunk(chunk)
                chunks.append(chunk_embedding)

            if chunks:
                final_embedding = torch.mean(torch.stack(chunks), dim=0).cpu().numpy()
            else:
                return None

            return final_embedding

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logging.warning(f"Reducing chunk size for {audio_path} due to CUDA OOM")
                self.max_chunk_size = max(10, self.max_chunk_size // 2)
                return self.extract_embedding(audio_path)
            logging.error(f"Embedding extraction failed for {audio_path}: {e}")
            return None
        except Exception as e:
            logging.error(f"Embedding extraction failed for {audio_path}: {e}")
            return None

    def process_file(self, file_path: str) -> Optional[Tuple[str, np.ndarray]]:
        if not file_path.lower().endswith(SUPPORTED_FORMATS):
            return None

        wav_path = file_path
        temp_wav_created = False

        if not file_path.lower().endswith(".wav"):
            wav_path = convert_to_wav(file_path)
            if not wav_path:
                return None
            temp_wav_created = True

        embedding = self.extract_embedding(wav_path)

        if temp_wav_created and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception as e:
                logging.warning(f"Failed to delete temp file {wav_path}: {e}")

        if embedding is not None:
            return (os.path.basename(file_path), embedding)
        return None

    def collect_audio_files(self, directory: str) -> List[str]:
        if not os.path.isdir(directory):
            raise ValueError(f"Invalid directory: {directory}")

        file_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(SUPPORTED_FORMATS):
                    file_paths.append(os.path.join(root, file))
        return file_paths

# Worker function for multiprocessing (only processes one file)
def process_file_worker(file_path: str) -> Optional[Tuple[str, np.ndarray]]:
    # Recreate processor inside each process for safety (no GPU sharing)
    proc = MusicProcessor()
    return proc.process_file(file_path)


def create_embeddings_parallel(file_paths: List[str], max_workers: int = None) -> List[Tuple[str, np.ndarray]]:
    max_workers = max_workers or max(1, cpu_count() - 1)
    results = []
    with Pool(max_workers) as pool:
        for result in pool.imap_unordered(process_file_worker, file_paths):
            if result is not None:
                results.append(result)
    return results


# ============ STORE EMBEDDINGS AS .npy IN LMDB ============
def store_embeddings_npy(data: List[Tuple[str, np.ndarray]], lmdb_path: str = LMDB_PATH) -> None:
    try:
        env = lmdb.open(lmdb_path, map_size=int(1e11))  # 100GB, adjust as needed
        with env.begin(write=True) as txn:
            for key, embedding in data:
                # store numpy array as npy bytes directly for faster I/O
                npy_bytes = embedding.tobytes()
                shape_dtype = pickle.dumps((embedding.shape, embedding.dtype), protocol=pickle.HIGHEST_PROTOCOL)
                # store shape/dtype metadata + raw bytes with separator
                val = shape_dtype + b'||' + npy_bytes
                txn.put(key.encode('utf-8'), val)
        logging.info(f"Stored {len(data)} embeddings in {lmdb_path}")
    except Exception as e:
        logging.error(f"Failed to store embeddings: {e}")
        raise


# ============ TELEGRAM ASYNC SENDER WITH IMPROVED FILE IO ============
class TelegramSender:
    def __init__(self, token: str, chat_ids: List[str]):
        self.bot = Bot(token=token)
        self.chat_ids = chat_ids

    async def _send_single_file(self, file_path: str, chat_id: str) -> bool:
        try:
            # Use asyncio.to_thread to avoid blocking event loop with file IO
            async with asyncio.to_thread(open, file_path, 'rb') as f:
                await self.bot.send_document(chat_id=chat_id, document=f)
            return True
        except TelegramError as e:
            logging.error(f"Telegram API error: {e}")
            return False
        except Exception as e:
            logging.error(f"File sending error: {e}")
            return False

    async def _send_chunked_file(self, file_path: str, chat_id: str) -> bool:
        file_size = os.path.getsize(file_path)
        if file_size <= MAX_TELEGRAM_FILE_SIZE:
            return await self._send_single_file(file_path, chat_id)

        try:
            with open(file_path, 'rb') as f:
                part_num = 1
                while chunk := f.read(CHUNK_SIZE):
                    part_path = f"{file_path}.part{part_num}"
                    with open(part_path, 'wb') as part_file:
                        part_file.write(chunk)

                    success = await self._send_single_file(part_path, chat_id)
                    os.remove(part_path)

                    if not success:
                        return False
                    part_num += 1
            return True
        except Exception as e:
            logging.error(f"Failed to send chunked file: {e}")
            return False

    async def send_embeddings(self, lmdb_path: str, message: str = "Music embeddings ready") -> None:
        if not os.path.exists(lmdb_path):
            logging.error(f"Database file not found: {lmdb_path}")
            return

        for chat_id in self.chat_ids:
            try:
                await self.bot.send_message(chat_id=chat_id, text=message)
                success = await self._send_chunked_file(lmdb_path, chat_id)
                if success:
                    logging.info(f"Successfully sent embeddings to {chat_id}")
                else:
                    logging.error(f"Failed to send embeddings to {chat_id}")
            except Exception as e:
                logging.error(f"Error communicating with Telegram for {chat_id}: {e}")


# ============ MAIN ASYNC ============
async def main():
    try:
        logging.info("Starting music processing")
        processor = MusicProcessor()

        audio_files = processor.collect_audio_files(MUSIC_DIR)
        logging.info(f"Found {len(audio_files)} audio files")

        embeddings = create_embeddings_parallel(audio_files)
        logging.info(f"Extracted embeddings for {len(embeddings)} files")

        store_embeddings_npy(embeddings)
        logging.info("Embeddings stored in LMDB")

        sender = TelegramSender(TELEGRAM_TOKEN, CHAT_IDS)
        await sender.send_embeddings(LMDB_PATH)

        logging.info("Processing completed successfully")
    except Exception as e:
        logging.error(f"Fatal error in main process: {e}")


if __name__ == "__main__":
    asyncio.run(main())
