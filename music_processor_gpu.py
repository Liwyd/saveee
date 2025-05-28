import os
import torch
import torchaudio
import ffmpeg
import lmdb
import pickle
import asyncio
from tqdm import tqdm
from dask import delayed, compute
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from telegram import Bot
from telegram.error import TelegramError
from pathlib import Path
from typing import List, Tuple, Optional
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('music_processing.log'),
        logging.StreamHandler()
    ]
)

# Constants
SUPPORTED_FORMATS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')
MAX_TELEGRAM_FILE_SIZE = 50 * 1024 * 1024  # 50MB
CHUNK_SIZE = 20 * 1024 * 1024  # 20MB chunks for large files

class MusicProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._init_models()
        
    def _init_models(self):
        """Initialize audio processing models"""
        try:
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)
            self.model.eval()
            logging.info("Models loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load models: {e}")
            raise

    def convert_to_wav(self, input_path: str) -> Optional[str]:
        """Convert audio file to WAV format"""
        output_path = f"{input_path}.temp.wav"
        try:
            (
                ffmpeg.input(input_path)
                .output(output_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
                .run(quiet=True, overwrite_output=True)
            )
            return output_path
        except ffmpeg.Error as e:
            logging.error(f"FFmpeg error for {input_path}: {e.stderr.decode()}")
            return None
        except Exception as e:
            logging.error(f"Conversion error for {input_path}: {e}")
            return None

    def extract_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        """Extract audio embeddings using Wav2Vec2"""
        try:
            waveform, sr = torchaudio.load(audio_path)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
            
            inputs = self.processor(
                waveform.squeeze(), 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_values.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(inputs)
                
            return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        except Exception as e:
            logging.error(f"Embedding extraction failed for {audio_path}: {e}")
            return None

    @delayed
    def process_file(self, file_path: str) -> Optional[Tuple[str, np.ndarray]]:
        """Process a single audio file"""
        if not file_path.lower().endswith(SUPPORTED_FORMATS):
            return None

        wav_path = file_path
        if not file_path.endswith(".wav"):
            wav_path = self.convert_to_wav(file_path)
            if not wav_path:
                return None

        embedding = self.extract_embedding(wav_path)
        
        # Cleanup temporary file
        if wav_path != file_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception as e:
                logging.warning(f"Failed to delete temp file {wav_path}: {e}")

        if embedding is not None:
            return (os.path.basename(file_path), embedding)
        return None

    def collect_audio_files(self, directory: str) -> List[str]:
        """Collect all supported audio files from directory"""
        if not os.path.isdir(directory):
            raise ValueError(f"Invalid directory: {directory}")
            
        file_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(SUPPORTED_FORMATS):
                    file_paths.append(os.path.join(root, file))
        return file_paths

    def create_embeddings(self, file_paths: List[str]) -> List[Tuple[str, np.ndarray]]:
        """Process multiple files in parallel"""
        tasks = [self.process_file(f) for f in file_paths]
        results = compute(*tasks, scheduler="processes")
        return [r for r in results if r is not None]

    def store_embeddings(self, data: List[Tuple[str, np.ndarray]], 
                       lmdb_path: str = "music_embeddings.lmdb") -> None:
        """Store embeddings in LMDB database"""
        try:
            env = lmdb.open(lmdb_path, map_size=1e12)  # 1TB
            with env.begin(write=True) as txn:
                for key, embedding in tqdm(data, desc="Storing embeddings"):
                    txn.put(key.encode('utf-8'), pickle.dumps(embedding))
            logging.info(f"Successfully stored {len(data)} embeddings in {lmdb_path}")
        except Exception as e:
            logging.error(f"Failed to store embeddings: {e}")
            raise

class TelegramSender:
    def __init__(self, token: str):
        self.bot = Bot(token=token)
        self.chat_ids = ["1451599691"]  # Should be loaded from config
        
    async def _send_chunked_file(self, file_path: str, chat_id: str) -> bool:
        """Send large files in chunks"""
        file_size = os.path.getsize(file_path)
        if file_size <= MAX_TELEGRAM_FILE_SIZE:
            return await self._send_single_file(file_path, chat_id)
            
        # Split large file
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

    async def _send_single_file(self, file_path: str, chat_id: str) -> bool:
        """Send a single file to Telegram"""
        try:
            with open(file_path, 'rb') as f:
                await self.bot.send_document(chat_id=chat_id, document=f)
            return True
        except TelegramError as e:
            logging.error(f"Telegram API error: {e}")
            return False
        except Exception as e:
            logging.error(f"File sending error: {e}")
            return False

    async def send_embeddings(self, lmdb_path: str, message: str = "Music embeddings ready") -> None:
        """Send embeddings database with status message"""
        if not os.path.exists(lmdb_path):
            logging.error(f"Database file not found: {lmdb_path}")
            return

        for chat_id in self.chat_ids:
            try:
                # Send status message
                await self.bot.send_message(chat_id=chat_id, text=message)
                
                # Send the database file
                success = await self._send_chunked_file(lmdb_path, chat_id)
                
                if success:
                    logging.info(f"Successfully sent embeddings to {chat_id}")
                else:
                    logging.error(f"Failed to send embeddings to {chat_id}")
            except Exception as e:
                logging.error(f"Error communicating with Telegram for {chat_id}: {e}")

async def main():
    try:
        TELEGRAM_TOKEN = "7929986601:AAFgh1oRiO5mmL3pFlmLnX8Qp2UFVoslHzQ"
        MUSIC_DIR = "./musics/"
        
        # Initialize components
        processor = MusicProcessor()
        sender = TelegramSender(TELEGRAM_TOKEN)

        # Process music files
        logging.info("Starting music processing")
        audio_files = processor.collect_audio_files(MUSIC_DIR)
        logging.info(f"Found {len(audio_files)} audio files")
        
        embeddings = processor.create_embeddings(audio_files)
        processor.store_embeddings(embeddings)
        
        # Send results
        await sender.send_embeddings("music_embeddings.lmdb")
        
        logging.info("Processing completed successfully")
    except Exception as e:
        logging.error(f"Fatal error in main process: {e}")

if __name__ == "__main__":
    # Create event loop for async operations
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()