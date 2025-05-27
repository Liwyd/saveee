import os
import librosa, asyncio
import numpy as np
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm
from telegram import Bot
from telegram.error import TelegramError
# sudo apt install ffmpeg
SUPPORTED_FORMATS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')

async def convert_to_wav(path):
    audio = AudioSegment.from_file(path)
    wav_path = path + ".temp.wav"
    audio.export(wav_path, format="wav")
    return wav_path

async def extract_features(file_path, sr=22050):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        features = {}

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}'] = np.mean(mfccs[i])

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_stft_mean'] = np.mean(chroma)

        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features['spectral_contrast_mean'] = np.mean(spec_contrast)

        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        features['tonnetz_mean'] = np.mean(tonnetz)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo[0]

        return features
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return None

async def process_directory(directory):
    data = []
    for root, _, files in os.walk(directory):
        for file in tqdm(files, desc="Processing files"):
            if not file.lower().endswith(SUPPORTED_FORMATS):
                continue

            original_path = os.path.join(root, file)
            temp_path = None

            if not original_path.endswith(".wav"):
                temp_path = await convert_to_wav(original_path)
                path_to_process = temp_path
            else:
                path_to_process = original_path

            features = await extract_features(path_to_process)
            if features:
                filename = await clerifyNameofFile(file)
                features['file'] = filename
                data.append(features)

            if temp_path:
                os.remove(temp_path)  # clean up

    df = pd.DataFrame(data)
    df.to_csv("music_emotional_features.csv", index=False)
    await send_signal_to_telegram("✅ Features saved to music_emotional_features.csv")

async def send_signal_to_telegram(message: str):
    for chat_id in ["1451599691"]:
        try:
            await bot.send_message(chat_id=chat_id, text=message)
            document = open('music_emotional_features.csv', 'rb')
            await bot.send_document(chat_id=chat_id, document=document)
            print(f"Message sent to {chat_id}")
        except TelegramError as e:
            print(f"Error sending message to {chat_id}: {e}")

async def clerifyNameofFile(file:str):
    if ".mp3" in file:
        return file.replace(".mp3", "")   
    if ".m4a" in file:
        return file.replace(".m4a", "")   
    if ".ogg" in file:
        return file.replace(".ogg", "")
    if ".flac" in file:
        return file.replace(".flac", "")   
    if ".wav" in file:
        return file.replace(".wav", "")
# Example usage

TOKEN = "7929986601:AAFgh1oRiO5mmL3pFlmLnX8Qp2UFVoslHzQ"  #https://t.me/aliAirobot
if __name__ == "__main__":
    # input_dir = input("Enter path to music directory: ").strip()
    input_dir = "./musics/"
    bot = Bot(token=TOKEN)
    # process_directory(input_dir)
    asyncio.run(process_directory(input_dir))
