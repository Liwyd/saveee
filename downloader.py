import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

# اطمینان از وجود پوشه ذخیره‌سازی
SAVE_PATH = "./musics"
os.makedirs(SAVE_PATH, exist_ok=True)

# URL لیست پخش
PLAYLIST_URL = "https://soundcloud.com/ali-k-877501694/sets/khfni"

# پارامترهای yt-dlp به صورت یک لیست
BASE_COMMAND = [
    "yt-dlp",
    "--ignore-errors",             # اگه فایلی مشکل داشت، اسکریپت متوقف نشه
    "--no-playlist",               # به‌صورت تکی دانلود می‌کنیم
    "-x", "--audio-format", "wav", # استخراج صوت به صورت wav
    "-o", os.path.join(SAVE_PATH, "%(title).100s.%(ext)s"), # نام‌گذاری امن
]

def get_playlist_tracks(url):
    try:
        result = subprocess.run(
            ["yt-dlp", "--flat-playlist", "-J", url],
            capture_output=True, text=True, check=True
        )
        import json
        info = json.loads(result.stdout)
        entries = info.get("entries", [])
        return [entry["url"] for entry in entries if "url" in entry]
    except Exception as e:
        print(f"[!] خطا در گرفتن لیست پخش: {e}")
        return []

def download_track(track_url):
    try:
        subprocess.run(BASE_COMMAND + [track_url], check=True)
        print(f"[✓] دانلود شد: {track_url}")
    except subprocess.CalledProcessError as e:
        print(f"[X] خطا در دانلود: {track_url} | {e}")

def download_playlist(url, max_threads=4):
    print("[*] گرفتن لیست آهنگ‌ها...")
    tracks = get_playlist_tracks(url)
    print(f"[+] {len(tracks)} ترک پیدا شد.")
    if not tracks:
        return

    print(f"[*] شروع دانلود با {max_threads} ترد...")
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        executor.map(download_track, tracks)

if __name__ == "__main__":
    download_playlist(PLAYLIST_URL, max_threads=4)
