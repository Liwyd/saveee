import os
import subprocess
#pip3 install yt-dlp

playlist_url = "https://soundcloud.com/ali-k-877501694/sets/khfni"
save_path = "./musics"
os.makedirs(save_path, exist_ok=True)


def download_soundcloud_playlist(url, path):
    try:
        subprocess.run([
            "scdl",
            "-l", url,
            "-o", path,
            "--onlymp3"
        ], check=True)
        print("[+] دانلود کامل شد.")
    except subprocess.CalledProcessError as e:
        print(f"[-] خطا در دانلود: {e}")

def download_with_ytdlp(url, path):
    try:
        subprocess.run([
            "yt-dlp",
            "-x", "--audio-format", "wav",
            "-o", os.path.join(path, "%(title)s.%(ext)s"),
            url
        ], check=True)
        print("[+] دانلود کامل شد.")
    except Exception as e:
        print(f"[-] خطا در دانلود یا تبدیل: {e}")

download_with_ytdlp(playlist_url, save_path)
