import urllib.request
import os
import sys

url = "https://huggingface.co/datasets/JEFFREY-VERDIERE/Creditcard/resolve/main/creditcard.csv"
dest = os.path.join("data", "creditcard.csv")

if os.path.exists(dest) and os.path.getsize(dest) > 100000000:
    print(f"Dataset already exists at {dest} ({os.path.getsize(dest)} bytes)")
    sys.exit(0)

print(f"Downloading {url} to {dest}...")
def progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = (downloaded / total_size) * 100
        if block_num % 1000 == 0:
            print(f"Downloaded {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent:.1f}%)")

urllib.request.urlretrieve(url, dest, reporthook=progress_hook)
print(f"Done! Final file size: {os.path.getsize(dest)} bytes")
