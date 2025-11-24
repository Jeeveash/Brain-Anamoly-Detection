import torch
from tqdm import tqdm
import requests
import os

def download_file(url, filename):
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()
    
    if total_size != 0 and progress_bar.n != total_size:
        print("ERROR: Failed to download the complete file")
    else:
        print(f"\nSuccessfully downloaded {filename}")

# Ensure the directory exists
os.makedirs("models/pretrained/efficientnet_b3", exist_ok=True)

# Download the model weights
download_file(
    "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b3_ra2-cf984f9c.pth",
    "models/pretrained/efficientnet_b3/model.pth"
)

print("\nModel weights downloaded successfully!")