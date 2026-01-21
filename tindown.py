import os
import requests
import zipfile
import io
from tqdm import tqdm

# Configuration
DATA_ROOT = './data'
URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
TARGET_DIR = os.path.join(DATA_ROOT, 'tiny-imagenet-200')

def download_and_unzip():
    os.makedirs(DATA_ROOT, exist_ok=True)
    
    # Check if already exists
    if os.path.exists(TARGET_DIR):
        print(f"Dataset already exists at {TARGET_DIR}")
        return

    print(f"Downloading TinyImageNet from {URL}...")
    response = requests.get(URL, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # Download with progress bar
    block_size = 1024 # 1 Kibibyte
    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    content = io.BytesIO()
    for data in response.iter_content(block_size):
        t.update(len(data))
        content.write(data)
    t.close()

    print("Extracting zip file...")
    with zipfile.ZipFile(content) as z:
        z.extractall(DATA_ROOT)
    print("Extraction complete.")

def format_val_folder():
    """
    Restructures ./tiny-imagenet-200/val so ImageFolder can read it.
    Raw structure:      val/images/val_0.JPEG
    Required structure: val/n01443537/val_0.JPEG
    """
    val_dir = os.path.join(TARGET_DIR, 'val')
    img_dir = os.path.join(val_dir, 'images')
    annot_file = os.path.join(val_dir, 'val_annotations.txt')

    if not os.path.exists(img_dir):
        print("Validation folder already formatted or missing.")
        return

    print("Restructuring validation set for PyTorch ImageFolder...")
    
    # 1. Read annotations to map Image -> Class
    # Format: val_0.JPEG <tab> n01443537 <tab> ...
    with open(annot_file, 'r') as f:
        lines = f.readlines()

    val_img_dict = {}
    for line in lines:
        parts = line.strip().split('\t')
        val_img_dict[parts[0]] = parts[1]

    # 2. Create subfolders and move images
    for img_filename, class_id in tqdm(val_img_dict.items()):
        # Create class folder if it doesn't exist
        class_dir = os.path.join(val_dir, class_id)
        os.makedirs(class_dir, exist_ok=True)
        
        # Source and Destination paths
        src = os.path.join(img_dir, img_filename)
        dst = os.path.join(class_dir, img_filename)
        
        # Move
        if os.path.exists(src):
            os.rename(src, dst)
            
    # 3. Cleanup
    if os.path.exists(img_dir) and not os.listdir(img_dir):
        os.rmdir(img_dir)
    
    print("Validation set formatted successfully.")

if __name__ == '__main__':
    download_and_unzip()
    format_val_folder()