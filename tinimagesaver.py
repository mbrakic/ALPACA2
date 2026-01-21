import os
import random
from datasets import load_dataset
from tqdm import tqdm

# 1. Setup
# ---------------------------------------------------------
OUTPUT_DIR = "TinyImageNet/images"
TARGET_COUNT = 200  # Tiny ImageNet has exactly 200 classes

# Create the directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created folder: {OUTPUT_DIR}")

# 2. Load the Dataset
# ---------------------------------------------------------
print("Loading Tiny ImageNet...")
dataset = load_dataset('zh-plus/tiny-imagenet')

# We use the validation set because it's smaller and guaranteed to have all classes
# (The training set is huge and shuffling it takes longer)
data_source = dataset['valid']

# 3. Logic to pick 1 unique image per class
# ---------------------------------------------------------
print("Selecting 1 random image for each of the 200 classes...")

saved_classes = set()
images_to_save = []

# To ensure randomness, we shuffle the indices of the dataset first
indices = list(range(len(data_source)))
random.shuffle(indices)

for idx in tqdm(indices):
    example = data_source[idx]
    label_int = example['label']
    
    # If we haven't seen this class yet, grab it
    if label_int not in saved_classes:
        # Get the string label (e.g., 'n01443537') from the features
        label_str = data_source.features['label'].int2str(label_int)
        
        images_to_save.append({
            'image': example['image'],
            'filename': f"{label_str}.jpg", # Using the WordNet ID as name
            'class_id': label_int
        })
        
        saved_classes.add(label_int)
    
    # Stop once we have all 200 classes
    if len(saved_classes) == TARGET_COUNT:
        break

# 4. Save to Disk
# ---------------------------------------------------------
print(f"Saving {len(images_to_save)} images to {OUTPUT_DIR}...")

for item in images_to_save:
    # Convert RGBA to RGB if necessary (some formats might be tricky)
    img = item['image'].convert("RGB")
    
    save_path = os.path.join(OUTPUT_DIR, item['filename'])
    img.save(save_path)

print("Done! Check the folder.")