import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from datasets import load_dataset
from tqdm import tqdm

# 1. Configuration
# ---------------------------------------------------------
IMAGES_DIR = "TinyImageNet/images"
MODEL_PATH = "TinyImageNet/best_model.pth" # Make sure this path matches your best saved model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Re-define Architecture (Must match training exactly)
# ---------------------------------------------------------
class TinyImageNetCNN(nn.Module):
    def __init__(self, num_classes=200):
        super(TinyImageNetCNN, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2) 
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# 3. Setup: Load Model & Create Mapping
# ---------------------------------------------------------
# A. Load the Mapping (String ID -> Integer Index)
print("Loading label mapping from dataset...")
# We only load the 'feature' info, not the whole dataset, so it's fast
ds = load_dataset('zh-plus/tiny-imagenet', split='valid')
# This list is ordered: index 0 is the first string, index 1 is the second...
class_names_list = ds.features['label'].names 
# Create a reverse dictionary: {'n01443537': 0, 'n01629819': 1, ...}
str_to_idx = {name: i for i, name in enumerate(class_names_list)}

# B. Load Model
print(f"Loading model from {MODEL_PATH}...")
model = TinyImageNetCNN(num_classes=200).to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval() # Freeze dropout/batchnorm

# C. Define Transforms (Must match Validation transforms)
stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

# 4. Run Evaluation Loop
# ---------------------------------------------------------
image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith('.jpg')]
print(f"Found {len(image_files)} images. Starting evaluation...")

correct = 0
total = 0
mistakes = [] # Keep track of what went wrong for debugging

for filename in tqdm(image_files):
    # 1. Parse True Label from filename (e.g., "n01443537.jpg" -> "n01443537")
    true_label_str = filename.split('.')[0]
    
    if true_label_str not in str_to_idx:
        print(f"Warning: Unknown label in filename {filename}")
        continue
        
    true_label_idx = str_to_idx[true_label_str]

    # 2. Load and Preprocess Image
    img_path = os.path.join(IMAGES_DIR, filename)
    image = Image.open(img_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(DEVICE) # Add batch dim

    # 3. Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        # Get the index with the highest score
        _, predicted_idx = torch.max(outputs, 1)
        predicted_idx = predicted_idx.item()

    # 4. Compare
    if predicted_idx == true_label_idx:
        correct += 1
    else:
        # Record mistake: (True Label Name, Predicted Label Name)
        pred_label_str = class_names_list[predicted_idx]
        mistakes.append((true_label_str, pred_label_str))
        
    total += 1

# 5. Report Results
# ---------------------------------------------------------
accuracy = 100 * correct / total
print("\n" + "="*30)
print(f"FINAL RESULTS")
print("="*30)
print(f"Images Tested: {total}")
print(f"Correct:       {correct}")
print(f"Accuracy:      {accuracy:.2f}%")
print("="*30)

if mistakes:
    print(f"\nFirst 5 Mistakes (True vs Predicted):")
    for truth, pred in mistakes[:5]:
        print(f" - Image was {truth}, Model thought it was {pred}")