import torch
import os
from torchvision import models
from PIL import Image

# --- CONFIGURATION ---
folder_name = "ImageNet"
weights_filename = "resnet50_weights.pth"
image_filename = "tench.JPEG" # Make sure this file exists!

# Define paths
current_dir = os.getcwd()
save_dir = os.path.join(current_dir, folder_name)
weights_path = os.path.join(save_dir, weights_filename)
image_path = os.path.join(save_dir, image_filename)

# Ensure the folder exists
os.makedirs(save_dir, exist_ok=True)

# --- STEP 1: DOWNLOAD WEIGHTS TO LOCAL FOLDER ---
# We get the URL from the official Weights object
weights_enum = models.ResNet50_Weights.DEFAULT
url = weights_enum.url

if not os.path.exists(weights_path):
    print(f"Weights not found at {weights_path}")
    print(f"Downloading from {url}...")
    
    # torch.hub helps us download safely to a specific path
    torch.hub.download_url_to_file(url, weights_path)
    print("Download complete.")
else:
    print(f"Found local weights at {weights_path}")

# --- STEP 2: LOAD MODEL FROM LOCAL FILE ---

# A. Initialize an "empty" architecture (weights=None)
# We must turn off the default internet download behavior
model = models.resnet50(weights=None)

# B. Load the state dictionary from your local file
state_dict = torch.load(weights_path)

# C. Apply the weights to the model
model.load_state_dict(state_dict)
model.eval() # Freeze for inference

# --- STEP 3: PROCESS IMAGE & RUN ---

# Use the standard transforms associated with these weights
# (This handles the resize to 224x224 and normalization for you)
preprocess = weights_enum.transforms()

try:
    # Load Image
    img = Image.open(image_path).convert('RGB')
    
    # Preprocess (Add batch dimension -> 1, 3, 224, 224)
    batch = preprocess(img).unsqueeze(0)

    # Run Inference
    print("Running inference...")
    with torch.no_grad():
        predictions = model(batch).squeeze(0).softmax(0)

    # Decode the top result
    class_id = predictions.argmax().item()
    score = predictions[class_id].item()
    category_name = weights_enum.meta["categories"][class_id]

    print("--- RESULTS ---")
    print(f"Image: {image_filename}")
    print(f"Prediction: {category_name}")
    print(f"Confidence: {100 * score:.2f}%")

except FileNotFoundError:
    print(f"ERROR: Could not find image file at: {image_path}")
    print("Please place an image named 'test_image.jpg' in the ImageNet folder.")