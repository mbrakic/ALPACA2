import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# 1. Re-define the Architecture (MUST match training exactly)
# -----------------------------------------------------------
class TinyImageNetCNN(nn.Module):
    def __init__(self, num_classes=200):
        super(TinyImageNetCNN, self).__init__()
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) 
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Classifier
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
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

# 2. Initialize and Load Weights
# -----------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyImageNetCNN(num_classes=200).to(DEVICE)

# Path to your saved file
WEIGHTS_PATH = "TinyImageNet/best_model.pth"

if os.path.exists(WEIGHTS_PATH):
    print(f"Loading weights from {WEIGHTS_PATH}...")
    
    # map_location ensures safe loading if you switch from GPU (training) to CPU (inference)
    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    
    # Load the parameters into the model
    model.load_state_dict(state_dict)
    print("Weights loaded successfully!")
else:
    print("Error: Weights file not found.")

# 3. Set to Evaluation Mode (CRITICAL)
# -----------------------------------------------------------
# This freezes Dropout and Batch Norm layers. 
# If you skip this, your predictions will be garbage.
model.eval()

# 4. (Optional) Run on a sample image
# -----------------------------------------------------------
def predict_image(image_path):
    # Same transforms as validation
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform = transforms.Compose([
        transforms.Resize((64, 64)), # Ensure size is correct
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    
    # Open image
    try:
        img = Image.open(image_path).convert('RGB')
    except:
        print("Could not open image.")
        return

    # Preprocess
    img_tensor = transform(img).unsqueeze(0).to(DEVICE) # Add batch dimension (1, 3, 64, 64)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        prob, class_idx = torch.max(probabilities, 0)
    
    print(f"Predicted Class Index: {class_idx.item()}")
    print(f"Confidence: {prob.item()*100:.2f}%")

# Create a dummy image to test the script
predict_image("path/to/your/test_image.jpg")