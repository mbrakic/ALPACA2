import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm

# 1. Setup Configuration
# ---------------------------------------------------------
FOLDER_NAME = "TinyImageNet"
BATCH_SIZE = 256
MAX_LR = 0.01  # Peak learning rate for OneCycle
EPOCHS = 30    # Increased to allow convergence with augmentations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(FOLDER_NAME):
    os.makedirs(FOLDER_NAME)

# 2. Define The Custom Model (Same VGG-Style Architecture)
# ---------------------------------------------------------
class TinyImageNetCNN(nn.Module):
    def __init__(self, num_classes=200):
        super(TinyImageNetCNN, self).__init__()
        
        # Block 1: 64x64 -> 32x32
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) 
        )
        
        # Block 2: 32x32 -> 16x16
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 3: 16x16 -> 8x8
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 4: 8x8 -> 4x4
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

# 3. Enhanced Data Augmentation
# ---------------------------------------------------------
print("Loading Dataset...")
dataset = load_dataset('zh-plus/tiny-imagenet')

stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

# STRONGER AUGMENTATION HERE
train_transform = transforms.Compose([
    transforms.RandomCrop(64, padding=4),       # Shifts the image slightly
    transforms.RandomHorizontalFlip(),          # Standard mirror
    transforms.RandomRotation(15),              # Rotates up to 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Changes lighting
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

def preprocess_train(examples):
    examples['pixel_values'] = [train_transform(img.convert("RGB")) for img in examples['image']]
    return examples

def preprocess_val(examples):
    examples['pixel_values'] = [val_transform(img.convert("RGB")) for img in examples['image']]
    return examples

dataset['train'].set_transform(preprocess_train)
dataset['valid'].set_transform(preprocess_val)

def collate_fn(examples):
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
    labels = torch.tensor([ex["label"] for ex in examples])
    return pixel_values, labels

train_loader = DataLoader(dataset['train'], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)
val_loader = DataLoader(dataset['valid'], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)

# 4. Init Model, Optimizer, and Scheduler
# ---------------------------------------------------------
model = TinyImageNetCNN(num_classes=200).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=MAX_LR) # Initial LR doesn't matter much with OneCycle

# Define Scheduler
# steps_per_epoch is needed so the scheduler knows when to update
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=MAX_LR, 
    epochs=EPOCHS, 
    steps_per_epoch=len(train_loader)
)

# 5. Training Loop
# ---------------------------------------------------------
print(f"Starting Pro training on {DEVICE} for {EPOCHS} epochs...")

best_acc = 0.0

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    
    for imgs, lbls in train_pbar:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        scheduler.step() # Step the scheduler every BATCH, not every epoch
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += lbls.size(0)
        correct += (predicted == lbls).sum().item()
        
        # Display current LR in progress bar
        current_lr = scheduler.get_last_lr()[0]
        train_pbar.set_postfix(loss=loss.item(), lr=f"{current_lr:.5f}")

    train_acc = 100 * correct / total
    
    # Validate
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for imgs, lbls in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Valid]"):
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            val_total += lbls.size(0)
            val_correct += (predicted == lbls).sum().item()
            
    val_acc = 100 * val_correct / val_total
    print(f"Epoch {epoch+1} Results -> Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
    
    # Save Best Model Only
    if val_acc > best_acc:
        best_acc = val_acc
        save_path = os.path.join(FOLDER_NAME, "best_model.pth")
        torch.save(model.state_dict(), save_path)
        print(f"New Best Model Saved! ({val_acc:.2f}%)")

print(f"Training Complete. Best Validation Accuracy: {best_acc:.2f}%")