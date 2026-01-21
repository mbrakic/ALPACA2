import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 1. Configuration
BATCH_SIZE = 256
EPOCHS = 50 
LEARNING_RATE = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = './CIFAR100'

# Create the directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Working directory: {os.path.abspath(SAVE_DIR)}")
print(f"Training on: {DEVICE}")

# 2. Data Preparation (SOTA Method: Heavy Augmentation & Normalization)
# CIFAR-100 mean and std deviation
stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*stats, inplace=True)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

# Download and load data into the 'CIFAR100' folder
print("Downloading/Loading Data...")
train_data = torchvision.datasets.CIFAR100(root=SAVE_DIR, train=True, 
                                           download=True, transform=train_transform)
test_data = torchvision.datasets.CIFAR100(root=SAVE_DIR, train=False, 
                                          download=True, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=2, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE*2, shuffle=False, 
                         num_workers=2, pin_memory=True)

# 3. Model Definition (Simple but Deep CNN)
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # A simple block helper function
        def conv_block(in_channels, out_channels, pool=False):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
                nn.BatchNorm2d(out_channels), 
                nn.ReLU(inplace=True)
            ]
            if pool: layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)
        
        # Network Body
        self.network = nn.Sequential(
            conv_block(3, 64),
            conv_block(64, 128, pool=True),  # 16x16
            conv_block(128, 256, pool=True), # 8x8
            conv_block(256, 512, pool=True), # 4x4
            nn.AdaptiveAvgPool2d(1),         # 1x1
            nn.Flatten(),
            nn.Dropout(0.2), # Regularization
            nn.Linear(512, 100) # 100 classes for CIFAR-100
        )
        
    def forward(self, x):
        return self.network(x)

model = SimpleCNN().to(DEVICE)

# 4. Training Setup (SOTA Methods: AdamW + OneCycleLR)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# OneCycleLR allows very fast convergence by fluctuating the learning rate
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, 
                                          steps_per_epoch=len(train_loader), 
                                          epochs=EPOCHS)

# 5. Training Loop
print("Starting Training...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Step optimizer and scheduler
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    train_acc = 100. * correct / total
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
            
    val_acc = 100. * val_correct / val_total
    
    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Loss: {running_loss/len(train_loader):.4f} | "
          f"Train Acc: {train_acc:.2f}% | "
          f"Val Acc: {val_acc:.2f}% | "
          f"LR: {current_lr:.6f}")

# 6. Save the model
save_path = os.path.join(SAVE_DIR, 'cifar100_weights.pth')
torch.save(model.state_dict(), save_path)
print(f"\nModel weights saved to: {save_path}")
print(f"Dataset location: {os.path.join(SAVE_DIR, 'cifar-100-python')}")