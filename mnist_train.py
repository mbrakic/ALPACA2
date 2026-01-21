import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 1. Configuration
BATCH_SIZE = 256
EPOCHS = 10  # Reduced: MNIST converges very quickly (99%+ in <10 epochs)
LEARNING_RATE = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = './MNIST'

# Create the directory
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Working directory: {os.path.abspath(SAVE_DIR)}")
print(f"Training on: {DEVICE}")

# 2. Data Preparation
# MNIST mean and std deviation (grayscale)
stats = ((0.1307,), (0.3081,))

train_transform = transforms.Compose([
    transforms.RandomCrop(28, padding=4),     # Shift content slightly
    transforms.RandomRotation(10),            # Rotate +/- 10 degrees (Better than flip for digits)
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

# Download and load data
print("Downloading/Loading MNIST...")
train_data = torchvision.datasets.MNIST(root=SAVE_DIR, train=True, 
                                        download=True, transform=train_transform)
test_data = torchvision.datasets.MNIST(root=SAVE_DIR, train=False, 
                                       download=True, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=2, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE*2, shuffle=False, 
                         num_workers=2, pin_memory=True)

# 3. Model Definition (Smaller Architecture for Efficiency)
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        def conv_block(in_channels, out_channels, pool=False):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
                nn.BatchNorm2d(out_channels), 
                nn.ReLU(inplace=True)
            ]
            if pool: layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)
        
        self.network = nn.Sequential(
            # Input: 1 x 28 x 28 (Grayscale)
            conv_block(1, 32),              # 32 x 28 x 28
            conv_block(32, 64, pool=True),  # 64 x 14 x 14
            conv_block(64, 128, pool=True), # 128 x 7 x 7
            
            # Classifier
            nn.AdaptiveAvgPool2d(1),        # 128 x 1 x 1
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)              # 10 Classes for MNIST
        )
        
    def forward(self, x):
        return self.network(x)

model = SmallCNN().to(DEVICE)

# 4. Training Setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, 
                                          steps_per_epoch=len(train_loader), 
                                          epochs=EPOCHS)

# 5. Training Loop
print(f"Starting Training for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    train_acc = 100. * correct / total
    
    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
            
    val_acc = 100. * val_correct / val_total
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Loss: {running_loss/len(train_loader):.4f} | "
          f"Train Acc: {train_acc:.2f}% | "
          f"Val Acc: {val_acc:.2f}% | "
          f"LR: {current_lr:.6f}")

# 6. Save the model
save_path = os.path.join(SAVE_DIR, 'mnist_weights.pth')
torch.save(model.state_dict(), save_path)
print(f"\nModel weights saved to: {save_path}")