import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import the model definitions from your provided script
import models

# --- Configuration ---
BATCH_SIZE = 128
EPOCHS = 50  # Adjust as needed (e.g., 50-100 for better convergence)
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = './data'

# Output directories
DIRS = {
    'MNIST': './MNIST',
    'CIFAR': './CIFAR',
    'TinyImageNet': './TinyImageNet'
}

for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

# --- Data Loading Helpers ---

def get_mnist_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = torchvision.datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)
    return (
        DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
        DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    )

def get_cifar_loaders():
    # Standard CIFAR-10 normalization
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    train_set = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=False, download=True, transform=transform_test)
    return (
        DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
        DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    )

def get_tinyimagenet_loaders():
    # TinyImageNet is not built-in. Assumes structure: ./data/tiny-imagenet-200/train and /val
    # Images are 64x64
    tiny_root = os.path.join(DATA_ROOT, 'tiny-imagenet-200')
    
    if not os.path.exists(tiny_root):
        print(f"\n[WARNING] TinyImageNet dataset not found at {tiny_root}.")
        print("Please download it (e.g., http://cs231n.stanford.edu/tiny-imagenet-200.zip) and unzip it into ./data/")
        return None, None

    # Mean/Std for ImageNet
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    # Using ImageFolder. 
    # Note: TinyImageNet val folder structure often needs formatting to work with ImageFolder directly.
    # Here we assume standard structure (class folders inside train/val).
    try:
        train_set = torchvision.datasets.ImageFolder(root=os.path.join(tiny_root, 'train'), transform=transform_train)
        test_set = torchvision.datasets.ImageFolder(root=os.path.join(tiny_root, 'val'), transform=transform_test)
        return (
            DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
            DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        )
    except Exception as e:
        print(f"[ERROR] Loading TinyImageNet failed: {e}")
        return None, None

# --- Training Engine ---

def train_model(model_func, dataset_name, save_name, **model_kwargs):
    print(f"\n{'='*60}")
    print(f"Processing Model: {save_name} on {dataset_name}")
    print(f"{'='*60}")

    # 1. Get Data
    if dataset_name == 'MNIST':
        train_loader, test_loader = get_mnist_loaders()
        in_ch, in_dim = 1, 28
    elif dataset_name == 'CIFAR':
        train_loader, test_loader = get_cifar_loaders()
        in_ch, in_dim = 3, 32
    elif dataset_name == 'TinyImageNet':
        train_loader, test_loader = get_tinyimagenet_loaders()
        in_ch, in_dim = 3, 64
        if train_loader is None: 
            print("Skipping due to missing data.")
            return

    # 2. Instantiate Model
    # We pass in_ch and in_dim explicitly to handle dimension matching
    try:
        model = model_func(in_ch=in_ch, in_dim=in_dim, **model_kwargs).to(DEVICE)
    except TypeError:
        # Fallback if model doesn't accept kwargs or specific args
        model = model_func().to(DEVICE)

    # 3. Setup Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[EPOCHS//2, EPOCHS*3//4], gamma=0.1)

    # 4. Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Use TQDM for progress bar if desired, else standard loop
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/len(train_loader), 'acc': 100.*correct/total})

        scheduler.step()
        
        # Validation (every epoch)
        val_acc = evaluate_model(model, test_loader)
        print(f"Epoch {epoch+1}: Train Acc: {100.*correct/total:.2f}% | Val Acc: {val_acc:.2f}%")

    # 5. Save
    save_path = os.path.join(DIRS[dataset_name], f"{save_name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to: {save_path}")

def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

# --- Main Execution List ---

if __name__ == "__main__":
    print(f"Training on device: {DEVICE}")

    # List of models to process
    # Format: (Function, Dataset_Type, Filename_to_save, {Extra Args})
    
    # 1. MNIST Models
    # train_model(models.mnist_mlp_3layer, 'MNIST', 'mnist_mlp_3layer')
    # train_model(models.mnist_cnn_4layer, 'MNIST', 'mnist_cnn_4layer')
    # train_model(models.mnist_cnn_4layer_8, 'MNIST', 'mnist_cnn_4layer_8')
    
    # 2. CIFAR-10 Models
    # Note: These functions default to in_dim=32, which matches CIFAR
    # train_model(models.cnn_4layer_stride1_padding0, 'CIFAR', 'cnn_4layer_stride1_padding0')
    # train_model(models.cnn_4layer_stride1_padding0_demo, 'CIFAR', 'cnn_4layer_stride1_padding0_demo')
    # train_model(models.cnn_6layer_stride1_padding0, 'CIFAR', 'cnn_6layer_stride1_padding0')
    
    # 3. TinyImageNet Models
    # Note: TinyImageNet is 64x64. We pass in_dim=64 to ensure Linear layers are sized correctly.
    # The models outputs 200 classes (hardcoded in models.py), matching TinyImageNet.
    train_model(models.cnn_4layer_stride2_imagenet, 'TinyImageNet', 'cnn_4layer_stride2_imagenet', in_dim=64)
    train_model(models.cnn_6layer_stride2_imagenet, 'TinyImageNet', 'cnn_6layer_stride2_imagenet', in_dim=64)

    print("\nAll requested models processed.")