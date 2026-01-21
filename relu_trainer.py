import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import the model generator from your installed library
from lipMIP.relu_nets import ReLUNet

def create_network(net_dimensions):
    # network dimensions is an array like [784, 256, 128, 10]
    network = ReLUNet(net_dimensions, bias=True)
    input_dimension = net_dimensions[0] 
    return network, input_dimension 

# --- Data Loading Helpers ---

def get_mnist_loaders(root, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    )

def get_cifar_loaders(root, batch_size):
    # CIFAR-100 Stats
    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    
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
    
    train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    )

def get_tinyimagenet_loaders(root, batch_size):
    tiny_root = os.path.join(root, 'tiny-imagenet-200')
    
    if not os.path.exists(os.path.join(tiny_root, 'train')):
        print(f"\n[WARNING] TinyImageNet train folder not found at {tiny_root}/train.")
        return None, None

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

    try:
        train_set = torchvision.datasets.ImageFolder(root=os.path.join(tiny_root, 'train'), transform=transform_train)
        # Using 'train' as test if val is not formatted, to prevent crash. Change to 'val' if data is ready.
        test_set = torchvision.datasets.ImageFolder(root=os.path.join(tiny_root, 'train'), transform=transform_test)
        
        return (
            DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4),
            DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
        )
    except Exception as e:
        print(f"[ERROR] Loading TinyImageNet failed: {e}")
        return None, None

def evaluate_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

# --- Main Training Function ---

def train_model(model_arch, dataset_name, save_name, **kwargs):
    # 1. Unpack arguments
    epochs = kwargs.get('EPOCHS', 10)
    batch_size = kwargs.get('BATCH_SIZE', 128)
    lr = kwargs.get('LEARNING_RATE', 0.001)
    device = kwargs.get('DEVICE', 'cpu')
    data_root = kwargs.get('DATA_ROOT', './data')
    save_dir = kwargs.get('SAVE_DIR', './models')

    print(f"\n{'='*60}")
    print(f"Processing: {save_name} | Dataset: {dataset_name} | Epochs: {epochs}")
    print(f"{'='*60}")

    # 2. Get Data
    if dataset_name == 'MNIST':
        train_loader, test_loader = get_mnist_loaders(data_root, batch_size)
    elif dataset_name == 'CIFAR':
        train_loader, test_loader = get_cifar_loaders(data_root, batch_size)
    elif dataset_name == 'TinyImageNet':
        train_loader, test_loader = get_tinyimagenet_loaders(data_root, batch_size)
        if train_loader is None: return

    # 3. Instantiate model
    model, _ = create_network(model_arch) 
    model = model.to(device)

    # 4. Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs//2, epochs*3//4], gamma=0.1)

    # 5. Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f"{running_loss/len(train_loader):.4f}", 'acc': f"{100.*correct/total:.2f}%"})

        scheduler.step()
        
        val_acc = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch+1}: Train Acc: {100.*correct/total:.2f}% | Val Acc: {val_acc:.2f}%")

    # 6. Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{save_name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to: {save_path}")

# --- Execution Block ---

if __name__ == "__main__":
    
    # Base Config
    train_args = {
        'BATCH_SIZE': 128,
        'EPOCHS': 10,
        'LEARNING_RATE': 0.001,
        'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'DATA_ROOT': './data'
    }

    print(f"Training on device: {train_args['DEVICE']}")

    # Dataset Configs
    dataset_configs = {
        # 'MNIST': (784, 10),
        # 'CIFAR': (3072, 100),
        'TinyImageNet': (12288, 200)
    }

    # Architectures
    arch_styles = {
        'small':  [256, 128],
        'medium': [512, 256, 128],
        'large':  [1024, 512, 256, 128]
    }

    # Epochs per dataset
    epoch_map = {
        'MNIST': 5, 
        'CIFAR': 40, 
        'TinyImageNet': 50
    }

    # Save directories
    dirs = {
        'MNIST': './MNIST_Models',
        'CIFAR': './CIFAR_Models',
        'TinyImageNet': './TinyImageNet_Models'
    }

    # Main Loop
    for dataset_name, (input_dim, output_dim) in dataset_configs.items():
        print(f"\n--- Preparing models for {dataset_name} ---")
        
        # Update dynamic args
        train_args['EPOCHS'] = epoch_map[dataset_name]
        train_args['SAVE_DIR'] = dirs[dataset_name]

        for size_name, hidden_layers in arch_styles.items():
            full_architecture = [input_dim] + hidden_layers + [output_dim]
            save_name = f"{dataset_name}_{size_name}"
            
            print(f"Training {save_name} | Arch: {full_architecture}")
            
            train_model(
                model_arch=full_architecture,
                dataset_name=dataset_name,
                save_name=save_name,
                **train_args 
            )

    print("\nAll models trained successfully.")

    for dataset_name, (input_dim, output_dim) in dataset_configs.items():
        save_dir = dirs[dataset_name]
        os.makedirs(save_dir, exist_ok=True)
        
        for size_name, hidden_layers in arch_styles.items():
            # Reconstruct the full architecture list
            full_architecture = [input_dim] + hidden_layers + [output_dim]
            
            # Define the dictionary and filename
            arch_dict = {'architecture': full_architecture}
            filename = f"{dataset_name}_{size_name}_architecture.pt"
            save_path = os.path.join(save_dir, filename)
            
            # Save
            torch.save(arch_dict, save_path)
            print(f"[{dataset_name}] Saved arch to: {save_path} -> {full_architecture}")

    print("Architectures saved.")

    # Define the stats and directories
    normalization_info = {
        # Dataset Name: (Save Dir, Filename, Stats Tuple)
        'MNIST': (
            './MNIST_Models', 
            'mnist_stat.pt', 
            ((0.1307,), (0.3081,))
        ),
        'CIFAR': (
            './CIFAR_Models', 
            'cifar100_stat.pt', 
            ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ),
        'TinyImageNet': (
            './TinyImageNet_Models', 
            'tinyimagenet_stat.pt', 
            ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        )
    }

    print("--- Saving Normalization Statistics ---")

    for dataset, (save_dir, filename, stats) in normalization_info.items():
        # Ensure directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Define full path
        save_path = os.path.join(save_dir, filename)
        
        # Save using torch
        torch.save(stats, save_path)
        print(f"[{dataset}] Saved stats to: {save_path}")

    print("Normalizations saved.")