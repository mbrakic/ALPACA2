import torch
import torchvision
import torchvision.transforms as transforms
import os
from PIL import Image

# --- Configuration ---
DATA_ROOT = './data'
SAVE_DIRS = {
    'MNIST':        './MNIST',
    'CIFAR':        './CIFAR',
    'TinyImageNet': './TinyImageNet'
}

# Standard Normalization Stats (Mean, Std)
# Using standard pre-computed values for these datasets
DATASET_STATS = {
    'MNIST': (
        (0.1307,), 
        (0.3081,)
    ),
    'CIFAR': (
        (0.4914, 0.4822, 0.4465), 
        (0.2023, 0.1994, 0.2010)
    ),
    'TinyImageNet': (
        (0.485, 0.456, 0.406), 
        (0.229, 0.224, 0.225)
    )
}

# Image dimensions for verification/generation
DIMENSIONS = {
    'MNIST': (1, 28, 28),
    'CIFAR': (3, 32, 32),
    'TinyImageNet': (3, 64, 64)
}

def get_test_dataset(name):
    """Returns the raw test dataset (PIL images if possible, or tensors)."""
    if name == 'MNIST':
        return torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=True)
    
    elif name == 'CIFAR':
        return torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True)
    
    elif name == 'TinyImageNet':
        tiny_root = os.path.join(DATA_ROOT, 'tiny-imagenet-200', 'val')
        if not os.path.exists(tiny_root):
            print("[!] TinyImageNet val folder not found. Generating Random Noise images instead.")
            return None
        return torchvision.datasets.ImageFolder(root=tiny_root)
    
    return None

def save_images_and_build_dict(num_images=10):
    test_suite_dictionary = {}
    
    for dataset_name, save_dir in SAVE_DIRS.items():
        print(f"--- Processing {dataset_name} ---")
        os.makedirs(save_dir, exist_ok=True)
        
        # --- NEW CODE: Save Normalization Stats ---
        if dataset_name in DATASET_STATS:
            stats = DATASET_STATS[dataset_name]
            # Construct filename: e.g., 'mnist_stat.pt'
            stat_filename = f"{dataset_name.lower()}_stat.pt"
            stat_path = os.path.join(save_dir, stat_filename)
            
            torch.save(stats, stat_path)
            print(f"    Saved stats to {stat_filename} {stats}")
        # ------------------------------------------

        # Initialize sub-dict for this dataset
        test_suite_dictionary[dataset_name] = {}
        
        # Get Data
        dataset = get_test_dataset(dataset_name)
        
        for i in range(num_images):
            img_filename = f"test_image_{i}.png"
            save_path = os.path.join(save_dir, img_filename)
            img_key = f"image_{i}"
            
            # Save Image
            if dataset is not None:
                # Grab image from dataset (dataset[i] returns (image, label))
                img, label = dataset[i]
                
                # Check if it's already a PIL image, if not convert
                if not isinstance(img, Image.Image):
                    img = transforms.ToPILImage()(img)
                
                img.save(save_path)
            else:
                # Fallback: Generate random noise if dataset missing
                c, h, w = DIMENSIONS[dataset_name]
                rand_tensor = torch.rand(c, h, w)
                img = transforms.ToPILImage()(rand_tensor)
                img.save(save_path)

            # Update Dictionary
            test_suite_dictionary[dataset_name][img_key] = img_filename
            
            print(f"    Saved {img_filename}")

    return test_suite_dictionary

if __name__ == "__main__":
    # 1. Generate Images and Save Stats
    full_dict = save_images_and_build_dict(10)
    
    # 2. Save the Dictionary itself
    dict_path = 'test_suite_dictionary_cnn.pt'
    torch.save(full_dict, dict_path)
    
    print(f"\n[Success] Test suite dictionary saved to {dict_path}")
    print("Structure Preview:")
    print(full_dict['MNIST']['image_0'])