import torch
import torchvision
import torchvision.transforms as transforms
import os
from PIL import Image

# --- Configuration ---
DATA_ROOT = './data'
SAVE_DIRS = {
    'MNIST':        './MNIST_Models',
    'CIFAR':        './CIFAR_Models',
    'TinyImageNet': './TinyImageNet_Models'
}

# Image dimensions for verification/generation
# (Though we will pull from actual datasets to be useful)
DIMENSIONS = {
    'MNIST': (1, 28, 28),
    'CIFAR': (3, 32, 32),
    'TinyImageNet': (3, 64, 64)
}

def get_test_dataset(name):
    """Returns the raw test dataset (PIL images if possible, or tensors)."""
    if name == 'MNIST':
        # MNIST raw is PIL
        return torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=True)
    
    elif name == 'CIFAR':
        # CIFAR100 raw is PIL
        return torchvision.datasets.CIFAR100(root=DATA_ROOT, train=False, download=True)
    
    elif name == 'TinyImageNet':
        # TinyImageNet requires the val folder we set up previously
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
                # Fallback: Generate random noise if dataset missing (e.g. TinyImageNet not downloaded)
                c, h, w = DIMENSIONS[dataset_name]
                # Create random tensor and save as image
                rand_tensor = torch.rand(c, h, w)
                img = transforms.ToPILImage()(rand_tensor)
                img.save(save_path)

            # Update Dictionary
            # We store the filename. The analysis script knows the folder path.
            test_suite_dictionary[dataset_name][img_key] = img_filename
            
            print(f"    Saved {img_filename}")

    return test_suite_dictionary

if __name__ == "__main__":
    # 1. Generate Images
    full_dict = save_images_and_build_dict(10)
    
    # 2. Save the Dictionary itself
    dict_path = 'test_suite_dictionary.pt'
    torch.save(full_dict, dict_path)
    
    print(f"\n[Success] Test suite dictionary saved to {dict_path}")
    print("Structure Preview:")
    print(full_dict['MNIST']['image_0']) # Should print 'test_image_0.png'