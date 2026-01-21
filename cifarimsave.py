import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

# 1. Initialize the CIFAR-10 Test Dataset
# Use ToTensor() so the image is a PyTorch tensor in range [0, 1]
test_dataset = torchvision.datasets.CIFAR10(
            root='./data', 
                train=False, 
                    download=True, 
                        transform=transforms.ToTensor()
                        )

# 2. Grab a single image by index (e.g., the 5th image in the test set)
# test_dataset[index] returns a tuple: (image_tensor, label_index)
img_tensor, label_idx = test_dataset[4]

# 3. Save it to a path
save_image(img_tensor, "CIFAR/cifar_test_image.png")
