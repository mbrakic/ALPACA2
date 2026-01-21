import os
import warnings
from argparse import ArgumentParser
from LBDN.train import *
from LBDN.evaluate import *
from eclipsE import eclipsE as eclipse 
from eclipsE_fast import eclipsE_fast as eclipse_fast
# warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Config:
    def __init__(self):

        self.seed = 123
        self.scale = 'small'
        self.layer = 'Sandwich'
        self.gamma = 1. 
        self.epochs = 100 

        self.mode = 'eval' 
        self.model = 'KWL' 
        self.dataset = 'cifar10' 
        self.loss = 'multimargin' 

        self.lr = 0.01 
        self.root_dir = 'LBDN/saved_models'
        self.train_batch_size = 256
        self.test_batch_size = 256 
        self.num_workers = 4 
        self.LLN = False 
        self.normalized = False
        self.cert_acc = False 

        self.lip_batch_size = 64
        self.print_freq = 10
        self.save_freq = 5
        
        self.in_channels = 3
        self.img_size = 32
        self.num_classes = 10

        self.width = {
            'small': 1,
            'medium': 2,
            'large': 4
        }[self.scale]



        if self.gamma is None:
            self.train_dir = f"{self.root_dir}_seed{self.seed}/{self.dataset}/{self.model}-{self.layer}-{self.scale}"
        elif self.LLN:
            self.train_dir = f"{self.root_dir}_seed{self.seed}/{self.dataset}/{self.model}-{self.layer}-{self.scale}-LLN-gamma{self.gamma:.1f}"
        else:
            self.train_dir = f"{self.root_dir}_seed{self.seed}/{self.dataset}/{self.model}-{self.layer}-{self.scale}-gamma{self.gamma:.1f}"

        os.makedirs("./data", exist_ok=True)
        os.makedirs(self.train_dir, exist_ok=True)

config = Config() 
# evaluate(config) 

model, testLoader, trainLoader = prepare_model(config) 
model.to(device)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

print("\nRunning accuracy test...")
correct = 0
total = 0

with torch.no_grad():
    # Loop through 25 batches (25 * 4 images/batch = 100 images)
    for i, data in enumerate(testLoader):
        if i >= 40:
            break
        
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'--> Accuracy on {total} test images: {accuracy:.2f} %')

def extract_and_format_weights(model):
    """
    Extracts weights from a PyTorch model's state_dict by identifying
    parameters that end with the '.weight' suffix. This method explicitly
    skips biases, psis, alphas, and any other non-weight parameters.

    Args:
        model (torch.nn.Module): The PyTorch model with loaded weights.

    Returns:
        dict: A dictionary where keys are 'w1', 'w2', etc., and values are
              the corresponding weight layers as reshaped NumPy arrays.
    """
    print("Extracting and formatting weights...")
    
    formatted_weights = {}
    layer_index = 1

    state_dict = model.state_dict()
    for name, params in state_dict.items():
        # Only process parameters that are weights.
        # This filters out biases, psis, alphas, and other parameters.
        if name.endswith('.weight'):
            print(f"  - Extracting layer: '{name}' with shape {params.shape}")

            # Convert the PyTorch tensor to a NumPy array
            numpy_weights = params.cpu().detach().numpy()

            # Reshape convolutional layers (which are > 2D) to be 2D matrices
            if numpy_weights.ndim > 2:
                reshaped_weights = numpy_weights.reshape(numpy_weights.shape[0], -1)
                print(f"    - Reshaped convolutional layer to {reshaped_weights.shape}")
            else:
                reshaped_weights = numpy_weights

            # Add the processed weights to our dictionary
            key = f'w{layer_index}'
            formatted_weights[key] = reshaped_weights
            layer_index += 1
        else:
            # This part is optional but helpful for debugging to see what's being skipped.
            print(f"  - Skipping non-weight parameter: '{name}'")
            
    print("\nWeight extraction complete.")
    return formatted_weights


weights_for_eclipse = extract_and_format_weights(model) 

print("\nRunning ECLipsE estimation on the extracted ResNet weights...")
# Check if the dictionary is empty before proceeding
if not weights_for_eclipse:
    raise ValueError("Weight dictionary is empty. No Conv2D or Linear layers found to analyze.")

lip, trivial, time_taken = eclipse(weights_for_eclipse)
lip_fast, _, time_taken_fast = eclipse_fast(weights_for_eclipse)

# --- Step 5: Print the estimation results ---
print("\n--- Lipschitz Estimation Results ---")
print(f"ECLipsE Estimate: {lip:.4f}")
print(f"Fast ECLipsE estimate: {lip_fast:.4f}")
print(f"Trivial Upper Bound: {trivial:.4f}")
print(f"Normalized Ratio (ECLipsE / Trivial): {lip/trivial:.4f}")
print(f"Normalized Ratio (ECLipsE_fast / Trivial): {lip_fast/trivial:.4f}")
print(f"Computation Time: {time_taken:.4f} seconds")
print(f"Fast Computation Time: {time_taken_fast:.4f} seconds")