import os
import warnings
import time
import torch
import itertools 
from argparse import ArgumentParser
from LBDN.train import *
from LBDN.evaluate import *
from LipPOT.LipPOT_TA import LipPOT

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

model, testLoader, trainLoader = prepare_model(config)
model.to(device)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

print("\nRunning accuracy test...")
correct = 0
total = 0

# Set model to evaluation mode for accuracy check
model.eval()
with torch.no_grad():
    # Loop through a subset of testloader for a quick check
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


# --- Gradient Norm Calculation ---
print("\nCalculating gradient norms for all training data...")

from LipConstEstimator import LipConstEstimator
est = LipConstEstimator(model=model)
est.model_review()
lip_trivial = est.estimate(method='trivial')
lip_fast = est.estimate(method='EclipsE_fast')
print(f'Trivial Lip Const = {lip_trivial}')
print(f'EclipsE Fast Lip Const = {lip_fast}')
print(f'Ratio = {lip_fast / lip_trivial}')





quit()

# Set model to evaluation mode but enable gradients for the calculation
model.eval()
grad_norms_list = []

# loader = trainLoader 
# loader = testLoader
loader = itertools.chain(trainLoader, testLoader)

# Start timer
start_time = time.time()

for data in loader:
    images, labels = data
    images = images.to(device)
    labels = labels.to(device)

    # Enable gradient calculation for the input images
    images.requires_grad_(True)

    outputs = model(images)

    # We need a scalar value to call .backward(). The sum of the outputs is a common choice.
    loss = outputs.sum()

    # Calculate gradients of the loss with respect to the model's inputs
    loss.backward()

    # Calculate the L2 norm for each image's gradient in the batch
    # Gradients are in images.grad, shape is (batch_size, C, H, W)
    # We flatten the C, H, W dimensions to calculate the norm for each image
    # l2 norm
    batch_norms = torch.linalg.norm(images.grad.view(images.size(0), -1), dim=1)
    # linf norm
    # batch_norms = torch.linalg.norm(images.grad.view(images.size(0), -1), ord=float('inf'), dim=1)
    grad_norms_list.append(batch_norms.cpu())

    # Zero out the gradients for the next batch
    model.zero_grad()


grad_norms = torch.cat(grad_norms_list).numpy() 

# # Concatenate all batch norms into a single tensor
# grad_norms = np.array(grad_norms_list).flatten()

# End timer
end_time = time.time()
elapsed_time = end_time - start_time

print("\n--- Gradient Norm Calculation Results ---")
print(f"Calculation finished in {elapsed_time:.2f} seconds.")
print(f"Stored gradient norms for {grad_norms.shape} training images.")
print(f"Shape of the final 'grad_norms' tensor: {grad_norms.shape}")

print("Running LipPOT analysis")

gamma = 0.05
n_search_samples = 25000 

LipPOT.run_full_analysis(
        data=grad_norms,
        gamma=gamma,
        n_search_samples=n_search_samples,
        show_plot=True,
        use_fine_graining=True,
        verbose=True
)

import matplotlib.pyplot as plt 
plt.show()

