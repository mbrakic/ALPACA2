import torch
import torch.nn as nn
import os
import numpy as np
from torchvision import models, transforms
from PIL import Image
import sys

# Ensure imports can find local files if necessary
sys.path.append(os.getcwd())

from LipPOT_iterative.LipPOT import LipPOT 
from ASSImagenetIterative import AdamSobolSampler
from lipMIP.hyperbox import Hyperbox # Assuming utils.py is in the Dependencies folder or root

# --- CONFIGURATION ---
FOLDER_NAME = "ImageNet"
WEIGHTS_FILENAME = "resnet50_weights.pth"
IMAGE_FILENAME = "tench.JPEG" 
EPSILON = 8/255  # The size of the attack box
STEPS = 400     # Steps for Adam
WALKERS = 64    # Number of parallel restart points
MAX_ITERATIONS = 5  # Maximum number of loops to attempt
BATCH_SIZE = 16

# --- DEVICE SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# --- MODEL SETUP ---
class NormalizedResNet(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, x):
        x_norm = self.normalize(x)
        return self.base_model(x_norm)

def setup_model_and_image():
    current_dir = os.getcwd()
    save_dir = os.path.join(current_dir, FOLDER_NAME)
    weights_path = os.path.join(save_dir, WEIGHTS_FILENAME)
    image_path = os.path.join(save_dir, IMAGE_FILENAME)
    os.makedirs(save_dir, exist_ok=True)

    weights_enum = models.ResNet50_Weights.DEFAULT
    url = weights_enum.url

    if not os.path.exists(weights_path):
        print(f"Downloading weights to {weights_path}...")
        torch.hub.download_url_to_file(url, weights_path)
    else:
        print(f"Found local weights at {weights_path}")

    base_model = models.resnet50(weights=None)
    state_dict = torch.load(weights_path, map_location=device)
    base_model.load_state_dict(state_dict)

    model = NormalizedResNet(base_model).to(device)
    model.eval()

    preprocess_0_1 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor() 
    ])

    try:
        img_pil = Image.open(image_path).convert('RGB')
        x0 = preprocess_0_1(img_pil).unsqueeze(0).to(device) 
    except FileNotFoundError:
        print(f"ERROR: Image not found at {image_path}. Please provide a valid image.")
        return None, None, None, None

    with torch.no_grad():
        logits = model(x0)
        probs = logits.softmax(dim=1)
        pred_id = probs.argmax().item()
        pred_name = weights_enum.meta["categories"][pred_id]
        confidence = probs[0, pred_id].item()

    print(f"\nTarget Image: {IMAGE_FILENAME}")
    print(f"Prediction: {pred_name} (ID: {pred_id})")
    print(f"Confidence: {confidence*100:.2f}%")

    c_vector = torch.zeros((1, 1000), device=device)
    c_vector[0, pred_id] = 1.0

    return model, x0, c_vector, pred_name

def setup_domain(x0):
    box_low = torch.clamp(x0 - EPSILON, 0.0, 1.0)
    box_hi = torch.clamp(x0 + EPSILON, 0.0, 1.0)
    flat_dim = x0.numel() 
    domain = Hyperbox.build_unit_hypercube(flat_dim)
    domain.box_low = box_low
    domain.box_hi = box_hi
    return domain

def run_iterative_sampling():
    model, x0, c_vector, pred_name = setup_model_and_image()
    if model is None: return

    domain = setup_domain(x0)

    sampler = AdamSobolSampler(
        model=model,
        c_vector=c_vector,
        domain=domain,
        device=device,
        attack_type='adam_topm',
        norm_type='linf',
        steps=STEPS,
        walkers=WALKERS,
        step_size=0.005,
        top_m=5,
        nms_radius=0.5*EPSILON,
        top_k_refine=20,
        batch_size=BATCH_SIZE
    )

    accumulated_inputs = None
    accumulated_norms = None

    print(f"\n--- Starting Iterative Sampling Loop (Max {MAX_ITERATIONS} iterations) ---")

    for i in range(MAX_ITERATIONS):
        print(f"\n>>> ITERATION {i+1} / {MAX_ITERATIONS}")
        
        # 1. Run Sampler
        print(f"Running sampler...")
        new_inputs, new_norms = sampler.run()
        print(f"Iteration {i+1} found {len(new_norms)} samples.")

        if len(new_norms) == 0:
            print("Warning: Sampler returned no samples this iteration.")
            if accumulated_norms is None:
                print("No samples accumulated. Aborting.")
                return
            else:
                print("Continuing with previously accumulated samples.")

        # 2. Accumulate Data
        if accumulated_norms is None:
            accumulated_inputs = new_inputs
            accumulated_norms = new_norms
        else:
            if len(new_norms) > 0:
                accumulated_inputs = np.concatenate([accumulated_inputs, new_inputs], axis=0)
                accumulated_norms = np.concatenate([accumulated_norms, new_norms], axis=0)
        
        print(f"Total accumulated samples: {len(accumulated_norms)}")

        # 3. Run LipPOT Analysis
        print(f"Running LipPOT analysis on {len(accumulated_norms)} samples...")
        
        # Note: Depending on your data size, you might want to increase n_search_samples
        lippot_results = LipPOT.run_full_analysis(
            data=accumulated_norms,
            coords=accumulated_inputs, 
            gamma=0.05, 
            n_search_samples=10000,
            show_plot=False, # Disable plot for loop to avoid blocking, enable at end if desired
            use_fine_graining=True,
            verbose=True
        )

        # 4. Check Convergence
        # We check the new property 'is_finite_success' which handles the logic for
        # infinite endpoints or failed optimizations.
        if lippot_results.optimization_succeeded:
            print("\n>>> SUCCESS: Finite endpoint estimated via optimization/search.")
            print(f"Final Estimated Endpoint: {lippot_results.final_L_high:.4f}")
            
            # Optional: Generate plot only on success
            LipPOT.plot_estimation_results(
                excesses=accumulated_norms[accumulated_norms > lippot_results.pot_results.threshold] - lippot_results.pot_results.threshold, 
                pot_results=lippot_results.pot_results
            )
            break
        else:
            print("\n>>> CONVERGENCE CHECK FAILED.")
            if lippot_results.analysis_halted_reason:
                print(f"Reason: {lippot_results.analysis_halted_reason}")
            else:
                print("Reason: Endpoint estimated as Infinite or Optimization failed.")
            
            if i < MAX_ITERATIONS - 1:
                print("Resampling to probe tail distribution...")
            else:
                print("Max iterations reached without convergence.")

    print("\n--- Process Complete ---")

if __name__ == "__main__":
    run_iterative_sampling()