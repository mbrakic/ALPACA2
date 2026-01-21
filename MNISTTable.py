import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import pandas as pd
import sys
import time

# Ensure imports can find local files
sys.path.append(os.getcwd())

# Import your model definitions
import models 

# Note: Assuming these imports exist in your environment
from ALPACA.ALPACA import ALPACA
from InformedSampling2 import LangevinSampler

from lipMIP.lipMIP import LipMIP
from lipMIP.hyperbox import Hyperbox 

# Imports for auto_LiRPA
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.jacobian import JacobianOP, GradNorm
from bab_runner import lirpa_local_lipschitz, compute_margin_jacobian_bound

# --- CONFIGURATION ---
FOLDER_NAME = 'MNIST'
MODEL_NAME = 'mnist_cnn_4layer' # Options: 'mnist_cnn_4layer', 'mnist_mlp_3layer', 'mnist_cnn_4layer_8'
WEIGHTS_FILENAME = f'{MODEL_NAME}.pth'
IMAGE_FILENAME = 'test_image_mnist.png' 
EPSILON = 3/255  # The size of the attack box
STEPS = 1024      # Steps for Adam
WALKERS = 128     # Number of parallel restart points
TEMP = 1e-4
STEP_SIZE = 0.05 * EPSILON
BATCH_SIZE = 64

# --- DEVICE SETUP ---
gpu_id = 0 
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# Wrapper to handle Input Normalization (x - mean) / std
class NormalizedModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        # MNIST Constants (Grayscale = 1 channel)
        self.register_buffer('mean', torch.tensor([0.1307]).view(1, 1, 1, 1))
        self.register_buffer('std', torch.tensor([0.3081]).view(1, 1, 1, 1))

    def forward(self, x):
        x_norm = (x - self.mean) / self.std
        return self.base_model(x_norm)

def setup_model_and_image():
    current_dir = os.getcwd()
    save_dir = os.path.join(current_dir, FOLDER_NAME)
    weights_path = os.path.join(save_dir, WEIGHTS_FILENAME)
    image_path = os.path.join(save_dir, IMAGE_FILENAME)
    
    # 1. Instantiate Model using models.py
    print(f"Initializing {MODEL_NAME} from models.py...")
    if MODEL_NAME == 'mnist_cnn_4layer':
        base_model = models.mnist_cnn_4layer()
    elif MODEL_NAME == 'mnist_mlp_3layer':
        base_model = models.mnist_mlp_3layer()
    elif MODEL_NAME == 'mnist_cnn_4layer_8':
        base_model = models.mnist_cnn_4layer_8()
    else:
        raise ValueError(f"Unknown model name: {MODEL_NAME}")
    
    # 2. Load Weights
    print(f"Loading weights from {weights_path}...")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found at {weights_path}. Run train_all_models.py first.")
        
    state_dict = torch.load(weights_path, map_location=device)
    base_model.load_state_dict(state_dict)
    
    # Wrap in Normalization
    model = NormalizedModel(base_model).to(device)
    model.eval()

    # 3. Prepare Image (Resize to 28x28 Grayscale)
    preprocess_mnist = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1), 
        transforms.ToTensor() 
    ])

    try:
        img_pil = Image.open(image_path)
        x0 = preprocess_mnist(img_pil).unsqueeze(0).to(device) 
    except FileNotFoundError:
        print(f"ERROR: Image not found at {image_path}. Using random noise for testing.")
        x0 = torch.rand((1, 1, 28, 28)).to(device)

    # 4. Get Prediction
    with torch.no_grad():
        logits = model(x0)
        probs = logits.softmax(dim=1)
        pred_id = probs.argmax().item()
        confidence = probs[0, pred_id].item()

    print(f"\nTarget Image: {IMAGE_FILENAME}")
    print(f"Prediction ID: {pred_id} (Confidence: {confidence*100:.2f}%)")

    # 5. Create C-Vector (One-hot for target class, 10 classes)
    c_vector = torch.zeros((1, 10), device=device)
    c_vector[0, pred_id] = 1.0

    return model, x0, c_vector, str(pred_id)

def setup_domain(x0):
    box_low = torch.clamp(x0 - EPSILON, 0.0, 1.0)
    box_hi = torch.clamp(x0 + EPSILON, 0.0, 1.0)
    flat_dim = x0.numel() 
    domain = Hyperbox.build_unit_hypercube(flat_dim)
    domain.box_low = box_low
    domain.box_hi = box_hi
    return domain

def run_lirpa_analysis():
    """
    Computes a LIRPA-based upper bound on the Lipschitz constant.
    """
    print(f"\n--- Starting auto-LiRPA analysis ({MODEL_NAME} + MNIST) ---")
    
    wrapper_model, x0, c_vector, pred_name = setup_model_and_image() 
    
    # Extract base model (Sequential)
    network = wrapper_model.base_model
    network.to(device)
    
    print("Preparing model for auto_LiRPA Jacobian analysis...")

    # =========================================================================
    # PATCH 1: Fuse Input Normalization into the First Layer
    # Crucial for LiRPA to handle input bounds correctly without extra nodes.
    # =========================================================================
    def fuse_normalization(model):
        print("1. Fusing Input Normalization (Mean/Std) into First Layer...")
        mean = torch.tensor([0.1307], device=device).view(1, 1, 1, 1)
        std = torch.tensor([0.3081], device=device).view(1, 1, 1, 1)
        
        # Check if the first layer is Conv2d (for CNNs) or Linear (for MLPs)
        first_layer = model[0]
        
        # Skip flattening layers if present at start
        idx = 0
        while not isinstance(first_layer, (nn.Conv2d, nn.Linear)) and idx < len(model):
            idx += 1
            first_layer = model[idx]

        with torch.no_grad():
            if isinstance(first_layer, nn.Conv2d):
                # Conv2d Fusion
                w_new = first_layer.weight / std
                term = (first_layer.weight * (mean / std)).reshape(first_layer.out_channels, -1).sum(dim=1)
                first_layer.weight.copy_(w_new)
                if first_layer.bias is None:
                    first_layer.bias = nn.Parameter(-term)
                else:
                    first_layer.bias.data.sub_(term)
            elif isinstance(first_layer, nn.Linear):
                # Linear Fusion (Flattened input)
                # Mean/Std are 1x1x1x1, need to flatten to match input features
                # MNIST is 28x28 = 784
                mean_flat = mean.view(-1).repeat(784) # Simple scalar repeat for MNIST
                std_flat = std.view(-1).repeat(784)
                
                w_new = first_layer.weight / std_flat
                # Bias shift: W * (mean/std)
                term = (first_layer.weight * (mean_flat / std_flat)).sum(dim=1)
                
                first_layer.weight.copy_(w_new)
                if first_layer.bias is None:
                    first_layer.bias = nn.Parameter(-term)
                else:
                    first_layer.bias.data.sub_(term)

    fuse_normalization(network)

    # Note: `mnist_cnn_4layer` in models.py does NOT use BatchNorm, MaxPool, or Dropout.
    # So we do not need the complex patching logic from the previous script.
    
    # =========================================================================
    # LIRPA SETUP 
    # =========================================================================
    domain = setup_domain(x0) 
    c_vector_lirpa = c_vector.T.to(device) 

    class MarginLipschitzWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.grad_norm = GradNorm(norm=1) 

        def forward(self, x, mask_input):
            y = self.model(x)
            margin_output = y.matmul(mask_input)
            jacobian = JacobianOP.apply(margin_output, x)
            lipschitz_val = self.grad_norm(jacobian)
            return lipschitz_val

    center_x0 = (domain.box_low + domain.box_hi) / 2.0
    
    # Handle clipping/general hyperbox
    ptb = PerturbationLpNorm(norm=np.inf, eps=None, x_L=domain.box_low, x_U=domain.box_hi)
    
    # For sanity check dummy inputs
    x_dummy_for_tracing = BoundedTensor(center_x0, PerturbationLpNorm(norm=np.inf, eps=0.0))
    
    ### ROBUST SANITY CHECK START ###
    print("\n--- Running Sanity Check ---")
    model_wrapper = MarginLipschitzWrapper(network)
    
    lirpa_model = BoundedModule(
        model_wrapper, (x_dummy_for_tracing, c_vector_lirpa),
        bound_opts={'conv_mode': 'patches'}, device=device)

    center_x0_grad = center_x0.clone().requires_grad_(True)
    y = network(center_x0_grad)
    margin = y.matmul(c_vector_lirpa) 
    
    grad_autograd = torch.autograd.grad(margin.sum(), center_x0_grad)[0]
    lip_autograd = grad_autograd.abs().flatten(1).sum(dim=-1)
    print(f"Lipschitz at center (via autograd): {lip_autograd.item():.6f}")

    lip_lirpa_forward = lirpa_model(center_x0, c_vector_lirpa)
    print(f"Lipschitz at center (via LiRPA fwd pass): {lip_lirpa_forward.item():.6f}")

    print("✅ Sanity check passed" if torch.allclose(lip_autograd, lip_lirpa_forward.flatten(), atol=1e-4) else "Warning: Sanity check mismatch")
    ### SANITY CHECK END ###

    x_bounded = BoundedTensor(center_x0, ptb)

    print("Computing upper bound with CROWN-IBP...")
    # ub = lirpa_model.compute_jacobian_bounds(x=(x_bounded, c_vector_lirpa), bound_lower=False)[1]
    ub = lirpa_model.compute_bounds(x=(x_bounded, c_vector_lirpa), 
            bound_lower=False, method='CROWN-IBP')[1]
    print("--- auto-LiRPA analysis complete ---")
    print(ub.item())
    return ub.item()

def run_iterative_sampling():
    model, x0, c_vector, pred_name = setup_model_and_image()

    if model is None: return

    domain = setup_domain(x0)

    sampler = LangevinSampler(
        model=model,
        c_vector=c_vector,
        domain=domain,
        device=device,
        # --- SGLD Specifics ---
        norm_type='linf',
        steps=STEPS,             
        walkers=WALKERS,         
        step_size=STEP_SIZE,     
        temperature=TEMP,        # NEW: Critical for SGLD
        # --- Spatial Decorrelation ---
        nms_radius=0.25 * EPSILON, # Radius to consider points "correlated"
        # --- Sobol Refinement ---
        top_k_refine=20,         # How many distinct peaks to probe
        k_box_epsilon=0.05*EPSILON,      # Size of the "small box" for dense Sobol sampling
        sobol_samples_per_k=2000,
        batch_size=BATCH_SIZE
    )

    accumulated_inputs = None
    accumulated_norms = None

    results_history = [] 

    accumulated_inputs, accumulated_norms = sampler.run()

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(len(accumulated_norms)), sorted(accumulated_norms))
    plt.savefig('plot_image_mnist.png')

    print(f"Total accumulated samples: {len(accumulated_norms)}")

    # 3. Run Alpaca Analysis
    print(f"Running Alpaca analysis on {len(accumulated_norms)} samples...")

    alpaca_results = ALPACA.run_full_analysis(
        data=accumulated_norms,
        coords=accumulated_inputs, 
        gamma=0.001, 
        n_search_samples=10000,
        show_plot=False, 
        use_fine_graining=True,
        verbose=True
    )

    # 4. Store Results (Do not break)
    current_result = {
        "Total_Samples": len(accumulated_norms),
        "Success": alpaca_results.optimization_succeeded,
        "Est_Endpoint": None,
        "Threshold_u": alpaca_results.pot_results.threshold,
        "Notes": ""
    }

    if alpaca_results.optimization_succeeded:
        print(f"\n> SUCCESS: Finite endpoint estimated: {alpaca_results.max_endpoint_from_optimization:.4f}")
        current_result["Est_Endpoint"] = alpaca_results.max_endpoint_from_optimization
        current_result["Notes"] = "Converged"
    else:
        print("\n> CONVERGENCE CHECK FAILED (Continuing to next iteration...)")
        current_result["Est_Endpoint"] = np.inf
        if alpaca_results.analysis_halted_reason:
            current_result["Notes"] = str(alpaca_results.analysis_halted_reason)
        else:
            current_result["Notes"] = "Infinite/Failed"

    # Append to history
    results_history.append(current_result)

    print("\n--- Process Complete ---")
    
    # 5. Create and Print DataFrame
    df = pd.DataFrame(results_history)
    
    # Optional: formatting to make the table look nicer
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.5f}'.format)
    
    print("\n=== RESULTS TABLE ===")
    print(df)
    
    # Optional: Save to CSV
    # df.to_csv("lippot_iteration_results.csv", index=False)

def run_lipmip_analysis():
    """
    Computes the Lipschitz constant using LipMIP.
    """
    network, x0, c_vector, pred = setup_model_and_image() 
    domain = setup_domain(x0) 
    norm = 'linf' 
    print("\n--- Starting LipMIP analysis ---")
    # network.to(device)
    print("Setting up LipMIP problem...")
    lipmip_problem = LipMIP(network, domain, c_vector,
                            primal_norm = norm, verbose=True)
    print("Computing maximum Lipschitz constant (this may take a moment)...")
    result = lipmip_problem.compute_max_lipschitz()
    print("--- LipMIP analysis complete ---")
    return result


if __name__ == "__main__":
    import torch
    torch.manual_seed(42)
    mipres = run_lipmip_analysis()
    # st = time.time() 
    # lirpa_res = run_lirpa_analysis()
    # lirpa_time = time.time() - st
    # st = time.time() 
    # run_iterative_sampling()
    # alpaca_time = time.time() - st 

    # print('lirpa result:', lirpa_res)
    # print(alpaca_time , lirpa_time)