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
from lipMIP.hyperbox import Hyperbox 

# Imports for auto_LiRPA
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.jacobian import JacobianOP, GradNorm
from convert import conv_to_linear
from bab_runner import lirpa_local_lipschitz, compute_margin_jacobian_bound

# --- CONFIGURATION ---
FOLDER_NAME = 'CIFAR'
MODEL_NAME = 'cnn_4layer_stride1_padding0' # Options: 'mnist_cnn_4layer', 'mnist_mlp_3layer', 'mnist_cnn_4layer_8'
WEIGHTS_FILENAME = f'{MODEL_NAME}.pth'
IMAGE_FILENAME = 'test_image_cifar.png' 
EPSILON = 3/255  # The size of the attack box
STEPS = 2048      # Steps for Adam
WALKERS = 128     # Number of parallel restart points
TEMP = 1e-4
STEP_SIZE = 0.05 * EPSILON
BATCH_SIZE = 64

# --- DEVICE SETUP ---
gpu_id = 0 
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

class NormalizedModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
        # CIFAR-10 Constants (RGB = 3 channels)
        # Reshaped to (1, 3, 1, 1) for broadcasting across (B, C, H, W)
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
        
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        # Normalize then pass through the base model
        x_norm = (x - self.mean) / self.std
        return self.base_model(x_norm)

# Wrapper to handle Input Normalization (x - mean) / std
def setup_model_and_image():
    current_dir = os.getcwd()
    save_dir = os.path.join(current_dir, FOLDER_NAME)
    weights_path = os.path.join(save_dir, WEIGHTS_FILENAME)
    image_path = os.path.join(save_dir, IMAGE_FILENAME)
    
    # 1. Instantiate Model using models.py
    print(f"Initializing {MODEL_NAME} from models.py...")
    if MODEL_NAME == 'cnn_4layer_stride1_padding0':
        base_model = models.cnn_4layer_stride1_padding0()
    elif MODEL_NAME == 'cnn_4layer_stride1_padding0_demo':
        base_model = models.cnn_4layer_stride1_padding0_demo()
    elif MODEL_NAME == 'cnn_6layer_stride1_padding0':
        base_model = models.cnn_6layer_stride1_padding0()
    else:
        raise ValueError(f"Unknown model name: {MODEL_NAME}")
    
    # 2. Load Weights
    print(f"Loading weights from {weights_path}...")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found at {weights_path}. Run train_all_models.py first.")
        
    state_dict = torch.load(weights_path, map_location=device)
    base_model.load_state_dict(state_dict)
    
    # dummy_shape = torch.zeros(1,2,32,32).shape
    # model = conv_to_linear(model, dummy_shape)
    # Wrap in Normalization
    model = NormalizedModel(base_model).to(device)
    model.eval()

    # 3. Prepare Image (Resize to 28x28 Grayscale)
    preprocess_cifar = transforms.Compose([
        transforms.ToTensor() 
    ])

    try:
        img_pil = Image.open(image_path)
        x0 = preprocess_cifar(img_pil).unsqueeze(0).to(device) 
    except FileNotFoundError:
        print(f"ERROR: Image not found at {image_path}. Using random noise for testing.")
        x0 = torch.rand((1, 3, 32, 32)).to(device)

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

    import copy
    testnet = copy.deepcopy(wrapper_model.base_model)
    testnet.to(device)
    
    print("Preparing model for auto_LiRPA Jacobian analysis...")

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

    # center_x0_grad = center_x0.clone().requires_grad_(True)
    # y = network(center_x0_grad)
    # margin = y.matmul(c_vector_lirpa) 
    
    # grad_autograd = torch.autograd.grad(margin.sum(), center_x0_grad)[0]
    # lip_autograd = grad_autograd.abs().flatten(1).sum(dim=-1)
    
    center_x0_grad = center_x0.clone().requires_grad_(True)
    y = testnet(center_x0_grad)
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
    # ub = lirpa_model.compute_jacobian_bounds(x=(x_bounded, c_vector_lirpa), 
    #         method = 'IBP', bound_lower=False)[1]
            # bound_lower=False)[1]
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
        nms_radius=0.1 * EPSILON, # Radius to consider points "correlated"
        # --- Sobol Refinement ---
        top_k_refine=20,         # How many distinct peaks to probe
        k_box_epsilon=0.2*EPSILON,      # Size of the "small box" for dense Sobol sampling
        sobol_samples_per_k=2000,
        batch_size=BATCH_SIZE
    )

    accumulated_inputs = None
    accumulated_norms = None
    
    # List to store dictionary results for the dataframe
    results_history = []

    print(f"Running full Langevin sampler...")
    accumulated_inputs, accumulated_norms = sampler.run()
    

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(len(accumulated_norms)), sorted(accumulated_norms))
    plt.savefig('plot_image_cifar.png')

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
        print("\n> CONVERGENCE CHECK FAILED")
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



if __name__ == "__main__":
    import torch
    torch.manual_seed(42)
    st = time.time() 
    lirpa_res = run_lirpa_analysis()
    lirpa_time = time.time() - st
    st = time.time() 
    run_iterative_sampling()
    alpaca_time = time.time() - st 

    print('lirpa result:', lirpa_res)
    print(alpaca_time , lirpa_time)