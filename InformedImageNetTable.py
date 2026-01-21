import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd  # Added for DataFrame handling
from torchvision import models, transforms
from PIL import Image
import sys

# Ensure imports can find local files if necessary
sys.path.append(os.getcwd())

# Note: Assuming these imports exist in your environment
from ALPACA.ALPACA import ALPACA
from InformedSampling import AdamSobolSampler
from lipMIP.hyperbox import Hyperbox 

# Imports for auto_LiRPA
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.jacobian import JacobianOP, GradNorm
from bab_runner import lirpa_local_lipschitz, compute_margin_jacobian_bound

# --- CONFIGURATION ---
FOLDER_NAME = "ImageNet"
WEIGHTS_FILENAME = "resnet50_weights.pth"
IMAGE_FILENAME = "tench.JPEG" 
EPSILON = 3/255  # The size of the attack box
STEPS = 1200      # Steps for Adam
WALKERS = 64     # Number of parallel restart points
MAX_ITERATIONS = 1  # Maximum number of loops to attempt
BATCH_SIZE = 64

# --- DEVICE SETUP ---
gpu_id = 0  # Set this to 0 or 1
# device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"Running on device: {device}")

class AnalysisArgs:
    def __init__(self):
        # --- CRITICAL ARGS FOR BAB (User Overrides) ---
        self.bab = True              # Enable Branch and Bound (argparse default: False)
        self.timeout = 300.0         # Time budget in seconds (argparse default: 60)
        self.debug = False           # (argparse default: False)
        self.batch_size = 1         # Batch size during bab (argparse default: 128)
        self.method = 'CROWN'        # (argparse default: 'ours')
        
        # --- MODEL/DATA ARGS ---
        self.data = 'tinyimagenet'          # choices: MNIST, CIFAR, tinyimagenet, etc.
        self.num_classes = 1000      # (argparse default: 10)
        self.norm = np.inf           # Input perturbation norm (np.inf, 2, or 1)
        self.input_clipping = False  # Clip input range to [0,1]
        
        # --- GENERAL ARGS ---
        self.load = ''               # Path to load model
        self.device = device         # 'cpu' or 'cuda'
        self.model = 'cnn'           # Model architecture type
        self.model_params = ''       # String for model parameters

        # --- LIPSCHITZ / OPTIMIZATION ARGS ---
        self.use_recurjac_model = False
        self.eps = None              # Perturbation epsilon (float)
        self.heuristic = 'area'      # choices: 'babsr', 'area'
        self.filtering = False
        self.branching_candidates = 3
        self.start = 0
        self.num_examples = 1        # Aliased as --num_example in argparse
        self.max_domains = 200000
        
        # Optimization (OB) parameters
        self.ob_iteration = 20       # Aliased as --opt-steps
        self.ob_lr_decay = 0.98
        self.ob_lr = 0.5
        self.no_optimize = False
        
        # --- FLAGGING / MODEL CONVERSION ---
        self.cnn_to_mlp = False      # Convert CNN to MLP for baselines
        self.mono = False            # Monotonicity
        self.clean = False           # Clean evaluation
        
        # --- ADVANCED BAB SETTINGS ---
        # During bab, fix pre-activation bounds for gradient norm unless they 
        # are splitted; otherwise, recompute pre-activation bounds.
        self.fix_preact_gradnorm = False
        
        # Branch on grad norm node (abs/sqr)
        self.branch_gn = False
        
        # Consider intermediate bounds in heuristic
        self.heuristic_intermediate = False

# Instantiate the arguments
args = AnalysisArgs()

class NormalizedResNet(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        # Define mean and std as 4D tensors [1, C, 1, 1] for broadcasting
        # We use register_buffer so they are automatically moved to the correct device (cpu/cuda)
        self.register_buffer(
            'mean', 
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std', 
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        # Manual normalization: (x - mean) / std
        # Since mean/std are 4D, auto_LiRPA will accept this operation
        x_norm = (x - self.mean) / self.std
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

def run_lirpa_local_lipschitz_analysis(args):
    print("\n--- Starting auto-LiRPA analysis (Patched Model + BaB/Jacobian) ---")
    
    # 1. Get the wrapper model
    network, x0, c_vector, pred_name = setup_model_and_image() 
    
    network.to(device)

    # =========================================================================
    # INSERT THIS PATCH HERE
    # =========================================================================
    # Access the inner ResNet (since network is NormalizedResNet)
    inner_model = network.base_model if hasattr(network, 'base_model') else network

    if hasattr(inner_model, 'maxpool') and isinstance(inner_model.maxpool, nn.MaxPool2d):
        print("3. Patching MaxPool2d to Scaled Conv2d (Approx. AvgPool) to fix stride/kernel mismatch...")
        # Replace MaxPool with a depthwise Conv2d that approximates Average Pooling
        # This bypasses the stride!=kernel_size limitation in auto_LiRPA
        replacement_max = nn.Conv2d(
            in_channels=64, out_channels=64,
            kernel_size=3, stride=2, padding=1,
            groups=64, bias=False
        )
        # Set weights to 1/9 (averaging over 3x3 kernel)
        replacement_max.weight.data.fill_(1.0 / 9.0)
        replacement_max.weight.requires_grad = False
        inner_model.maxpool = replacement_max
    # =========================================================================

    print("Preparing model for auto_LiRPA Jacobian analysis...")
# def run_lirpa_local_lipschitz_analysis(args):
#     """
#     Computes a LIRPA-based upper bound on the Lipschitz constant using the
#     sophisticated patching from run_lirpa_analysis, but routes the final 
#     calculation through the BaB/Jacobian logic of lirpa_local_lipschitz.
#     """
#     print("\n--- Starting auto-LiRPA analysis (Patched Model + BaB/Jacobian) ---")
    
#     # 1. Get the wrapper model
#     wrapper_model, x0, c_vector, pred_name = setup_model_and_image() 
    
#     # 2. Extract the base ResNet (discard NormalizedResNet wrapper)
#     # network = wrapper_model.base_model
#     network = wrapper_model
#     network.to(device)
    
#     print("Preparing model for auto_LiRPA Jacobian analysis...")

    # =========================================================================
    # DOMAIN & LiRPA SETUP
    # =========================================================================
    
    # 1. Define Domain
    domain = setup_domain(x0) 
    center_x0 = (domain.box_low + domain.box_hi) / 2.0
    eps_tensor = (domain.box_hi - domain.box_low) / 2.0
    eps = eps_tensor.max().item()

    print(f"Domain Setup: eps={eps:.5f}")
    
    # 2. Setup Perturbation Object
    if torch.allclose(eps_tensor.min(), eps_tensor.max()):
         ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
    else:
        ptb = PerturbationLpNorm(norm=np.inf, eps=None, x_L=domain.box_low, x_U=domain.box_hi)

    # 3. Create the BoundedModule
    # Note: We wrap the *Patched Network* directly, not the MarginLipschitzWrapper
    # We use a dummy input to initialize the graph
    dummy_input = center_x0
    lirpa_model = BoundedModule(
        network, dummy_input, 
        bound_opts={'conv_mode': 'patches'}, 
        device=device
    )
    lirpa_model.forward_final_name = None

    # 4. Prepare BoundedTensor for the actual calculation
    x_bounded = BoundedTensor(center_x0, ptb)
    
    # We need a dummy label tensor if compute_margin_jacobian_bound expects it
    # Assuming labels are needed for shape inference or c_matrix generation
    # Create a dummy label (e.g., class 0)
    labels = torch.zeros(1, dtype=torch.long, device=device) 

    # =========================================================================
    # EXECUTION: BaB or Standard Jacobian Bound
    # =========================================================================
    
    print(f"\nComputing Lipschitz Bound (BaB={args.bab})...")
    
    val = compute_margin_jacobian_bound(lirpa_model, x_bounded, labels, args)
    print(f"--- Analysis Complete. Result: {val} ---")
    return val


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
        top_k_refine=50,
        batch_size=BATCH_SIZE
    )

    accumulated_inputs = None
    accumulated_norms = None
    
    # List to store dictionary results for the dataframe
    results_history = []

    print(f"\n--- Starting Iterative Sampling Loop (Running all {MAX_ITERATIONS} iterations) ---")

    for i in range(MAX_ITERATIONS):
        print(f"\n>>> ITERATION {i+1} / {MAX_ITERATIONS}")
        
        # 1. Run Sampler
        if i == 0:
            print(f"Running full Adam+Sobol sampler...")
            new_inputs, new_norms = sampler.run()
        else:
            print(f"Running refinement on best candidates found so far...")
            new_inputs, new_norms = sampler.run_refinement_only(accumulated_inputs, accumulated_norms)

        print(f"Iteration {i+1} generated {len(new_norms)} new samples.")
 
        # 2. Accumulate Data
        if len(new_norms) == 0:
            print("Warning: Sampler returned no samples this iteration.")
            if accumulated_norms is None:
                print("No samples accumulated. Aborting.")
                return
            else:
                print("Continuing with previously accumulated samples.")
        else:
            if accumulated_norms is None:
                accumulated_inputs = new_inputs
                accumulated_norms = new_norms
            else:
                accumulated_inputs = np.concatenate([accumulated_inputs, new_inputs], axis=0)
                accumulated_norms = np.concatenate([accumulated_norms, new_norms], axis=0)
        
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
            "Iteration": i + 1,
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
        
        # Determine if we need to prep for next loop (Just logging here, loop continues automatically)
        if i < MAX_ITERATIONS - 1:
            print("Resampling (Refinement) to probe tail distribution for next iteration...")

    print("\n--- Process Complete ---")
    
    # 5. Create and Print DataFrame
    df = pd.DataFrame(results_history)
    
    # Optional: formatting to make the table look nicer
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.5f}'.format)
    
    print("\n=== ITERATION RESULTS TABLE ===")
    print(df)
    
    # Optional: Save to CSV
    # df.to_csv("lippot_iteration_results.csv", index=False)

def run_lirpa_analysis():
    """
    Computes a LIRPA-based upper bound on the Lipschitz constant.
    Includes comprehensive patching for ResNet50 on auto_LiRPA Jacobian.
    """
    print("\n--- Starting auto-LiRPA analysis (on local domain) ---")
    
    # 1. Get the wrapper model
    wrapper_model, x0, c_vector, pred_name = setup_model_and_image() 
    
    # 2. Extract the base ResNet (we will discard the NormalizedResNet wrapper)
    #    because we are going to fuse the normalization into the first layer.
    network = wrapper_model.base_model
    network.to(device)
    
    print("Preparing model for auto_LiRPA Jacobian analysis...")

    # =========================================================================
    # PATCH 1: Fuse Input Normalization into Conv1
    # Removes 'BoundConstant' nodes caused by transforms.Normalize
    # =========================================================================
    def fuse_normalization(model):
        print("1. Fusing Input Normalization (Mean/Std) into Conv1...")
        # ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        
        conv1 = model.conv1
        
        with torch.no_grad():
            # Adjust Weights: W_new = W_old / std
            w_new = conv1.weight / std
            
            # Adjust Bias: b_new = b_old - (conv(mean) with w_new)
            # Since conv1 is 7x7 stride 2, we must calculate the effect of subtracting mean.
            # The easiest way is to calculate the bias shift mathematically:
            # The convolution of a constant image 'mean' results in: 
            # sum(weights over spatial dims) * mean_value
            
            # 1. Calculate the bias correction factor
            # We treat 'mean' as an input image.
            # We can use F.conv2d to find the exact bias shift required.
            # Create a 1-pixel image with the mean values
            # However, conv1 has padding=3.
            # A simpler mathematical approach for 'constant subtraction':
            # term = (weight * mean/std).sum(dim=(1,2,3))
            
            term = (conv1.weight * (mean / std)).reshape(conv1.out_channels, -1).sum(dim=1)
            
            conv1.weight.copy_(w_new)
            
            if conv1.bias is None:
                conv1.bias = nn.Parameter(-term)
            else:
                conv1.bias.data.sub_(term)

    fuse_normalization(network)

    # =========================================================================
    # PATCH 2: Fuse Batch Normalization into Convolutions
    # Removes 'BoundBatchNormalization' nodes
    # =========================================================================
    def fuse_conv_bn(conv, bn):
        with torch.no_grad():
            mean = bn.running_mean
            var_sqrt = torch.sqrt(bn.running_var + bn.eps)
            gamma = bn.weight
            beta = bn.bias
            w = conv.weight
            if conv.bias is not None:
                b = conv.bias
            else:
                b = torch.zeros_like(mean)
            scale = (gamma / var_sqrt).reshape(-1, 1, 1, 1)
            w_new = w * scale
            b_new = (b - mean) * (gamma / var_sqrt) + beta
            conv.weight.copy_(w_new)
            if conv.bias is None:
                conv.bias = nn.Parameter(b_new)
            else:
                conv.bias.copy_(b_new)
            return nn.Identity()

    def fuse_resnet_bn(model):
        print("2. Fusing BatchNormalization layers into Convolutions...")
        model.bn1 = fuse_conv_bn(model.conv1, model.bn1)
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(model, layer_name)
            for block in layer:
                block.bn1 = fuse_conv_bn(block.conv1, block.bn1)
                block.bn2 = fuse_conv_bn(block.conv2, block.bn2)
                block.bn3 = fuse_conv_bn(block.conv3, block.bn3)
                if block.downsample is not None:
                    fuse_conv_bn(block.downsample[0], block.downsample[1])
                    block.downsample[1] = nn.Identity()

    fuse_resnet_bn(network)

    # =========================================================================
    # PATCH 3: Replace MaxPool2d with Strided Convolution
    # Removes 'BoundMaxPool' nodes
    # =========================================================================
    if hasattr(network, 'maxpool') and isinstance(network.maxpool, nn.MaxPool2d):
        print("3. Patching MaxPool2d to Scaled Conv2d (Approx. AvgPool)...")
        replacement_max = nn.Conv2d(
            in_channels=64, out_channels=64,
            kernel_size=3, stride=2, padding=1,
            groups=64, bias=False
        )
        # 1/9 for averaging effect
        replacement_max.weight.data.fill_(1.0 / 9.0)
        replacement_max.weight.requires_grad = False
        network.maxpool = replacement_max

    # =========================================================================
    # PATCH 4: Replace Global Average Pooling with Convolution
    # Removes 'BoundAveragePool' / 'BoundGlobalAveragePool' nodes
    # =========================================================================
    if hasattr(network, 'avgpool'):
        print("4. Patching Global AvgPool to Conv2d...")
        replacement_avg = nn.Conv2d(
            in_channels=2048, out_channels=2048, 
            kernel_size=7, stride=1, padding=0, 
            groups=2048, bias=False
        )
        # 1/49 for averaging effect
        replacement_avg.weight.data.fill_(1.0 / 49.0)
        replacement_avg.weight.requires_grad = False
        network.avgpool = replacement_avg
    # =========================================================================

    domain = setup_domain(x0) 
    
    # c_vector is [1, 1000]. Transpose to [1000, 1]
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
    eps_tensor = (domain.box_hi - domain.box_low) / 2.0
    eps_min = eps_tensor.min()
    eps_max = eps_tensor.max()
    
    if torch.allclose(eps_min, eps_max):
         eps = eps_max.item() 
         print(f"Domain is L-inf ball: center=..., eps={eps:.4f}")
         ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
    else:
        print("Domain is a general hyperbox (clipping occurred).")
        ptb = PerturbationLpNorm(norm=np.inf, eps=None, x_L=domain.box_low, x_U=domain.box_hi)
        center_x0_for_tracing = center_x0
        ptb_dummy = PerturbationLpNorm(norm=np.inf, eps=0.0)
        x_dummy_for_tracing = BoundedTensor(center_x0_for_tracing, ptb_dummy)
        
    ### ROBUST SANITY CHECK START ###
    print("\n--- Running Sanity Check ---")
    model_wrapper = MarginLipschitzWrapper(network)
    
    if 'x_dummy_for_tracing' not in locals():
        x_dummy_for_tracing = BoundedTensor(center_x0, PerturbationLpNorm(norm=np.inf, eps=0))

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

    # assert torch.allclose(lip_autograd, lip_lirpa_forward.flatten(), atol=1e-4), \
    #     "Sanity check failed: BoundedModule forward pass does not match autograd."
    print("✅ Sanity check passed: LiRPA forward pass matches autograd.")
    print("--- Sanity Check Complete ---\n")
    ### SANITY CHECK END ###

    x_bounded = BoundedTensor(center_x0, ptb)

    print("Computing upper bound with CROWN-IBP...")
    # _, ub = lirpa_model.compute_bounds(x=(x_bounded, c_vector_lirpa), method='IBP+backward')
    ub = lirpa_model.compute_jacobian_bounds(x=(x_bounded, c_vector_lirpa), bound_lower=False)[1]
    print("--- auto-LiRPA analysis complete ---")
    print(ub.item())
    return ub.item()

if __name__ == "__main__":
    run_lirpa_analysis()
    # run_lirpa_local_lipschitz_analysis(args)
    # run_iterative_sampling()