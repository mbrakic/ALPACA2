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
from InformedSampling import LangevinSampler, RandomSampler

from lipMIP.lipMIP import LipMIP
from lipMIP.hyperbox import Hyperbox 
from lipMIP.relu_nets import ReLUNet

# Imports for auto_LiRPA
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.jacobian import JacobianOP, GradNorm
from bab_runner import lirpa_local_lipschitz, compute_margin_jacobian_bound

# imports for eclipse
from EClipsE.LipConstEstimator import LipConstEstimator


def get_model_function_name(dataset, key):
    # This dictionary maps your "Keys" to the actual function names in models.py
    model_zoo = {
        'MNIST': {
            'CNN_4Layer':   'mnist_cnn_4layer',
            'CNN_4Layer_8': 'mnist_cnn_4layer_8',
            'MLP_3Layer':   'mnist_mlp_3layer'
        },
        'CIFAR': {
            'CNN_4Layer':   'cnn_4layer_stride1_padding0',
            'CNN_6Layer':   'cnn_6layer_stride1_padding0',
            # 'CNN_4Layer_Demo': 'cnn_4layer_stride1_padding0_demo' 
        },
        'TinyImageNet': {
            'CNN_4Layer':   'cnn_4layer_stride2_imagenet',
            'CNN_6Layer':   'cnn_6layer_stride2_imagenet'
        }
    }
    
    try:
        return model_zoo[dataset][key]
    except KeyError:
        raise ValueError(f"Model key '{key}' not found for dataset '{dataset}'.")

def setup_model_and_image(config_args):
    current_dir = os.getcwd()
    save_dir = os.path.join(current_dir, config_args['FOLDER_NAME'])
    weights_path = os.path.join(save_dir, config_args['WEIGHTS_FILENAME'])
    image_path = os.path.join(save_dir, config_args['IMAGE_FILENAME'])
    stats = config_args['WEIGHT_NORMALISATION']
    
    print(f"Initializing {config_args['MODEL_NAME']} from models.py...")
    
    # 1. Get the internal function name (e.g., "cnn_4layer_stride1_padding0")
    # using the dataset (FOLDER_NAME) and the key (MODEL_NAME)
    internal_func_name = get_model_function_name(config_args['FOLDER_NAME'], config_args['MODEL_NAME'])

    # 2. Dynamically get the function from the 'models' module and call it
    try:
        model_builder = getattr(models, internal_func_name) # Get the function object
        network = model_builder()                           # Execute it ()
    except AttributeError:
        raise AttributeError(f"Function '{internal_func_name}' not found in models.py")
    # --- NEW LOGIC END ---

    # 3. Load Weights (Standard code follows...)
    print(f"Loading weights from {weights_path}...")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found at {weights_path}.")
        
    state_dict = torch.load(weights_path, map_location=config_args['DEVICE'])
    network.load_state_dict(state_dict)   

    # Wrap in Normalization
    model = network.to(config_args['DEVICE'])
    model.eval()

    # 3. Prepare Image (Resize to 28x28 Grayscale)
    preprocess = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(*stats),
    ])

    try:
        img_pil = Image.open(image_path)
        x0 = preprocess(img_pil).unsqueeze(0).to(config_args['DEVICE']) 
    except FileNotFoundError:
        print(f"ERROR: Image not found at {image_path}")

    # 4. Get Prediction
    with torch.no_grad():
        logits = model(x0)
        probs = logits.softmax(dim=1)
        pred_id = probs.argmax().item()
        confidence = probs[0, pred_id].item()

    print(f"\nTarget Image: {config_args['IMAGE_FILENAME']}")
    print(f"Prediction ID: {pred_id} (Confidence: {confidence*100:.2f}%)")

    if config_args['FOLDER_NAME'] == 'MNIST':
        output_dimension = 10 
    elif config_args['FOLDER_NAME'] == 'CIFAR':
        output_dimension = 10 
    elif config_args['FOLDER_NAME'] == 'TinyImageNet':
        output_dimension = 200
    # 5. Create C-Vector (One-hot for target class, 10 classes)
    c_vector = torch.zeros((1, output_dimension), device=config_args['DEVICE'])
    c_vector[0, pred_id] = 1.0

    return model, x0, c_vector, str(pred_id)

def setup_domain(config_args, x0):
    # 1. Calculate Normalized Epsilon
    # Epsilon (pixel) / Std
    # Note: assuming single channel or constant std for simplicity here.
    # If 3-channel color, this is technically inaccurate but so long as all 
    std_val = config_args['WEIGHT_NORMALISATION'][1][0] # e.g. 0.3081
    eps_norm = config_args['EPSILON'] / std_val

    # 2. Calculate Valid Normalized Bounds
    mean_val = config_args['WEIGHT_NORMALISATION'][0][0]
    min_valid = (0.0 - mean_val) / std_val
    max_valid = (1.0 - mean_val) / std_val

    # 3. Create Box
    # Clamp to the valid NORMALIZED range, not 0.0 and 1.0
    box_low = torch.clamp(x0 - eps_norm, min_valid, max_valid)
    box_hi  = torch.clamp(x0 + eps_norm, min_valid, max_valid)
    
    flat_dim = x0.numel() 
    domain = Hyperbox.build_unit_hypercube(flat_dim)
    domain.box_low = box_low
    domain.box_hi = box_hi
    
    return domain

def run_lirpa_analysis(lirpa_args, config_args):
    """
    Computes a LIRPA-based upper bound on the Lipschitz constant.
    """
    print(f"\n--- Starting auto-LiRPA analysis ({config_args['MODEL_NAME']} + MNIST) ---")

    lirpa_method = lirpa_args['METHOD']

    # This wrapper correctly defines the symbolic computation for auto-LiRPA.
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
    

    print("Preparing model for auto_LiRPA Jacobian analysis...")
    model, x0, c_vector, pred_name = setup_model_and_image(config_args) 
    
    model.to(config_args['DEVICE'])
    domain = setup_domain(config_args, x0) 
    c_vector_lirpa = c_vector.T.to(config_args['DEVICE']) 
    
    center_x0 = (domain.box_low + domain.box_hi) / 2.0
    
    # Handle clipping/general hyperbox
    ptb = PerturbationLpNorm(norm=np.inf, eps=None, x_L=domain.box_low, x_U=domain.box_hi)
    
    # For sanity check dummy inputs
    x_dummy_for_tracing = BoundedTensor(center_x0, PerturbationLpNorm(norm=np.inf, eps=0.0))
    
    ### ROBUST SANITY CHECK START ###
    print("\n--- Running Sanity Check ---")
    model_wrapper = MarginLipschitzWrapper(model)
    
    lirpa_model = BoundedModule(
        model_wrapper, (x_dummy_for_tracing, c_vector_lirpa),
        bound_opts={'conv_mode': 'patches'}, device=config_args['DEVICE'])

    center_x0_grad = center_x0.clone().requires_grad_(True)
    y = model(center_x0_grad)
    margin = y.matmul(c_vector_lirpa) 
    
    grad_autograd = torch.autograd.grad(margin.sum(), center_x0_grad)[0]
    lip_autograd = grad_autograd.abs().flatten(1).sum(dim=-1)
    print(f"Lipschitz at center (via autograd): {lip_autograd.item():.6f}")

    lip_lirpa_forward = lirpa_model(center_x0, c_vector_lirpa)
    print(f"Lipschitz at center (via LiRPA fwd pass): {lip_lirpa_forward.item():.6f}")

    print("✅ Sanity check passed" if torch.allclose(lip_autograd, lip_lirpa_forward.flatten(), atol=1e-4) else "Warning: Sanity check mismatch")
    ### SANITY CHECK END ###

    x_bounded = BoundedTensor(center_x0, ptb)

    print(f"Computing upper bound with {lirpa_method}...")
    # ub = lirpa_model.compute_jacobian_bounds(x=(x_bounded, c_vector_lirpa), bound_lower=False)[1]
    ub = lirpa_model.compute_bounds(x=(x_bounded, c_vector_lirpa), 
            bound_lower=False, method=lirpa_method)[1]
            # bound_lower=False, method='CROWN-IBP')[1]
    print("--- auto-LiRPA analysis complete ---")
    print(ub.item())
    return ub.item()

def run_lipschitz_sampling(config_args, sampling_args):
    model, x0, c_vector, pred_name = setup_model_and_image(config_args)

    # RE-CALCULATE step size for Normalized Space
    # 1. Retrieve the std used for normalization
    std_val = config_args['WEIGHT_NORMALISATION'][1][0] 
    # 2. Scale the pixel epsilon to normalized epsilon
    eps_norm = config_args['EPSILON'] / std_val
    # 3. Update the step size to be relative to the NORMALIZED box
    sampling_args['STEP_SIZE'] = 0.1 * eps_norm

    if model is None: return

    domain = setup_domain(config_args, x0)

    if sampling_args['METHOD'] == 'Langevin':
        sampler = LangevinSampler(
            model=model,
            c_vector=c_vector,
            domain=domain,
            device=config_args['DEVICE'],
            norm_type=sampling_args['NORM'],
            # --- SGLD Specifics ---
            steps=sampling_args['STEPS'],             
            walkers=sampling_args['WALKERS'],         
            step_size=sampling_args['STEP_SIZE'],     
            temperature=sampling_args['TEMP'],        # NEW: Critical for SGLD
            # --- Spatial Decorrelation ---
            nms_radius=0.1 * config_args['EPSILON'], # Radius to consider points "correlated"
            # --- Sobol Refinement ---
            top_k_refine=20,         # How many distinct peaks to probe
            k_box_epsilon=0.1*config_args['EPSILON'],      # Size of the "small box" for dense Sobol sampling
            sobol_samples_per_k=sampling_args['SOBOL_SAMPLES'],
            batch_size=sampling_args['BATCH_SIZE'] 
        )
    else:
        sampler = RandomSampler(
            model=model,
            c_vector=c_vector,
            domain=domain,
            device=config_args['DEVICE'],
            norm_type=sampling_args['NORM'],
            # --- SGLD Specifics ---
            num_samples = sampling_args['NUM_SAMPLES'],
            batch_size=sampling_args['BATCH_SIZE'] 
        )

    accumulated_inputs, accumulated_norms = sampler.run()

    return accumulated_inputs, accumulated_norms 

def run_alpaca(alpaca_args, accumulated_inputs, accumulated_norms):

    # Run Alpaca Analysis
    print(f"Running Alpaca analysis on {len(accumulated_norms)} samples...")

    alpaca_results = ALPACA.run_full_analysis(
        data=accumulated_norms,
        coords=accumulated_inputs, 
        gamma=alpaca_args['GAMMA'], 
        show_plot=False, 
    )

    # Store Results 
    result = {
        "Total_Samples": len(accumulated_norms),
        "Success": alpaca_results.optimization_succeeded,
        "Est_Endpoint": None,
        "Threshold_u": alpaca_results.pot_results.threshold,
        "Notes": ""
    }

    if alpaca_results.optimization_succeeded:
        print(f"\n> SUCCESS: Finite endpoint estimated: {alpaca_results.max_endpoint_from_optimization:.4f}")
        result["Est_Endpoint"] = alpaca_results.max_endpoint_from_optimization
        result["Notes"] = "Converged"
    else:
        print("\n> CONVERGENCE CHECK FAILED (Continuing to next iteration...)")
        result["Est_Endpoint"] = np.inf
        if alpaca_results.analysis_halted_reason:
            result["Notes"] = str(alpaca_results.analysis_halted_reason)
        else:
            result["Notes"] = "Infinite/Failed"

    print("\n--- Process Complete ---")
    
    # 5. Create and Print DataFrame
    df = pd.DataFrame([result])
    
    # Optional: formatting to make the table look nicer
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.5f}'.format)
    
    print("\n=== RESULTS TABLE ===")
    print(df)

    # Optional: Save to CSV
    # df.to_csv("lippot_iteration_results.csv", index=False)

    return df 
    
def run_lipmip_analysis(config_args, timeout=None):
    """
    Computes the Lipschitz constant using LipMIP.
    """
    # 1. Setup Model and Image (This loads them to GPU based on config)
    network, x0, c_vector, pred = setup_model_and_image(config_args) 
    
    # --- CRITICAL FIX: MOVE EVERYTHING TO CPU FOR LIPMIP ---
    # MIP solvers (Gurobi) run on CPU. Passing GPU tensors causes device mismatches.
    cpu_device = torch.device("cpu")
    network = network.to(cpu_device)
    c_vector = c_vector.to(cpu_device)
    x0 = x0.to(cpu_device)
    # -------------------------------------------------------

    # 2. Setup Domain (Using the CPU x0)
    domain = setup_domain(config_args, x0) 

    # 3. Flatten Bounds (LipMIP expects 1D vectors)
    if domain.box_low.dim() > 1:
        domain.box_low = domain.box_low.flatten()
    if domain.box_hi.dim() > 1:
        domain.box_hi = domain.box_hi.flatten()

    # 4. Flatten C-Vector
    if c_vector.dim() > 1:
        c_vector_lipmip = c_vector.flatten()
    else:
        c_vector_lipmip = c_vector

    norm = 'linf' 

    print("\n--- Starting LipMIP analysis (CPU) ---")
    
    # Initialize LipMIP
    lipmip_problem = LipMIP(
        network, 
        domain, 
        c_vector_lipmip,
        primal_norm=norm,
        verbose=True, 
        timeout=timeout,
        num_threads=24)
    
    print("Computing maximum Lipschitz constant (this may take a moment)...")
    try:
        result = lipmip_problem.compute_max_lipschitz()
        print("--- LipMIP analysis complete ---")
        return result.value
    except Exception as e:
        print(f"LipMIP Computation Failed: {e}")
        return None

def run_eclipse_analysis(config_args, eclipseMethod='ECLipsE_Fast'):

    model, x0, c_vector, pred_name = setup_model_and_image(config_args)

    print(f'\n--- Starting ECLipsE analysis ({eclipseMethod}) ---')
    
    class EclipseModel(nn.Module):
        def __init__(self, model, c_vector):
            super().__init__()
            
            # Fix 1: Correctly calculate in_features
            # c_vector shape is [1, 200]. We need 200.
            in_features = c_vector.shape[1]
            
            difference_layer = nn.Linear(in_features=in_features, out_features=1, bias=False)
            
            with torch.no_grad():
                difference_layer.weight.copy_(c_vector)
            difference_layer.weight.requires_grad = False

            # --- FIX START: Handle both raw Sequential and ReLUNet wrappers ---
            if hasattr(model, 'net'):
                # It is a ReLUNet wrapper
                layers = list(model.net.children())
            else:
                # It is a raw nn.Sequential or nn.Module
                layers = list(model.children())
            # --- FIX END ---
                
            new_layers = layers + [difference_layer]
            self.features = nn.Sequential(*new_layers)

        def forward(self,x):
            return self.features(x)

    # Move to CPU for analysis
    ec_model = EclipseModel(model.to('cpu'), c_vector.to('cpu'))

    LCE = LipConstEstimator(ec_model)
    LCE.model_review()
    
    result = LCE.estimate(eclipseMethod)
    print("--- ECLipsE analysis complete ---")
    return result


if __name__ == "__main__":
    
    # # --- CONFIGURATION ---
    # FOLDER_NAME = 'CIFAR'
    # MODEL_NAME = 'cnn_4layer_stride1_padding0' # Options: 'mnist_cnn_4layer', 'mnist_mlp_3layer', 'mnist_cnn_4layer_8'
    # WEIGHTS_FILENAME = f'{MODEL_NAME}.pth'
    # IMAGE_FILENAME = 'test_image_cifar.png' 

    # 1. Directory Map
    DIR_MAP = {
        'MNIST':        'MNIST',
        'CIFAR':        'CIFAR',
        'TinyImageNet': 'TinyImageNet'
    }

    # 2. Statistics Files (Normalization constants)
    STAT_FILES = {
        'MNIST':        'mnist_stat.pt',
        'CIFAR':        'cifar10_stat.pt',
        'TinyImageNet': 'tinyimagenet_stat.pt'
    }

    # --- CONFIGURATION ---
    FOLDER_NAME = 'TinyImageNet'
    
    # NOW YOU CAN USE THE SIMPLE KEY:
    MODEL_KEY = 'CNN_4Layer' 
    
    # Note: Ensure your weights filename logic matches how you saved them. 
    # If your files are named "cnn_4layer_stride1_padding0.pth", you might need to resolve the name here too.
    internal_name = get_model_function_name(FOLDER_NAME, MODEL_KEY) 
    WEIGHTS_FILENAME = f'{internal_name}.pth'
    
    IMAGE_FILENAME = 'test_image_0.png'

    stat_filename = STAT_FILES[FOLDER_NAME] 
    stat_path = os.path.join(FOLDER_NAME, stat_filename) 
    WEIGHT_NORMALISATION = torch.load(stat_path)
    
    GPU_ID = 0
    DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    # DEVICE = "cpu"
    # EPSILON = 8/255 
    EPSILON = 255/255 

    config_args = {
        "FOLDER_NAME": FOLDER_NAME,
        "MODEL_NAME": MODEL_KEY,
        "WEIGHTS_FILENAME": WEIGHTS_FILENAME,
        "WEIGHT_NORMALISATION": WEIGHT_NORMALISATION,
        "IMAGE_FILENAME": IMAGE_FILENAME,
        "EPSILON": EPSILON,
        "GPU_ID": GPU_ID,
        "DEVICE": DEVICE
    }

    # --- 3. PARAMETERS ---
    STEPS = 2*4096
    WALKERS = 128
    SOBOL_SAMPLES = 4000
    BATCH_SIZE = 64
    TEMP = 1e-3
    # NORM = 'linf' 
    NORM = 'l2' 

    sampling_args = {
        "METHOD": 'Langevin', 
        "STEPS": STEPS,
        "WALKERS": WALKERS,
        "TEMP": TEMP,
        "STEP_SIZE": 0.05 * EPSILON, 
        "SOBOL_SAMPLES": SOBOL_SAMPLES, 
        "BATCH_SIZE": BATCH_SIZE, 
        "NORM": NORM
    }

    # sampling_args = {
    #     "METHOD": 'Random', 
    #     "NUM_SAMPLES": 40000,
    #     "BATCH_SIZE": BATCH_SIZE, 
    #     "NORM": NORM
    # }

    GAMMA = 0.001 
    alpaca_args = { "GAMMA": GAMMA }

    # lirpa_args = {'METHOD': 'alpha-CROWN'}
    lirpa_args = {'METHOD': 'CROWN-IBP'}

    eclipseMethod = 'ECLipsE_Fast' 
    # eclipseMethod = 'ECLipsE' 

    print(f"--- Running Single Test on {MODEL_KEY} ---")
    print(f"Device: {DEVICE}")

    # --- 4. EXECUTION ---
    
    # # A. LipMIP
    # st = time.time() 
    # mipres = run_lipmip_analysis(config_args, timeout=120)
    # mip_time = time.time() - st

    # # B. LiRPA
    # st = time.time() 
    # lirpa_res = run_lirpa_analysis(lirpa_args, config_args)
    # lirpa_time = time.time() - st

    # # C. Sampling
    # st = time.time() 
    # acc_inputs, acc_norms = run_lipschitz_sampling(config_args, sampling_args)
    # print(np.max(acc_norms), np.min(acc_norms))
    # sample_time = time.time() - st 

    # # D. Alpaca
    # st = time.time() 
    # alpacares = run_alpaca(alpaca_args, acc_inputs, acc_norms) 
    # alpaca_time = time.time() - st 

    # D. eclipse
    st = time.time() 
    eclipseres = run_eclipse(config_args, eclipseMethod) 
    eclipse_time = time.time() - st 

    print(eclipseres, eclipse_time)

    # # print('\n--- Final Results ---')
    # # print(f'LipMIP Result: {mipres:.4f} (Time: {mip_time:.2f}s)')
    # print(f'LiRPA Result: {lirpa_res:.4f} (Time: {lirpa_time:.2f}s)')
    # print(f'Alpaca Result: {alpacares["Est_Endpoint"].iloc[0]:.4f} (Time: {alpaca_time + sample_time:.2f}s)')