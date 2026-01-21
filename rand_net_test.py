import argparse # Make sure to add this import at the top of your file
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import itertools
import pandas as pd
import time

# Import class definitions for loading objects
from lipMIP.relu_nets import ReLUNet
import lipMIP.neural_nets.data_loaders as data_loaders
from lipMIP.lipMIP import LipMIP
from lipMIP.hyperbox import Hyperbox

# Imports for auto_LiRPA
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.jacobian import JacobianOP, GradNorm

# from LipPOT.LipPOT_TA import LipPOT
from ALPACA.ALPACA import ALPACA

from EClipsE.LipConstEstimator import LipConstEstimator

# --- 1. Global Device Definition ---
# Define the device at the start and use it consistently.
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(f"--- Using device: {device} ---")


def create_network(net_dimensions):
    # network dimensions is an array like [4,8,16,2] for example
    network = ReLUNet(net_dimensions, bias=True)
    input_dimension = net_dimensions[0] 

    return network, input_dimension 

class DummyDataset(Dataset): 
    def __init__(self, num_samples, dimension, num_classes=2):
        self.num_samples = num_samples
        self.dimension = dimension
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a single random input tensor and a random target label
        inputs = torch.rand(self.dimension)
        target = torch.randint(0, self.num_classes, (1,)).squeeze()
        return inputs, target


def create_data(dimension, num_samples, batch_size):
    dataset = DummyDataset(num_samples = num_samples, dimension = dimension)
    dataloader = DataLoader(dataset, batch_size = batch_size) 
    return dataloader

def get_local_lipschitz_norms(network, c_vector, data_loader, norm_type='l1'):
    """
    Computes the norm of the gradient for each input in the dataset, which
    corresponds to the local Lipschitz constant at that point.

    Args:
        network (nn.Module): The neural network.
        data_loader (DataLoader): The data loader for the dataset.
        norm_type (str): The type of norm to compute ('l1', 'l2', or 'linf').

    Returns:
        np.array: An array of the gradient norms for each input sample.
    """
    network.to(device) 
    network.eval()
    grad_norms_list = []

    # Define the 'ord' parameter for torch.linalg.norm based on the norm_type
    # be cautios because if you want the l1 lipschitz, you need to take the linf
    # perturbation and vice versa. it's a 1/p + 1/q thing.
    if norm_type == 'l1':
        # ord = 1
        ord = float('inf')
    elif norm_type == 'l2':
        ord = 2
    elif norm_type == 'linf':
        # ord = float('inf')
        ord = 1
    else:
        raise ValueError("norm_type must be 'l1', 'l2', or 'linf'")


    for inputs, _ in data_loader:
        # We need to tell PyTorch to track gradients for the input tensor
        inputs = inputs.to(device) 
        inputs.requires_grad_(True)

        # Forward pass
        outputs = network(inputs)

        # Define the scalar output as the dot product of the output logits
        # and the c_vector. This is the margin whose gradient we are interested in.
        scalar_output = (outputs * c_vector).sum()

        # --- FIX: Backward pass to compute gradients ---
        # This is the crucial step that populates the .grad attribute.
        scalar_output.backward()

        # The gradients are computed with respect to the inputs.
        # We get the gradient from inputs.grad, not outputs.grad.
        gradients = inputs.grad

        # batch_norms = torch.linalg.norm(gradients, ord=ord, dim=0)
        batch_norms = torch.linalg.norm(gradients, ord=ord, dim=1)

        grad_norms_list.append(batch_norms.cpu().detach())

        # Zero out the gradients for the next batch
        network.zero_grad()
        inputs.grad.zero_()

    grad_norms = torch.cat(grad_norms_list).numpy()
    return grad_norms

def run_lipmip_analysis(network, c_vector,  domain, norm):
    """
    Computes the Lipschitz constant using LipMIP.
    """
    print("\n--- Starting LipMIP analysis ---")
    # network.to(device)
    print("Setting up LipMIP problem...")
    lipmip_problem = LipMIP(network, domain, c_vector,
                            primal_norm = norm, verbose=True)
    print("Computing maximum Lipschitz constant (this may take a moment)...")
    result = lipmip_problem.compute_max_lipschitz()
    print("--- LipMIP analysis complete ---")
    return result

def run_lirpa_analysis(network, c_vector, domain):
    """
    Computes a LIRPA-based upper bound on the Lipschitz constant, now with
    a robust sanity check that validates the BoundedModule at eps=0.
    """
    print("\n--- Starting auto-LiRPA analysis ---")
    network.to(device) # Ensure model is on the right device
    # necessary for the way lirpa wants things
    c_vector = c_vector.unsqueeze(1).to(device)

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

    # Define the center point and perturbation size (epsilon) for the domain.
    center_x0 = ((domain.box_low + domain.box_hi) / 2.0).unsqueeze(0)
    eps = ((domain.box_hi - domain.box_low) / 2.0)[0].item()
    print(f"Representing domain as Linf ball: center={center_x0.cpu().numpy().squeeze()}, eps={eps:.4f}")

    ### ROBUST SANITY CHECK START ###
    print("\n--- Running Sanity Check ---")
    # 1. Initialize BoundedModule with a dummy BoundedTensor at eps=0 for tracing.
    model_wrapper = MarginLipschitzWrapper(network)
    x_dummy_for_tracing = BoundedTensor(center_x0, PerturbationLpNorm(norm=np.inf, eps=0))
    lirpa_model = BoundedModule(
        model_wrapper, (x_dummy_for_tracing, c_vector),
        bound_opts={'conv_mode': 'patches'}, device=device)

    # 2. Calculate the ground truth value using standard PyTorch autograd.
    center_x0_grad = center_x0.clone().requires_grad_(True)
    y = network(center_x0_grad)
    margin = y[:, 0] - y[:, 1]
    grad_autograd = torch.autograd.grad(margin.sum(), center_x0_grad)[0]
    lip_autograd = grad_autograd.abs().flatten(1).sum(dim=-1)
    print(f"Lipschitz at center (via autograd): {lip_autograd.item():.6f}")

    # 3. Get the value from a forward pass of the BoundedModule.
    #    When called with a regular tensor, it should be numerically identical.
    lip_lirpa_forward = lirpa_model(center_x0, c_vector)
    print(f"Lipschitz at center (via LiRPA fwd pass): {lip_lirpa_forward.item():.6f}")

    # 4. Assert that the two methods produce the same result.
    assert torch.allclose(lip_autograd, lip_lirpa_forward.flatten()), \
        "Sanity check failed: BoundedModule forward pass does not match autograd."
    print("✅ Sanity check passed: LiRPA forward pass matches autograd.")
    print("--- Sanity Check Complete ---\n")
    ### SANITY CHECK END ###

    # Now, compute the actual bound over the full domain with the real epsilon.
    x_bounded = BoundedTensor(center_x0, PerturbationLpNorm(norm=np.inf, eps=eps))

    print("Computing upper bound with CROWN-IBP...")
    # We use the same lirpa_model instance created for the sanity check.
    # _, ub = lirpa_model.compute_bounds(x=(x_bounded, c_vector), method='IBP+backward')
    ub = lirpa_model.compute_jacobian_bounds(x=(x_bounded, c_vector), bound_lower=False)[1]
    print("--- auto-LiRPA analysis complete ---")
    return ub.item()

def run_lippot(network, c_vector, data_loader, norm_type, gamma):

    grad_norms = get_local_lipschitz_norms(network, c_vector, data_loader, norm_type)

    print('\n--- Starting LipPOT analysis ---')
    print('Max observed local Lipschitz constant = ', max(grad_norms))

    results = ALPACA.run_full_analysis(
        data = grad_norms,
        gamma=gamma,
        n_search_samples=10000,
        show_plot=False,
        use_fine_graining=True,
        verbose=True
    )
    print("--- LipPOT analysis complete ---")
    return results

def run_eclipse(model, c_vector, method):
    print(f'\n--- Starting ECLipsE analysis ({method}) ---')
    class EclipseModel(nn.Module):
        def __init__(self, model):
            super().__init__()
            difference_layer = nn.Linear(in_features=len(c_vector), out_features=1, bias=False)
            with torch.no_grad():
                difference_layer.weight.copy_(c_vector)
            difference_layer.weight.requires_grad = False

            layers = list(model.net.children())
            new_layers = layers + [difference_layer]
            self.features = nn.Sequential(*new_layers)

        def forward(self,x):
            return self.features(x)

    ec_model = EclipseModel(model.to('cpu'))

    LCE = LipConstEstimator(ec_model)
    # LCE = LipConstEstimator(model)
    LCE.model_review()
    
    result = LCE.estimate(method)
    print("--- ECLipsE analysis complete ---")
    return result


# --- Main Execution Block ---
if __name__ == '__main__':
    # --- 1. Set up Argument Parser ---
    parser = argparse.ArgumentParser(
        description="Run Lipschitz constant estimation algorithms on a saved network.",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )
    parser.add_argument(
        'net_dimensions',
        type=int,
        nargs='*',  # accepts one or more space-separated integers
        default = [1,10,128,10,2],
        help='A list of all layer dimensions, including input and output (e.g., 10 128 10).'
    )
    parser.add_argument(
        "--save",
        action="store_true", # This makes it a flag; if present, its value is True.
        help="If included, saves the results DataFrame to a CSV file."
    )
    args = parser.parse_args()

    torch.manual_seed(42)

    # --- 2. Use Parsed Arguments ---
    net_dimensions = args.net_dimensions
    batch_size = 512 
    num_samples = int(1e5)
    SAVE_RESULTS = args.save

    model, DIMENSION = create_network(net_dimensions)

    norm = 'linf'
    # norm = 'l1'
    # norm = 'l2'
    gamma = 0.0001

    # List to store results dictionaries
    results_list = []

    c_vector = torch.tensor([1.,-1.], device = device)

    if model:

        # --- Define the domain and move its tensors to the correct device ---
        domain = Hyperbox.build_unit_hypercube(DIMENSION)
        domain.box_low = domain.box_low.to(device)
        domain.box_hi = domain.box_hi.to(device)

        # both = True
        both = False

        if not(norm=='l2'):
            # --- Run LipMIP analysis ---
            start_time = time.time()
            mipres = run_lipmip_analysis(model, c_vector, domain, norm)
            end_time = time.time()
            results_list.append({
                'Algorithm': 'LipMIP',
                'Upper Bound': mipres.value,
                'Lower Bound': np.nan,
                'Execution Time (s)': end_time - start_time,
                'LipPOT Exit Reason': ''
            })

            # --- Run auto-LiRPA analysis ---
            start_time = time.time()
            lirpares = run_lirpa_analysis(model, c_vector, domain)
            end_time = time.time()
            results_list.append({
                'Algorithm': 'auto-LiRPA',
                'Upper Bound': lirpares,
                'Lower Bound': np.nan,
                'Execution Time (s)': end_time - start_time,
                'LipPOT Exit Reason': ''
            })

        # # --- Run LipPOT analysis (for all norms) ---
        # start_time = time.time()

        # # data_loader = create_data(num_samples, DIMENSION, batch_size)
        # data_loader = create_data(DIMENSION, num_samples, batch_size)

        # potres = run_lippot(model, c_vector, data_loader, norm, gamma)
        # end_time = time.time()

        # exit_reason = ''
        # # --- Early Exit Checks for LipPOT ---
        # if potres.pot_results.gpd_fit_failed or not potres.pot_results.central_gpd_params or potres.pot_results.central_gpd_params.shape >= 0:
        #     exit_reason = potres.pot_results.error_message or "Initial GPD fit implies an infinite endpoint (shape >= 0)."

        # results_list.append({
        #     'Algorithm': 'LipPOT',
        #     'Upper Bound': potres.final_L_high,
        #     'Lower Bound': potres.final_L_low,
        #     'Execution Time (s)': end_time - start_time,
        #     'LipPOT Exit Reason': exit_reason
        # })

        # # --- Run ECLipsE analysis (for all norms) ---
        # start_time = time.time()
        # # eclipseres = run_eclipse(model, 'ECLipsE_Fast')
        # eclipseres = run_eclipse(model, c_vector, 'ECLipsE')
        # end_time = time.time()
        # results_list.append({
        #     'Algorithm': 'ECLipsE',
        #     'Upper Bound': eclipseres.cpu().numpy().item(),
        #     'Lower Bound': np.nan,
        #     'Execution Time (s)': end_time - start_time,
        #     'LipPOT Exit Reason': ''
        # })

        # --- Create and Print Final DataFrame ---
        results_df = pd.DataFrame(results_list)
        results_df = results_df.set_index('Algorithm')

        print("\n" + "="*85)
        print(" " * 28 + "LIPSCHITZ CONSTANT RESULTS")
        print("="*85)
        print(results_df.to_string(float_format="{:.4f}".format))
        print("="*85)

        # --- Optionally Save DataFrame to CSV ---
        if SAVE_RESULTS:
            save_dir = "results"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Create a unique filename with a timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"results_{folder_path}_{timestamp}.csv"
            filepath = os.path.join(save_dir, filename)

            # Save the DataFrame
            results_df.to_csv(filepath)
            print(f"\n✅ Results successfully saved to: {filepath}")