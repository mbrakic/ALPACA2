import numpy as np
import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import Flatten
from auto_LiRPA.jacobian import JacobianOP, GradNorm
from LipPOT.LipPOT_TA import LipPOT


def build_model(in_ch=3, in_dim=32):
    model = nn.Sequential(
        Flatten(),
        nn.Linear(in_ch*in_dim**2, 100),
        nn.ReLU(),
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 10),
    )
    return model


def example_local_lipschitz(model_ori, x0, bound_opts, eps, device):
    """Example: computing Linf local Lipschitz constant."""

    class LocalLipschitzWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.grad_norm = GradNorm(norm=1)

        def forward(self, x, mask):
            y = self.model(x)
            y_selected = y.matmul(mask)
            jacobian = JacobianOP.apply(y_selected, x)
            lipschitz = self.grad_norm(jacobian)
            return lipschitz

    mask = torch.zeros(10, 1, device=device)
    mask[1, 0] = 1
    model = BoundedModule(LocalLipschitzWrapper(model_ori), (BoundedTensor(x0), mask),
                          bound_opts=bound_opts, device=device)

    y = model_ori(x0.requires_grad_(True))
    ret_ori = torch.autograd.grad(y[:, 1].sum(), x0)[0].abs().flatten(1).sum(dim=-1).view(-1)
    ret_new = model(x0, mask).view(-1)
    assert torch.allclose(ret_ori, ret_new)

    ret = []
    for eps in eps:
        x = BoundedTensor(x0, PerturbationLpNorm(norm=np.inf, eps=eps))
        lip = []
        for i in range(mask.shape[0]):
            mask.zero_()
            mask[i, 0] = 1
            ub = model.compute_jacobian_bounds((x, mask), bound_lower=False)[1]
            lip.append(ub)
        lip = torch.concat(lip).max()
        print(f'Linf local Lipschitz constant for eps={eps:.5f}: {lip.item()}')
        ret.append(lip.detach())

    return ret


def compute_lipschitz(model_ori, x0, eps, bound_opts=None, device='cpu'):
    results = [[] for _ in range(3)]
    model_ori = model_ori.to(device)
    x0 = x0.to(device)
    print('Model:', model_ori)

    results = example_local_lipschitz(model_ori, x0, bound_opts, eps, device)

    return results


def get_norms_batched(model, x0, eps, num_samples, device, batch_size=128):
    """
    Efficiently samples points in an Linf hypercube and computes the max L1 
    gradient norm for each sample using batching. Includes a special case for eps=0.

    Args:
        model (nn.Module): The neural network model.
        x0 (torch.Tensor): The center point of the hypercube. 
                           Should have a batch dimension of 1 (e.g., [1, C, H, W]).
        eps (float): The radius of the Linf hypercube.
        num_samples (int): The number of points to sample from the hypercube.
        device (torch.device): The device to run computations on (e.g., 'cuda' or 'cpu').
        batch_size (int): The number of samples to process in each batch to manage memory.

    Returns:
        torch.Tensor: A tensor of shape [num_samples] containing the maximum L1 
                      gradient norm for each sampled point.
    """
    model.to(device)
    x0 = x0.to(device)
    model.eval() # Set the model to evaluation mode

    # --- Special case for eps = 0 ---
    # If eps is 0, all "samples" are just the center point x0.
    # We can compute this once and repeat the result, which is much faster.
    if eps == 0:
        print("Epsilon is 0. Computing gradient norm at the center point x0 once.")
        x0.requires_grad_(True)
        y = model(x0)
        num_outputs = y.shape[-1]
        
        point_grad_norms = []
        for j in range(num_outputs):
            output_scalar = y[0, j]
            grad = torch.autograd.grad(
                outputs=output_scalar,
                inputs=x0,
                retain_graph=True
            )[0]
            l1_norm = torch.sum(torch.abs(grad))
            point_grad_norms.append(l1_norm)
        
        # Detach from graph, move to device, and repeat for num_samples
        max_norm = torch.stack(point_grad_norms).max()
        return max_norm.detach().to(device).repeat(num_samples)


    # --- Original logic for eps > 0 ---
    all_max_norms = []
    print(f"Sampling {num_samples} points from the Linf hypercube in batches...")

    # Process in mini-batches to avoid memory issues with very large num_samples
    for i in range(0, num_samples, batch_size):
        current_batch_size = min(batch_size, num_samples - i)
        
        # 1. Create a batch of random samples within the Linf hypercube
        # The base point x0 is broadcasted to match the perturbation's shape
        perturbation = (torch.rand(current_batch_size, *x0.shape[1:], device=device) * 2 - 1) * eps
        x_batch = x0 + perturbation
        x_batch.requires_grad_(True)

        # 2. Single forward pass for the entire batch
        y_batch = model(x_batch)
        num_outputs = y_batch.shape[-1]

        # This will store the L1 norms for each sample for each output neuron
        # Shape: [num_outputs, current_batch_size]
        batch_norms_per_neuron = []

        # 3. Compute gradient norms for each output neuron across the whole batch
        for j in range(num_outputs):
            # Select the j-th output for all samples in the batch
            output_scalars = y_batch[:, j]
            
            # We need to compute the gradient of a sum for autograd to work on a batch
            # The gradient of the sum is the sum of gradients, and since we pass
            # grad_outputs=torch.ones_like(...), we effectively get the gradient for each sample.
            grad = torch.autograd.grad(
                outputs=output_scalars,
                inputs=x_batch,
                grad_outputs=torch.ones_like(output_scalars),
                retain_graph=True
            )[0]
            
            # 4. Calculate the L1 norm of the gradients for the entire batch
            # We sum the absolute values over all input dimensions (C, H, W)
            l1_norms = torch.sum(torch.abs(grad.view(current_batch_size, -1)), dim=1)
            batch_norms_per_neuron.append(l1_norms)

        # 5. Find the maximum norm across all output neurons for each sample in the batch
        # Stack into a [num_outputs, current_batch_size] tensor and find the max along dim 0
        max_norms_for_batch = torch.stack(batch_norms_per_neuron).max(dim=0)[0]
        all_max_norms.append(max_norms_for_batch)

    print("Sampling complete.")
    # Concatenate results from all mini-batches
    grad_norms = torch.cat(all_max_norms).cpu().numpy() 
    np.random.shuffle(grad_norms)
    return grad_norms

def run_lippot(model, x0, eps, num_samples, device, batch_size=128):
    grad_norms = get_norms_batched(model, x0, eps, num_samples, device, batch_size)

    print('Max observed = ', max(grad_norms))

    LipPOT.run_full_analysis(
        data = grad_norms, 
        gamma=0.05, 
        n_search_samples=10000, 
        show_plot=False,
        use_fine_graining=True, 
        verbose=True
    )

    return


def diagnostic_comparison(model_ori, x0, eps_val, device='cpu'):

    """
    Diagnostic function to compare auto-LiRPA bounds with empirical sampling
    at various points to identify where the discrepancy occurs.
    """
    
    model_ori = model_ori.to(device)
    x0 = x0.to(device)
    
    print("=" * 60)
    print(f"Diagnostic Test for eps = {eps_val}")
    print("=" * 60)
    
    # 1. Compute exact gradient norm at the center point x0
    x0_test = x0.clone().requires_grad_(True)
    y = model_ori(x0_test)
    num_outputs = y.shape[-1]
    
    center_norms = []
    for j in range(num_outputs):
        grad = torch.autograd.grad(y[0, j], x0_test, retain_graph=True)[0]
        l1_norm = grad.abs().sum().item()
        center_norms.append(l1_norm)
    
    print(f"\n1. Gradient norms at center point x0:")
    for j, norm in enumerate(center_norms):
        print(f"   Output {j}: {norm:.6f}")
    print(f"   Maximum: {max(center_norms):.6f}")

    # # 2. Sample corners of the L∞ hypercube (most extreme points)
    # if eps_val > 0:
    #     print(f"\n2. Gradient norms at hypercube corners:")
    #     corner_max_norms = []
        
    #     # Test a few random corners
    #     for corner_idx in range(min(10, 2**x0.numel())):
    #         # Create a random corner point
    #         signs = torch.randint(0, 2, x0.shape, device=device) * 2 - 1
    #         x_corner = x0 + signs * eps_val
    #         x_corner.requires_grad_(True)
            
    #         y_corner = model_ori(x_corner)
    #         corner_norms = []
    #         for j in range(num_outputs):
    #             grad = torch.autograd.grad(y_corner[0, j], x_corner, retain_graph=True)[0]
    #             l1_norm = grad.abs().sum().item()
    #             corner_norms.append(l1_norm)
            
    #         max_norm = max(corner_norms)
    #         corner_max_norms.append(max_norm)
    #         print(f"   Corner {corner_idx}: {max_norm:.6f}")
        
    #     print(f"   Maximum across corners: {max(corner_max_norms):.6f}")
    
    # 3. Run auto-LiRPA's computation
    print(f"\n3. Auto-LiRPA bounds:")
    
    class LocalLipschitzWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.grad_norm = GradNorm(norm=1)
        
        def forward(self, x, mask):
            y = self.model(x)
            y_selected = y.matmul(mask)
            jacobian = JacobianOP.apply(y_selected, x)
            lipschitz = self.grad_norm(jacobian)
            return lipschitz
    
    mask = torch.zeros(num_outputs, 1, device=device)
    model_wrapped = BoundedModule(
        LocalLipschitzWrapper(model_ori), 
        (BoundedTensor(x0), mask),
        bound_opts={'conv_mode': 'patches'},
        device=device
    )
    
    x_bounded = BoundedTensor(x0, PerturbationLpNorm(norm=np.inf, eps=eps_val))
    
    lirpa_bounds = []
    for i in range(num_outputs):
        mask.zero_()
        mask[i, 0] = 1
        
        # Get both lower and upper bounds
        lb, ub = model_wrapped.compute_jacobian_bounds(
            (x_bounded, mask), 
            bound_lower=True,
            bound_upper=True
        )
        
        lirpa_bounds.append((lb.item() if lb is not None else None, ub.item()))
        print(f"   Output {i}: Lower={lb.item() if lb is not None else 'N/A':.6f}, Upper={ub.item():.6f}")
    
    max_ub = max(ub for _, ub in lirpa_bounds)
    print(f"   Maximum upper bound: {max_ub:.6f}")

    # 4. Extensive random sampling
    print(f"\n4. Empirical sampling (1000 random points):")
    max_empirical = 0
    
    for _ in range(1000):
        if eps_val > 0:
            perturbation = (torch.rand_like(x0) * 2 - 1) * eps_val
            x_sample = x0 + perturbation
        else:
            x_sample = x0.clone()
        
        x_sample.requires_grad_(True)
        y_sample = model_ori(x_sample)
        
        sample_norms = []
        for j in range(num_outputs):
            grad = torch.autograd.grad(y_sample[0, j], x_sample, retain_graph=True)[0]
            l1_norm = grad.abs().sum().item()
            sample_norms.append(l1_norm)
        
        max_empirical = max(max_empirical, max(sample_norms))
    
    print(f"   Maximum found: {max_empirical:.6f}")
    
    # 5. Comparison and diagnosis
    print(f"\n5. DIAGNOSIS:")
    print(f"   Auto-LiRPA upper bound: {max_ub:.6f}")
    print(f"   Empirical maximum:      {max_empirical:.6f}")
    
    if max_empirical > max_ub * 1.001:  # Allow 0.1% tolerance for numerical errors
        print(f"   ⚠️  ERROR: Empirical value exceeds provable upper bound by {(max_empirical/max_ub - 1)*100:.2f}%")
        print(f"   This indicates a problem with either:")
        print(f"   - Auto-LiRPA's bound computation (too loose or buggy)")
        print(f"   - Different interpretation of the norm")
        print(f"   - Numerical instabilities")
    else:
        print(f"   ✓ Bounds are consistent (empirical ≤ provable upper bound)")
    
    return max_ub, max_empirical


# Additional test to verify the exact computation at a specific point
def verify_single_point_computation(model_ori, x0, device='cpu'):
    """
    Verify that both methods compute the same thing at a single point (eps=0).
    """
    print("\n" + "=" * 60)
    print("Single Point Verification (eps=0)")
    print("=" * 60)
    
    model_ori = model_ori.to(device)
    x0 = x0.to(device)
    
    # Manual computation
    x0_test = x0.clone().requires_grad_(True)
    y = model_ori(x0_test)
    
    manual_norms = []
    for j in range(y.shape[-1]):
        grad = torch.autograd.grad(y[0, j], x0_test, retain_graph=True)[0]
        l1_norm = grad.abs().sum().item()
        manual_norms.append(l1_norm)
    
    print(f"Manual computation:")
    for j, norm in enumerate(manual_norms):
        print(f"  Output {j}: {norm:.6f}")
    print(f"  Maximum: {max(manual_norms):.6f}")
    
    # Your function's computation
    from_your_function = get_norms_batched(model_ori, x0, 0, 1, device, batch_size=1)[0]
    print(f"\nYour function's result: {from_your_function:.6f}")
    
    # Check consistency
    if abs(from_your_function - max(manual_norms)) < 1e-6:
        print("✓ Your function matches manual computation")
    else:
        print(f"⚠️  Discrepancy detected: {abs(from_your_function - max(manual_norms)):.6e}")
    
    return max(manual_norms), from_your_function


# Run the diagnostic
if __name__ == '__main__':
    torch.manual_seed(0)
    
    # Use the same setup as your code
    model_ori = build_model(in_dim=8)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x0 = torch.randn(1, 3, 8, 8, device=device)
    
    # First verify your computation is correct at a single point
    verify_single_point_computation(model_ori, x0, device)
    
    # Then run diagnostic for different eps values
    for eps_val in [0, 1./255, 4./255]:
        lirpa_bound, empirical_max = diagnostic_comparison(model_ori, x0, eps_val, device)
        print("\n")