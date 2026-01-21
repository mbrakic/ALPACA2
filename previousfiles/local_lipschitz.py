
"""Examples of computing local lipschitz constants.

We show examples of:
- Computing Linf local Lipschitz constants
"""

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
    gradient norm for each sample using batching. Includes a special case for eps=0
    that uses the Jacobian to find the local Lipschitz constant.

    Args:
        model (nn.Module): The neural network model.
        x0 (torch.Tensor): The center point of the hypercube. 
                           Should have a batch dimension of 1 (e.g., [1, C, H, W]).
        eps (float): The radius of the Linf hypercube.
        num_samples (int): The number of points to sample from the hypercube.
        device (torch.device): The device to run computations on (e.g., 'cuda' or 'cpu').
        batch_size (int): The number of samples to process in each batch to manage memory.

    Returns:
        torch.Tensor or np.ndarray: A tensor/array of shape [num_samples] containing 
                                    the maximum L1 gradient norm for each sampled point.
    """
    model.to(device)
    x0 = x0.to(device)
    model.eval() # Set the model to evaluation mode

    # --- Special case for eps = 0 (REWRITTEN) ---
    # If eps is 0, all "samples" are the center point x0. The Lipschitz constant
    # is the induced infinity-norm of the Jacobian at that single point.
    if eps == 0:
        print("Epsilon is 0. Computing Jacobian norm at the center point x0 once.")
        
        # Remove the batch dimension for the single point calculation
        x0_point = x0.squeeze(0)
        x0_point.requires_grad_(True)

        # The jacobian function expects a callable that takes the input
        # and returns the output. We handle reshaping inside the lambda.
        try:
            jacobian_matrix = torch.autograd.functional.jacobian(
                lambda x: model(x.unsqueeze(0)), # Add batch dim back for model
                x0_point,
                create_graph=False
            )
        except Exception as e:
            print(f"An error occurred during Jacobian computation: {e}")
            return torch.tensor([float('nan')]).repeat(num_samples)

        # The output of jacobian might have extra dimensions (e.g., [1, num_outputs, ...])
        # We need to reshape it to [num_outputs, num_input_features]
        num_outputs = jacobian_matrix.shape[1]
        jacobian_matrix = jacobian_matrix.squeeze().view(num_outputs, -1)

        # The l-infinity induced norm is the maximum absolute row sum (which is an L1 norm of the row)
        max_norm = torch.sum(torch.abs(jacobian_matrix), dim=1).max()
        max_norm = max_norm.detach().cpu().numpy()
        
        return max_norm

    else:
        # --- Original logic for eps > 0 (UNCHANGED) ---
        all_max_norms = []
        print(f"Sampling {num_samples} points from the Linf hypercube in batches...")

        # Process in mini-batches to avoid memory issues
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            
            perturbation = (torch.rand(current_batch_size, *x0.shape[1:], device=device) * 2 - 1) * eps
            x_batch = x0 + perturbation
            x_batch.requires_grad_(True)

            y_batch = model(x_batch)
            num_outputs = y_batch.shape[-1]

            batch_norms_per_neuron = []
            for j in range(num_outputs):
                output_scalars = y_batch[:, j]
                
                grad = torch.autograd.grad(
                    outputs=output_scalars,
                    inputs=x_batch,
                    grad_outputs=torch.ones_like(output_scalars),
                    retain_graph=True
                )[0]
                
                l1_norms = torch.sum(torch.abs(grad.view(current_batch_size, -1)), dim=1)
                batch_norms_per_neuron.append(l1_norms)

            max_norms_for_batch = torch.stack(batch_norms_per_neuron).max(dim=0)[0]
            all_max_norms.append(max_norms_for_batch)

        print("Sampling complete.")
        grad_norms = torch.cat(all_max_norms).cpu().numpy()
        return grad_norms


# def get_norms_batched(model, x0, eps, num_samples, device, batch_size=128):
#     """
#     Efficiently samples points in an Linf hypercube and computes the max L1 
#     gradient norm for each sample using batching. Includes a special case for eps=0.

#     Args:
#         model (nn.Module): The neural network model.
#         x0 (torch.Tensor): The center point of the hypercube. 
#                            Should have a batch dimension of 1 (e.g., [1, C, H, W]).
#         eps (float): The radius of the Linf hypercube.
#         num_samples (int): The number of points to sample from the hypercube.
#         device (torch.device): The device to run computations on (e.g., 'cuda' or 'cpu').
#         batch_size (int): The number of samples to process in each batch to manage memory.

#     Returns:
#         torch.Tensor: A tensor of shape [num_samples] containing the maximum L1 
#                       gradient norm for each sampled point.
#     """
#     model.to(device)
#     x0 = x0.to(device)
#     model.eval() # Set the model to evaluation mode

#     # --- Special case for eps = 0 ---
#     # If eps is 0, all "samples" are just the center point x0.
#     # We can compute this once and repeat the result, which is much faster.
#     if eps == 0:
#         print("Epsilon is 0. Computing gradient norm at the center point x0 once.")
#         x0.requires_grad_(True)
#         y = model(x0)
#         num_outputs = y.shape[-1]
        
#         point_grad_norms = []
#         for j in range(num_outputs):
#             output_scalar = y[0, j]
#             grad = torch.autograd.grad(
#                 outputs=output_scalar,
#                 inputs=x0,
#                 retain_graph=True
#             )[0]
#             l1_norm = torch.sum(torch.abs(grad))
#             point_grad_norms.append(l1_norm)
        
#         # Detach from graph, move to device, and repeat for num_samples
#         max_norm = torch.stack(point_grad_norms).max()
#         return max_norm.detach().to(device).repeat(num_samples)


#     # --- Original logic for eps > 0 ---
#     all_max_norms = []
#     print(f"Sampling {num_samples} points from the Linf hypercube in batches...")

#     # Process in mini-batches to avoid memory issues with very large num_samples
#     for i in range(0, num_samples, batch_size):
#         current_batch_size = min(batch_size, num_samples - i)
        
#         # 1. Create a batch of random samples within the Linf hypercube
#         # The base point x0 is broadcasted to match the perturbation's shape
#         perturbation = (torch.rand(current_batch_size, *x0.shape[1:], device=device) * 2 - 1) * eps
#         x_batch = x0 + perturbation
#         x_batch.requires_grad_(True)

#         # 2. Single forward pass for the entire batch
#         y_batch = model(x_batch)
#         num_outputs = y_batch.shape[-1]

#         # This will store the L1 norms for each sample for each output neuron
#         # Shape: [num_outputs, current_batch_size]
#         batch_norms_per_neuron = []

#         # 3. Compute gradient norms for each output neuron across the whole batch
#         for j in range(num_outputs):
#             # Select the j-th output for all samples in the batch
#             output_scalars = y_batch[:, j]
            
#             # We need to compute the gradient of a sum for autograd to work on a batch
#             # The gradient of the sum is the sum of gradients, and since we pass
#             # grad_outputs=torch.ones_like(...), we effectively get the gradient for each sample.
#             grad = torch.autograd.grad(
#                 outputs=output_scalars,
#                 inputs=x_batch,
#                 grad_outputs=torch.ones_like(output_scalars),
#                 retain_graph=True
#             )[0]
            
#             # 4. Calculate the L1 norm of the gradients for the entire batch
#             # We sum the absolute values over all input dimensions (C, H, W)
#             l1_norms = torch.sum(torch.abs(grad.view(current_batch_size, -1)), dim=1)
#             batch_norms_per_neuron.append(l1_norms)

#         # 5. Find the maximum norm across all output neurons for each sample in the batch
#         # Stack into a [num_outputs, current_batch_size] tensor and find the max along dim 0
#         max_norms_for_batch = torch.stack(batch_norms_per_neuron).max(dim=0)[0]
#         all_max_norms.append(max_norms_for_batch)

#     print("Sampling complete.")
#     # Concatenate results from all mini-batches
#     grad_norms = torch.cat(all_max_norms).cpu().numpy() 
#     np.random.shuffle(grad_norms)
#     return grad_norms

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



if __name__ == '__main__':
    """
    
    This code basically does the same thing as the Local-Lipschitz-Constants
    paper but does not do branch-and-bounding as far as i'm aware. 
    
    """
    torch.manual_seed(6)

    # Create a small model and load pre-trained parameters.
    model_ori = build_model(in_dim=8)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x0 = torch.randn(1, 3, 8, 8, device=device)
    # eps = [0, 1./255, 4./255]
    eps = [0./255]

    lirpa_results = compute_lipschitz(model_ori, x0, eps, device=device)
    print(lirpa_results)

    grad_norms = get_norms_batched(model_ori, x0, eps[0], 1000, device)
    print(grad_norms)

    print('Max observed = ', max(grad_norms))

    # run_lippot(model_ori, x0, eps[0], 50000, device)


    
