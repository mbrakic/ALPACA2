import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from scipy.spatial import cKDTree
from scipy.stats import qmc
from typing import Tuple, Optional, Literal
from tqdm import tqdm
import pandas as pd
import time

# Import your specific custom libraries
from lipMIP.hyperbox import Hyperbox 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- [NEW] SGLD Attacker Class (Replaces AdamAttacker) ---
class SGLDAttacker:
    """
    Stochastic Gradient Langevin Dynamics Sampler.
    Unlike Adam (which collapses to a point), this samples from the tail distribution.
    Update rule: x_{t+1} = x_t + (lr/2) * grad + N(0, lr)
    """
    def __init__(self, network, c_vector, domain, norm_type, num_restarts, step_size, temperature=0.01):
        self.network = network.to(device).eval()
        self.c_vector = c_vector.to(device)
        self.domain = domain
        self.norm_type = norm_type
        self.num_restarts = num_restarts
        self.step_size = step_size
        self.temperature = temperature # Higher T = wider exploration of the tail

        # Handle tensor shapes
        if domain.box_low.dim() == 4 and domain.box_low.shape[0] == 1:
            self.box_low = domain.box_low.squeeze(0).to(device)
            self.box_hi = domain.box_hi.squeeze(0).to(device)
        else:
            self.box_low = domain.box_low.to(device)
            self.box_hi = domain.box_hi.to(device)
            
        self.input_shape = self.box_low.shape 
        
        if self.norm_type == 'l1': self.ord = float('inf')
        elif self.norm_type == 'l2': self.ord = 2
        elif self.norm_type == 'linf': self.ord = 1
        else: raise ValueError("norm_type must be 'l1', 'l2', or 'linf'")

        self.points = self._initialize_points()

    def _initialize_points(self):
        """Random uniform initialization within domain"""
        rand_floats = torch.rand(self.num_restarts, *self.input_shape, device=device)
        domain_width = self.box_hi - self.box_low
        points = rand_floats * domain_width + self.box_low
        return points

    def _project(self, points):
        """Clamp to domain constraints"""
        return torch.max(torch.min(points, self.box_hi), self.box_low)

    def run_attack(self, num_steps):
        print(f"Starting SGLD Sampling with {self.num_restarts} chains (T={self.temperature})...")
        
        self.points.requires_grad = True
        all_collected_grad_norms = []
        all_collected_inputs = [] # Optional: Sample thinning to save memory

        optimizer = torch.optim.SGD([self.points], lr=self.step_size) # SGD is base for SGLD

        for i in tqdm(range(num_steps), desc="SGLD Sampling"):
            # 1. Forward Pass
            outputs = self.network(self.points)
            scalar_output = (outputs * self.c_vector).sum(dim=1)
            # We want to MAXIMIZE output, so minimize negative
            loss = -scalar_output.sum()

            optimizer.zero_grad()
            loss.backward()

            # 2. SGLD Update Step
            # Delta = (step_size / 2) * grad + sqrt(step_size) * noise
            with torch.no_grad():
                # Gradient term (force pushing up the hill)
                grad = self.points.grad
                
                # Noise term (Langevin dynamics exploration)
                noise = torch.randn_like(self.points) * np.sqrt(self.step_size * self.temperature)
                
                # Update
                # Note: We add grad because we want to maximize. 
                # Standard SGLD is x += (eps/2)*grad + sqrt(eps)*noise
                self.points.data.add_(0.5 * self.step_size * grad) 
                self.points.data.add_(noise)
                
                # Projection
                self.points.data = self._project(self.points.data)

            # 3. Collect Statistics (Calculate Norm of Gradient)
            if self.points.grad is not None:
                current_grads = self.points.grad.detach().view(self.num_restarts, -1)
                grad_norms = torch.linalg.norm(current_grads, ord=self.ord, dim=1)
                all_collected_grad_norms.append(grad_norms.cpu())
                
                # [OPTIONAL] Save inputs every X steps to reduce correlation
                if i % 10 == 0: 
                    all_collected_inputs.append(self.points.detach().cpu())
            else:
                 # Fallback
                all_collected_grad_norms.append(torch.zeros(self.num_restarts))

            # Detach for next iter
            self.points.grad.zero_()

        # Compile results
        # We flatten the time dimension: every step of SGLD is a valid sample from the distribution
        final_norms = torch.stack(all_collected_grad_norms).transpose(0, 1).numpy() # (Restarts, Steps)
        
        # If we collected inputs:
        if len(all_collected_inputs) > 0:
            final_inputs = torch.stack(all_collected_inputs).transpose(0, 1).numpy() # (Restarts, Steps/10, C, H, W)
            # Expand norms to match input thinning if necessary, or just return last
            # For this implementation, we will just return the final state of each chain 
            # to match your original data structure expected by the Sampler
            return self.points.detach().cpu().numpy(), final_norms[:,-1] # Return final points
        
        return self.points.detach().cpu().numpy(), final_norms[:,-1]


# --- Helper Functions (Preserved & Optimized) ---

def generate_sobol_in_region(d, n_points, lower_bounds, upper_bounds, shape, seed=None):
    """
    Generates Quasi-Monte Carlo samples.
    """
    scramble = (seed is not None)
    lb_flat = lower_bounds.flatten()
    ub_flat = upper_bounds.flatten()
    
    # Use Latin Hypercube for High-Dim (ImageNet is ~150k dims)
    if d > 20000:
        sampler = qmc.LatinHypercube(d=d, seed=seed)
    else:
        sampler = qmc.Sobol(d=d, seed=seed, scramble=scramble)
    
    unit_samples = sampler.random(n=n_points) 
    scaled_samples_flat = qmc.scale(unit_samples, lb_flat, ub_flat)
    return scaled_samples_flat.reshape(n_points, *shape)

def get_input_norms(network, c_vector, data_loader, norm_type='linf', desc="Sobol Evaluation"):
    network.eval()
    grad_norms_list = []
    inputs_list = [] 

    if norm_type == 'l1': ord = float('inf')
    elif norm_type == 'l2': ord = 2
    elif norm_type == 'linf': ord = 1
    
    for inputs, _ in data_loader:
        inputs_list.append(inputs) 
        inputs = inputs.to(device)
        inputs.requires_grad_(True)

        outputs = network(inputs)
        scalar_output = (outputs * c_vector).sum(dim=1)
        scalar_output.sum().backward()
        
        gradients = inputs.grad
        if gradients is None:
            batch_norms = torch.zeros(inputs.shape[0], device='cpu')
        else:
            gradients_flat = gradients.reshape(inputs.shape[0], -1)
            batch_norms = torch.linalg.norm(gradients_flat, ord=ord, dim=1)
        
        grad_norms_list.append(batch_norms.cpu().detach())
        network.zero_grad()

    all_inputs = torch.cat(inputs_list).numpy()
    grad_norms = torch.cat(grad_norms_list).numpy()
    
    return all_inputs, grad_norms


# --- [MODIFIED] LangevinSampler Class ---

class LangevinSampler:
    """
    Replaces AdamSobolSampler. Uses SGLD to probe the tail, then applies
    spatial declustering and Sobol refinement.
    """
    def __init__(self, model, c_vector, domain, device, 
                 norm_type='linf', steps=500, walkers=64, step_size=0.01, 
                 temperature=0.01, # New param for tail width
                 nms_radius=0.1, top_k_refine=20, min_k_diff=1e-4, k_box_epsilon=1e-3, 
                 sobol_samples_per_k=800, batch_size=32):
        
        self.model = model
        self.c_vector = c_vector
        self.domain = domain
        self.device = device
        
        # Shape handling
        if domain.box_low.dim() == 4 and domain.box_low.shape[0] == 1:
            self.input_shape = domain.box_low.shape[1:] 
        else:
            self.input_shape = domain.box_low.shape

        self.flat_dim = domain.box_low.numel()
        
        self.norm_type = norm_type
        self.steps = steps
        self.walkers = walkers 
        self.step_size = step_size
        self.temperature = temperature
        
        self.nms_radius = nms_radius
        self.top_k_refine = top_k_refine
        self.min_k_diff = min_k_diff
        self.k_box_epsilon = k_box_epsilon
        self.sobol_samples_per_k = sobol_samples_per_k
        self.batch_size = batch_size

    def run(self):
        """Full SGLD + Spatial Decorrelation + Sobol Refinement"""
        
        # 1. SGLD Sampling (The "Global" Search)
        print(f"\n--- STAGE 1: Running SGLD Global Search ---")
        sgld = SGLDAttacker(self.model, self.c_vector, self.domain, self.norm_type, 
                            self.walkers, self.step_size, self.temperature)
        
        # We get the final states of our walkers
        sgld_inputs, sgld_norms = sgld.run_attack(num_steps=self.steps)
        
        if sgld_inputs.size == 0:
            return np.array([]), np.array([])

        # 2. Spatial Decorrelation (NMS)
        # SGLD walkers might end up in the same basin. We filter them.
        print("\n--- STAGE 2: Spatial Decorrelation (NMS) ---")
        unique_inputs, unique_norms = self._spatial_decorrelation(
            sgld_inputs, sgld_norms, self.nms_radius
        )
        print(f"Decorrelated peaks found: {len(unique_norms)} (from {self.walkers} walkers)")

        # 3. Select Top K
        k_to_refine = min(self.top_k_refine, len(unique_norms))
        print(f"\n--- STAGE 3: Selecting Top {k_to_refine} diverse peaks ---")
        
        top_k_norms, top_k_inputs = self._get_diverse_top_k(
            unique_norms, unique_inputs, k_to_refine, self.min_k_diff
        )

        # 4. Sobol Refinement
        # Now we densely sample the immediate vicinity of the discovered peaks
        print(f"\n--- STAGE 4: Sobol Refinement (Local Volume Estimation) ---")
        final_inputs, final_norms = self._run_sobol_refinement(top_k_inputs, top_k_norms)

        return final_inputs, final_norms

    def _spatial_decorrelation(self, inputs: np.ndarray, values: np.ndarray, radius: float):
        """
        Filters points that are too close in Euclidean space.
        """
        if len(values) == 0: return np.array([]), np.array([])

        flat_inputs = inputs.reshape(inputs.shape[0], -1)
        
        # KDTree is efficient for finding neighbors
        kdtree = cKDTree(flat_inputs)
        sorted_indices = np.argsort(values)[::-1]
        
        suppressed = np.zeros(len(values), dtype=bool)
        kept_indices = []

        for i in sorted_indices:
            if suppressed[i]: continue
            kept_indices.append(i)
            # Find all points within radius of this peak and suppress them
            neighbors = kdtree.query_ball_point(flat_inputs[i], r=radius)
            suppressed[neighbors] = True
            
        kept_indices = np.array(kept_indices, dtype=int)
        return inputs[kept_indices], values[kept_indices]

    def _get_diverse_top_k(self, norms, inputs, k, min_diff):
        """Standard top-k selection"""
        sorted_indices = np.argsort(norms)[::-1]
        selected_indices = []
        last_val = None

        for idx in sorted_indices:
            if len(selected_indices) >= k: break
            curr_val = norms[idx]
            if last_val is None or abs(last_val - curr_val) >= min_diff:
                selected_indices.append(idx)
                last_val = curr_val
                
        final_indices = np.array(selected_indices)
        return norms[final_indices], inputs[final_indices]

    def _run_sobol_refinement(self, centers, center_norms):
        """Local sampling around peaks"""
        final_inputs_list = []
        final_norms_list = []
        
        orig_low = self.domain.box_low.cpu().numpy()
        orig_high = self.domain.box_hi.cpu().numpy()

        for i, (center, norm_val) in enumerate(zip(centers, center_norms)):
            # Define small box around the peak
            small_low = np.maximum(center - self.k_box_epsilon, orig_low)
            small_high = np.minimum(center + self.k_box_epsilon, orig_high)
            
            # Generate Sobol/LHS samples
            sobol_inputs = generate_sobol_in_region(
                d=self.flat_dim,
                n_points=self.sobol_samples_per_k,
                lower_bounds=small_low,
                upper_bounds=small_high,
                shape=self.input_shape,
                seed=int(np.random.rand() * 1000)
            )
            
            # Evaluate samples
            tensor_inputs = torch.from_numpy(sobol_inputs).float()
            dataset = TensorDataset(tensor_inputs, torch.zeros(len(tensor_inputs), dtype=torch.long))
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            
            box_inputs, box_norms = get_input_norms(
                self.model, self.c_vector, loader, 
                norm_type=self.norm_type
            )
            
            # Combine the center peak with its surrounding samples
            # This gives you the "Distribution" of the peak, not just the max
            final_inputs_list.append(np.concatenate([center[None, ...], box_inputs], axis=0))
            final_norms_list.append(np.concatenate([np.array([norm_val]), box_norms], axis=0))
            
        if not final_inputs_list:
            return np.array([]), np.array([])

        return np.concatenate(final_inputs_list, axis=0), np.concatenate(final_norms_list, axis=0)