import math
import time
import torch
from auto_LiRPA import BoundedTensor
from auto_LiRPA.utils import get_spec_matrix
from auto_LiRPA.perturbations import PerturbationLpNorm
from bab import bab_gradnorm

def compute_margin_jacobian_bound(model, x, labels, args):
    """
    Calculates the Lipschitz constant specifically for the margin 
    between output label 0 and output label 1: Lip(f_0(x) - f_1(x)).
    """

    time_begin = time.time()

    assert x.size(0) == 1  # batch size must be 1
    c = torch.ones(1, 1, 1).to(x)  # For backward graph
    c_forward = get_spec_matrix(x, labels, args.num_classes)  # For forward graph

    # --- DEFINE THE GRADIENT DIRECTION FOR MARGIN (0 vs 1) ---
    # We want the gradient of the scalar function: g(x) = f_0(x) - f_1(x)
    # The weight vector w for this combination is: [1, -1, 0, ..., 0]
    grad_start = torch.zeros(1, 1, args.num_classes).to(x)
    grad_start[0, 0, 0] = 1.0
    grad_start[0, 0, 1] = -1.0

    # --- STEP 1: INITIAL "CHEAP" BOUND ---
    # Run the model backward with the specific grad_start vector
    # model(x, grad_start, final_node_name=model.forward_final_name)
    # model(x, grad_start, final_node_name=model.final_name)
    # model(x, grad_start)

    # Compute loose bound without BaB (bab=False) first
    print("Computing initial bound for Margin 0-1...")
    ret = -bab_gradnorm(
        model, x, grad_start, 
        c=-c, c_forward=c_forward,
        opt_forward=True, 
        args=args, 
        bab=False
    )

    if args.norm == 2:
        ret = math.sqrt(ret)
    
    initial_bound = ret
    print(f"Initial cheap bound: {initial_bound}")

    # --- STEP 2: REFINEMENT WITH FULL TIMEOUT ---
    # Use whatever time is left to refine this specific margin
    time_remaining = args.timeout - (time.time() - time_begin)
    
    if time_remaining > 0:
        print(f"Refining bound with BaB (timeout={time_remaining:.2f}s)...")
        
        # Re-trigger model required before calling bab_gradnorm again
        model(x, grad_start, final_node_name=model.forward_final_name)
        model(x, grad_start)

        # We pass the full remaining time to this single call
        refined_ret = -bab_gradnorm(
            model, x, grad_start,
            c=-c, c_forward=c_forward, 
            args=args, 
            timeout=time_remaining,
            bab=True # Enable Branch and Bound
        )
        
        if args.norm == 2:
            refined_ret = math.sqrt(refined_ret)
        
        # Take the tighter of the two (usually refined is lower/better)
        # Note: Depending on implementation, bab_gradnorm usually returns a valid upper bound
        # so we just take the result.
        ret = refined_ret
    
    print(f'Final Lipschitz Constant for Margin 0-1: {ret}\n')
    return ret


def lirpa_local_lipschitz(model, data, labels, data_lb, data_ub, args=None):
    ptb = PerturbationLpNorm(norm=args.norm, x_L=data_lb, x_U=data_ub)
    x = data = BoundedTensor(data, ptb)
    
    # If args.bab is True, we use our new margin-specific function
    if args.bab:
        return compute_margin_jacobian_bound(model, x, labels, args)
    else:
        # Fallback or standard call if bab is not enabled
        # Note: If you want non-BaB margin bounds, you'd need to adapt this too,
        # but for now we route BaB requests to the new function.
        return model.compute_jacobian_bounds(x, labels=labels)