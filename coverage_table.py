import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import time
import signal
import copy
import pandas as pd
import numpy as np
import gc
import sys

sys.path.append(os.getcwd())

from ALPACA.ALPACA import ALPACA
from InformedSampling import LangevinSampler
from lipMIP.lipMIP import LipMIP
from lipMIP.hyperbox import Hyperbox
from lipMIP.relu_nets import ReLUNet

# --- CONSTANTS ---
FOLDER_NAME    = './MNIST_Models'
MODEL_NAME     = 'MNIST_small'
WEIGHTS_FILE   = f'{MODEL_NAME}.pth'
ARCH_FILE      = f'{MODEL_NAME}_architecture.pt'
DATA_ROOT      = './data'
OUTPUT_CSV     = 'coverage_test.csv'
TARGET_COUNT   = 1000
EPSILON        = 1 / 255
GAMMA          = 0.01
LIPMIP_TIMEOUT = 150
GPU_ID         = 0
DEVICE         = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
WEIGHT_NORM    = ((0.1307,), (0.3081,))

SAMPLING_ARGS = {
    "STEPS":         128,
    "WALKERS":       256,
    "TEMP":          1e-1,
    "SOBOL_SAMPLES": 8000,
    "BATCH_SIZE":    64,
}

COLUMNS = ['ALPACA_Attempts', 'ALPACA_Endpoint', 'ALPACA_Time', 'LipMIP_Result', 'LipMIP_Time']


# --- TIMEOUT UTILITIES ---
class TimeoutError(Exception):
    pass

class Timeout:
    def __init__(self, seconds):
        self.seconds = seconds

    def handle_timeout(self, signum, frame):
        raise TimeoutError(f"Timed out after {self.seconds}s")

    def __enter__(self):
        if self.seconds > 0:
            signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.seconds > 0:
            signal.alarm(0)


# --- HELPERS ---
def setup_domain(x0):
    """Build a linf Hyperbox around x0 clipped to valid normalised pixel range."""
    std_val  = WEIGHT_NORM[1][0]
    mean_val = WEIGHT_NORM[0][0]
    eps_norm  = EPSILON / std_val
    min_valid = (0.0 - mean_val) / std_val
    max_valid = (1.0 - mean_val) / std_val

    box_low = torch.clamp(x0 - eps_norm, min_valid, max_valid)
    box_hi  = torch.clamp(x0 + eps_norm, min_valid, max_valid)

    domain = Hyperbox.build_unit_hypercube(x0.numel())
    domain.box_low = box_low
    domain.box_hi  = box_hi
    return domain


def run_sampling(network, x0, c_vector, domain):
    std_val  = WEIGHT_NORM[1][0]
    eps_norm = EPSILON / std_val
    step_size = 0.1 * eps_norm

    sampler = LangevinSampler(
        model=network,
        c_vector=c_vector,
        domain=domain,
        device=DEVICE,
        norm_type='linf',
        steps=SAMPLING_ARGS['STEPS'],
        walkers=SAMPLING_ARGS['WALKERS'],
        step_size=step_size,
        temperature=SAMPLING_ARGS['TEMP'],
        nms_radius=0.1 * EPSILON,
        top_k_refine=20,
        k_box_epsilon=0.1 * EPSILON,
        sobol_samples_per_k=SAMPLING_ARGS['SOBOL_SAMPLES'],
        batch_size=SAMPLING_ARGS['BATCH_SIZE'],
    )
    return sampler.run()


def run_lipmip(network, x0, c_vector):
    """Run LipMIP on CPU using a deep copy of the network."""
    cpu = torch.device("cpu")
    cpu_net = copy.deepcopy(network).to(cpu)
    x0_cpu  = x0.cpu()
    cv_cpu  = c_vector.cpu().flatten()

    # Build CPU domain
    std_val   = WEIGHT_NORM[1][0]
    mean_val  = WEIGHT_NORM[0][0]
    eps_norm  = EPSILON / std_val
    min_valid = (0.0 - mean_val) / std_val
    max_valid = (1.0 - mean_val) / std_val
    box_low   = torch.clamp(x0_cpu - eps_norm, min_valid, max_valid).flatten()
    box_hi    = torch.clamp(x0_cpu + eps_norm, min_valid, max_valid).flatten()

    domain = Hyperbox.build_unit_hypercube(x0_cpu.numel())
    domain.box_low = box_low
    domain.box_hi  = box_hi

    problem = LipMIP(
        cpu_net,
        domain,
        cv_cpu,
        primal_norm='linf',
        verbose=True,
        timeout=LIPMIP_TIMEOUT,
        num_threads=24,
    )
    result = problem.compute_max_lipschitz()
    return result.value


def append_row(row):
    pd.DataFrame([row], columns=COLUMNS).to_csv(
        OUTPUT_CSV, mode='a', header=False, index=False
    )


# --- MAIN ---
if __name__ == "__main__":

    # --- Initialise / resume CSV ---
    count = 0
    if os.path.exists(OUTPUT_CSV):
        try:
            existing = pd.read_csv(OUTPUT_CSV)
            count = len(existing)
        except pd.errors.EmptyDataError:
            count = 0

        if count > 0:
            print(f"Resuming — {count} data points already in {OUTPUT_CSV}.")
        else:
            pd.DataFrame(columns=COLUMNS).to_csv(OUTPUT_CSV, index=False)
            print(f"Found empty {OUTPUT_CSV}, reinitialising.")
    else:
        pd.DataFrame(columns=COLUMNS).to_csv(OUTPUT_CSV, index=False)
        print(f"Created new {OUTPUT_CSV}.")

    # Load architecture and weights
    arch_path = os.path.join(FOLDER_NAME, ARCH_FILE)
    loaded_arch       = torch.load(arch_path)
    network_dims      = loaded_arch['architecture']
    output_dim        = network_dims[-1]

    network = ReLUNet(network_dims, bias=True)
    state_dict = torch.load(os.path.join(FOLDER_NAME, WEIGHTS_FILE), map_location=DEVICE)
    network.load_state_dict(state_dict)
    network = network.to(DEVICE)
    network.eval()

    # MNIST test set (normalised + flattened)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*WEIGHT_NORM),
        transforms.Lambda(lambda x: x.flatten()),
    ])
    mnist_test = torchvision.datasets.MNIST(
        root=DATA_ROOT, train=False, download=True, transform=transform
    )

    dataset_idx = 0
    attempts = 0  # ALPACA runs since last successful save

    while count < TARGET_COUNT:
        img_idx = dataset_idx % len(mnist_test)
        dataset_idx += 1

        x0_flat, _ = mnist_test[img_idx]
        x0 = x0_flat.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits  = network(x0)
            pred_id = logits.argmax().item()

        c_vector = torch.zeros((1, output_dim), device=DEVICE)
        c_vector[0, pred_id] = 1.0

        domain = setup_domain(x0)

        print(f"\n[{count}/{TARGET_COUNT}] Dataset idx {img_idx}, pred={pred_id}")

        # --- Sampling ---
        try:
            acc_inputs, acc_norms = run_sampling(network, x0, c_vector, domain)
        except Exception as e:
            print(f"  Sampling failed: {e}")
            gc.collect()
            torch.cuda.empty_cache()
            continue

        # --- ALPACA ---
        alpaca_start = time.time()
        try:
            alpaca_results = ALPACA.run_full_analysis(
                data=acc_norms,
                coords=acc_inputs,
                gamma=GAMMA,
                show_plot=False,
            )
        except Exception as e:
            print(f"  ALPACA failed: {e}")
            del acc_inputs, acc_norms
            gc.collect()
            torch.cuda.empty_cache()
            continue
        alpaca_time = time.time() - alpaca_start

        del acc_inputs, acc_norms
        gc.collect()
        torch.cuda.empty_cache()

        attempts += 1

        if not alpaca_results.optimization_succeeded:
            print(f"  ALPACA did not succeed — skipping. (attempts this batch: {attempts})")
            continue

        alpaca_endpoint = alpaca_results.max_endpoint_from_optimization
        print(f"  ALPACA: endpoint={alpaca_endpoint:.4f}  ({alpaca_time:.2f}s)")

        # --- LipMIP ---
        lipmip_result = None
        lipmip_time   = 0.0
        try:
            st = time.time()
            with Timeout(LIPMIP_TIMEOUT):
                lipmip_result = run_lipmip(network, x0, c_vector)
            lipmip_time = time.time() - st
            print(f"  LipMIP: {lipmip_result}  ({lipmip_time:.2f}s)")

        except TimeoutError:
            lipmip_result = "TIMEOUT"
            lipmip_time   = LIPMIP_TIMEOUT
            print(f"  LipMIP timed out ({LIPMIP_TIMEOUT}s).")

        except Exception as e:
            lipmip_result = "ERROR"
            lipmip_time   = 0.0
            print(f"  LipMIP failed: {e}")

        # --- Save ---
        append_row({
            'ALPACA_Attempts': attempts,
            'ALPACA_Endpoint': alpaca_endpoint,
            'ALPACA_Time':     alpaca_time,
            'LipMIP_Result':   lipmip_result,
            'LipMIP_Time':     lipmip_time,
        })
        count += 1
        attempts = 0  # reset for next batch
        print(f"  [Saved] Successful runs: {count}/{TARGET_COUNT}")

    print(f"\n=== Done — {TARGET_COUNT} data points collected. Results in {OUTPUT_CSV} ===")
