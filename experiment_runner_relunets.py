import torch
import os
import time
import signal
import pandas as pd
import numpy as np

# --- ASSUMED EXTERNAL FUNCTIONS ---
from relu_blueprint import run_lipmip_analysis, run_lirpa_analysis, run_lipschitz_sampling

# --- TIMEOUT UTILITIES ---
class TimeoutError(Exception):
    """Custom exception for timeout events."""
    pass

class Timeout:
    """
    Context manager to handle timeouts.
    Raises TimeoutError if the code block runs longer than `seconds`.
    """
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        if self.seconds > 0:
            signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        if self.seconds > 0:
            signal.alarm(0)

# --- GLOBAL CONFIGURATION (Directories only) ---
DIR_MAP = {
    'MNIST':        './MNIST_Models',
    'CIFAR':        './CIFAR_Models',
    'TinyImageNet': './TinyImageNet_Models'
}

STAT_FILES = {
    'MNIST':        'mnist_stat.pt',
    'CIFAR':        'cifar100_stat.pt',
    'TinyImageNet': 'tinyimagenet_stat.pt'
}

GPU_ID = 0
DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

def run_experiment_suite(test_image_map, 
                         sampling_args, 
                         output_csv='experiment_results_2.csv', 
                         max_images=None,
                         timeout_seconds=1200,
                         run_lipmip=False,
                         run_lirpa=False,
                         run_sampling=True,
                         lirpa_args=None):
    """
    Loops over Datasets > Model Sizes > Test Images.
    Runs selected methods (LipMIP, LiRPA, Sampling).
    """
    
    # Default LiRPA args if none provided
    if lirpa_args is None:
        lirpa_args = {'METHOD': 'CROWN-IBP'}

    # Initialize DataFrame
    columns = [
        'Dataset', 'Model_Size', 'Image_Name', 
        'LipMIP_Result', 'LipMIP_Time',
        'LiRPA_Result', 'LiRPA_Time',
        'Sampling_Time', 'Sampling_File_Path'
    ]
    
    if os.path.exists(output_csv):
        print(f"Resuming analysis, appending to {output_csv}...")
    else:
        df = pd.DataFrame(columns=columns)
        df.to_csv(output_csv, index=False)

    datasets = ['MNIST', 'CIFAR', 'TinyImageNet']
    sizes = ['small', 'medium', 'large']
    
    print(f"Starting Experiment Suite on device: {DEVICE}")
    print(f"Timeout configured to: {timeout_seconds} seconds per method.")
    print(f"Methods active - LipMIP: {run_lipmip}, LiRPA: {run_lirpa}, Sampling: {run_sampling}")
    print(f"LiRPA Args: {lirpa_args}")

    # --- 1. Loop Datasets ---
    for dataset_name in datasets:
        if dataset_name not in test_image_map:
            print(f"Skipping {dataset_name} (No images provided in map).")
            continue
            
        current_images = test_image_map[dataset_name]
        folder_name = DIR_MAP[dataset_name]
        
        # --- 2. Loop Model Sizes ---
        for model_size in sizes:
            
            # --- PREPARE MODEL CONFIG ---
            model_name = f"{dataset_name}_{model_size}"
            weights_filename = f"{model_name}.pth"
            arch_filename = f"{model_name}_architecture.pt"
            stat_filename = STAT_FILES[dataset_name]
            
            if not os.path.exists(os.path.join(folder_name, weights_filename)):
                print(f"[!] Warning: Model {model_name} not found. Skipping.")
                continue

            try:
                arch_path = os.path.join(folder_name, arch_filename)
                stat_path = os.path.join(folder_name, stat_filename)
                
                loaded_arch = torch.load(arch_path)
                network_dimensions = loaded_arch['architecture']
                weight_normalization = torch.load(stat_path)
            except Exception as e:
                print(f"[!] Error loading config for {model_name}: {e}")
                continue

            # --- 3. Loop Test Images ---
            print(f"--- Processing {dataset_name} {model_size} (Limit: {max_images if max_images else 'All'}) ---")
            
            image_count = 0 

            for img_key, img_filename in current_images.items():
                
                if max_images is not None and image_count >= max_images:
                    print(f"    [Limit Reached] Stopping after {image_count} images for this model.")
                    break

                print(f"\n>>> Analyzing: {dataset_name} | {model_size} | {img_key}")
                
                if not os.path.exists(os.path.join(folder_name, img_filename)):
                    print(f"    [!] Image file {img_filename} not found in {folder_name}")
                    continue

                # Setup Config Args
                epsilon = 8/255
                config_args = {
                    "FOLDER_NAME": folder_name,
                    "MODEL_NAME": model_name,
                    "WEIGHTS_FILENAME": weights_filename,
                    "WEIGHT_NORMALISATION": weight_normalization,
                    "IMAGE_FILENAME": img_filename,
                    "NETWORK_DIMENSIONS": network_dimensions,
                    "EPSILON": epsilon,
                    "GPU_ID": GPU_ID,
                    "DEVICE": DEVICE,
                    "TIMEOUT_SECONDS": timeout_seconds 
                }

                # Dictionary to hold results for this row
                # Initialize with "SKIPPED". The saver logic will ensure these do not
                # overwrite existing valid data in the CSV.
                row_data = {
                    'Dataset': dataset_name,
                    'Model_Size': model_size,
                    'Image_Name': img_key,
                    'LipMIP_Result': "SKIPPED", 'LipMIP_Time': "SKIPPED",
                    'LiRPA_Result': "SKIPPED", 'LiRPA_Time': "SKIPPED",
                    'Sampling_Time': "SKIPPED", 'Sampling_File_Path': "SKIPPED"
                }

                # --- A. RUN LIPMIP ---
                if run_lipmip:
                    try:
                        st = time.time()
                        with Timeout(timeout_seconds):
                            mip_res = run_lipmip_analysis(config_args, timeout=timeout_seconds)
                        
                        row_data['LipMIP_Time'] = time.time() - st
                        row_data['LipMIP_Result'] = mip_res
                        print(f"    LipMIP: {mip_res} ({row_data['LipMIP_Time']:.2f}s)")

                    except TimeoutError:
                        print(f"    [!] LipMIP Timed Out ({timeout_seconds}s)")
                        row_data['LipMIP_Result'] = "TIMEOUT"
                        row_data['LipMIP_Time'] = timeout_seconds

                    except Exception as e:
                        print(f"    [!] LipMIP Failed: {e}")
                        row_data['LipMIP_Result'] = "ERROR"
                        row_data['LipMIP_Time'] = 0

                    _append_and_save(row_data, output_csv, columns)

                # --- B. RUN LiRPA ---
                if run_lirpa:
                    try:
                        st = time.time()
                        with Timeout(timeout_seconds):
                            # Pass the variable lirpa_args
                            lirpa_res = run_lirpa_analysis(lirpa_args, config_args)

                        row_data['LiRPA_Time'] = time.time() - st
                        row_data['LiRPA_Result'] = lirpa_res
                        print(f"    LiRPA:  {lirpa_res} ({row_data['LiRPA_Time']:.2f}s)")
                    
                    except TimeoutError:
                        print(f"    [!] LiRPA Timed Out ({timeout_seconds}s)")
                        row_data['LiRPA_Result'] = "TIMEOUT"
                        row_data['LiRPA_Time'] = timeout_seconds

                    except Exception as e:
                        print(f"    [!] LiRPA Failed: {e}")
                        row_data['LiRPA_Result'] = "ERROR"
                        row_data['LiRPA_Time'] = 0
                    
                    _append_and_save(row_data, output_csv, columns)

                # --- C. RUN SAMPLING ---
                if run_sampling:
                    try:
                        sampling_args = {
                            "METHOD": sampling_args['METHOD'],
                            "STEPS": sampling_args['STEPS'],
                            "WALKERS": sampling_args['WALKERS'],
                            "TEMP": sampling_args['TEMP'],
                            "BATCH_SIZE": sampling_args['BATCH_SIZE'],
                            "STEP_SIZE": 0.05 * epsilon 
                        }

                        st = time.time()
                        with Timeout(timeout_seconds):
                            acc_inputs, acc_norms = run_lipschitz_sampling(config_args, sampling_args)
                        
                        row_data['Sampling_Time'] = time.time() - st
                        
                        sample_save_name = f"samples_{dataset_name}_{model_size}_{img_key}.pt"
                        sample_save_path = os.path.join(folder_name, sample_save_name)
                        torch.save((acc_inputs, acc_norms), sample_save_path)
                        
                        row_data['Sampling_File_Path'] = sample_save_path
                        print(f"    Sampling: Saved to {sample_save_name} ({row_data['Sampling_Time']:.2f}s)")

                    except TimeoutError:
                        print(f"    [!] Sampling Timed Out ({timeout_seconds}s)")
                        row_data['Sampling_File_Path'] = "TIMEOUT"
                        row_data['Sampling_Time'] = timeout_seconds

                    except Exception as e:
                        print(f"    [!] Sampling Failed: {e}")
                        row_data['Sampling_File_Path'] = "ERROR"
                        row_data['Sampling_Time'] = 0

                    _append_and_save(row_data, output_csv, columns)
                
                image_count += 1

    print("\n=== Experiment Suite Completed ===")

def _append_and_save(row_data, filename, columns):
    """
    Helper to safely update the CSV.
    Logic:
    1. If row does NOT exist: Append it.
    2. If row DOES exist: Update ONLY the columns that are NOT 'SKIPPED' or None in the new data.
       This prevents overwriting previous valid results with 'SKIPPED' flags.
    """
    
    # Values that should be considered "no data" and thus should NOT overwrite existing entries
    SKIP_VALUES = ["SKIPPED", None]

    if not os.path.exists(filename):
        # Create new
        pd.DataFrame([row_data], columns=columns).to_csv(filename, index=False)
        return

    # Load existing
    df = pd.read_csv(filename)
    
    # Identify index of existing row
    mask = (
        (df['Dataset'] == row_data['Dataset']) & 
        (df['Model_Size'] == row_data['Model_Size']) & 
        (df['Image_Name'] == row_data['Image_Name'])
    )
    
    if mask.any():
        idx = df[mask].index[0]
        # Update existing row safely
        for col in columns:
            new_val = row_data.get(col)
            # Only update if the new value is actual data (not skipped/none)
            # We treat 'TIMEOUT' and 'ERROR' as actual data that SHOULD overwrite success
            # if the user re-ran it.
            if new_val not in SKIP_VALUES:
                df.at[idx, col] = new_val
    else:
        # Append new row
        new_row_df = pd.DataFrame([row_data], columns=columns)
        df = pd.concat([df, new_row_df], ignore_index=True)

    # Save
    df[columns].to_csv(filename, index=False)

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    
    # 1. Define Sampling Params

    # if Langevin
    sampling_args = {
        "METHOD": 'Langevin', 
        "STEPS": 2048,
        "WALKERS": 128,
        "TEMP": 1e-4,
        "BATCH_SIZE": 64, 
        "NORM" : 'linf'
    }
    # # if random
    # sampling_args = {
    #     "METHOD": 'Random', 
    #     "NUM_SAMPLES": 80000,
    #     "BATCH_SIZE": 64
    # }

    # 2. Define LiRPA Args
    # lirpa_args = {'METHOD': 'CROWN-IBP'}
    lirpa_args = {'METHOD': 'alpha-CROWN'}

    if os.path.exists('test_suite_dictionary.pt'):
        test_suite_images = torch.load('test_suite_dictionary.pt')
        
        run_experiment_suite(
            test_image_map=test_suite_images,
            sampling_args=sampling_args,
            max_images=1,
            timeout_seconds=1200, 
            run_lipmip=False,      # Skipped
            run_lirpa=False,        # Run (will update LiRPA col, leave LipMIP alone)
            run_sampling=True,     # Run (will update Sampling col)
            lirpa_args=lirpa_args
        )
    else:
        print("Please provide the test_suite_dictionary.pt file.")