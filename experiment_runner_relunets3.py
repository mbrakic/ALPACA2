import torch
import os
import time
import signal
import pandas as pd
import numpy as np

# --- ASSUMED EXTERNAL FUNCTIONS ---
# Ensure these are available in your environment or comment them out for syntax checking
try:
    from relu_blueprint import run_lipmip_analysis, run_lirpa_analysis, run_lipschitz_sampling
except ImportError:
    # Placeholder for standalone testing if files are missing
    pass

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
                         epsilon, 
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
    
    Args:
        sampling_args (dict): Dictionary keyed by dataset name containing sampling params.
                              Example: {'MNIST': {...}, 'CIFAR': {...}}
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
        
        # --- RETRIEVE DATASET SPECIFIC SAMPLING ARGS ---
        # We look up the args for 'MNIST', 'CIFAR', etc.
        dataset_specific_sampling_params = sampling_args.get(dataset_name)
        
        if run_sampling and dataset_specific_sampling_params is None:
            print(f"[Warning] No sampling_args provided for {dataset_name}. Sampling will be skipped for this dataset.")

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
                # Check if sampling is on AND we found args for this dataset
                if run_sampling and dataset_specific_sampling_params is not None:
                    try:
                        # Build the runtime sampling args using the dataset-specific params
                        runtime_sampling_args = {
                            "METHOD": dataset_specific_sampling_params['METHOD'],
                            "STEPS": dataset_specific_sampling_params['STEPS'],
                            "WALKERS": dataset_specific_sampling_params['WALKERS'],
                            "TEMP": dataset_specific_sampling_params['TEMP'],
                            "BATCH_SIZE": dataset_specific_sampling_params['BATCH_SIZE'],
                            "STEP_SIZE": 0.05 * epsilon 
                        }
                        
                        # Add optional NORM param if it exists
                        if 'NORM' in dataset_specific_sampling_params:
                            runtime_sampling_args['NORM'] = dataset_specific_sampling_params['NORM']

                        st = time.time()
                        with Timeout(timeout_seconds):
                            acc_inputs, acc_norms = run_lipschitz_sampling(config_args, runtime_sampling_args)
                        
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
    """
    SKIP_VALUES = ["SKIPPED", None]

    if not os.path.exists(filename):
        pd.DataFrame([row_data], columns=columns).to_csv(filename, index=False)
        return

    df = pd.read_csv(filename)
    
    mask = (
        (df['Dataset'] == row_data['Dataset']) & 
        (df['Model_Size'] == row_data['Model_Size']) & 
        (df['Image_Name'] == row_data['Image_Name'])
    )
    
    if mask.any():
        idx = df[mask].index[0]
        for col in columns:
            new_val = row_data.get(col)
            if new_val not in SKIP_VALUES:
                df.at[idx, col] = new_val
    else:
        new_row_df = pd.DataFrame([row_data], columns=columns)
        df = pd.concat([df, new_row_df], ignore_index=True)

    df[columns].to_csv(filename, index=False)

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    
    # 0. Set epsilon 
    EPSILON = 8/255
    # 1. Define Sampling Params per Dataset
    
    # Example: MNIST might need fewer walkers or specific temp
    mnist_sampling = {
        "METHOD": 'Langevin', 
        "STEPS": 256,          # Less steps for MNIST
        "WALKERS": 128,
        "TEMP": 1e-1,
        "STEP_SIZE": 0.05 * EPSILON, 
        "SOBOL_SAMPLES": 4000, 
        "BATCH_SIZE": 64, 
        "NORM" : 'linf'
    }

    # Example: CIFAR/TinyImageNet might need more robust sampling
    cifar_sampling = {
        "METHOD": 'Langevin', 
        "STEPS": 4096,          # More steps for complex data
        "WALKERS": 128,
        "TEMP": 1e-2,
        "STEP_SIZE": 0.05 * EPSILON, 
        "SOBOL_SAMPLES": 4000, 
        "BATCH_SIZE": 64, 
        "NORM" : 'linf'
    }

    tinyimagenet_sampling = {
        "METHOD": 'Langevin', 
        "STEPS": 2*4096,          # More steps for complex data
        "WALKERS": 128,
        "TEMP": 1e-3,
        "STEP_SIZE": 0.05 * EPSILON, 
        "SOBOL_SAMPLES": 4000, 
        "BATCH_SIZE": 64, 
        "NORM" : 'linf'
    }

    # Wrap them in a master dictionary keyed by dataset name
    sampling_args_map = {
        'MNIST': mnist_sampling,
        'CIFAR': cifar_sampling,
        'TinyImageNet': tinyimagenet_sampling
    }

    # 2. Define LiRPA Args
    lirpa_args = {'METHOD': 'CROWN-IBP'}

    if os.path.exists('test_suite_dictionary.pt'):
        test_suite_images = torch.load('test_suite_dictionary.pt')
        
        run_experiment_suite(
            test_image_map=test_suite_images,
            epsilon = EPSILON,
            sampling_args=sampling_args_map, # Passing the map now
            max_images=1,
            timeout_seconds=1200, 
            run_lipmip=False,      
            run_lirpa=False,        
            run_sampling=True,     
            lirpa_args=lirpa_args
        )
    else:
        print("Please provide the test_suite_dictionary.pt file.")