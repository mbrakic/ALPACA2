import torch
import os
import time
import signal
import pandas as pd
import numpy as np
import gc 

# --- ASSUMED EXTERNAL FUNCTIONS ---
try:
    # Added run_eclipse to imports
    from cnn_blueprint import (
        run_lipmip_analysis, run_lirpa_analysis, run_lipschitz_sampling, run_eclipse_analysis
    )
except ImportError:
    pass

# --- TIMEOUT UTILITIES ---
class TimeoutError(Exception):
    """Custom exception for timeout events."""
    pass

class Timeout:
    """
    Context manager to handle timeouts.
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

# --- GLOBAL CONFIGURATION ---

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

# 3. Model Zoo
MODEL_ZOO = {
    'MNIST': {
        'CNN_4Layer':   'mnist_cnn_4layer',
        'CNN_4Layer_8': 'mnist_cnn_4layer_8',
        'MLP_3Layer':   'mnist_mlp_3layer'
    },
    'CIFAR': {
        'CNN_4Layer':   'cnn_4layer_stride1_padding0',
        # 'CNN_4Layer_Demo': 'cnn_4layer_stride1_padding0_demo', 
        'CNN_6Layer':   'cnn_6layer_stride1_padding0'
    },
    'TinyImageNet': {
        'CNN_4Layer':   'cnn_4layer_stride2_imagenet',
        'CNN_6Layer':   'cnn_6layer_stride2_imagenet'
    }
}

GPU_ID = 0
DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

def run_experiment_suite(test_image_map,
                          epsilon, 
                          sampling_args, 
                          output_csv='experiment_results_cnn.csv', 
                          max_images=None,
                          timeout_seconds=1200,
                          run_lipmip=False,
                          run_lirpa=False,
                          run_sampling=True,
                          run_eclipse=False,  # <--- NEW ARGUMENT
                          norm='linf',        # <--- NEW ARGUMENT (options: 'linf', 'l2')
                          lirpa_args=None):
    """
    Loops over Datasets > Specific Models > Test Images.
    Supports switching norms (Linf/L2) and running Eclipse analysis.
    """
    
    # Default LiRPA args if none provided
    if lirpa_args is None:
        lirpa_args = {'METHOD': 'CROWN-IBP'}

    # Initialize DataFrame
    # Added Eclipse columns and Norm column
    columns = [
        'Dataset', 'Model_Name', 'Image_Name', 'Norm',
        'LipMIP_Result', 'LipMIP_Time',
        'LiRPA_Result', 'LiRPA_Time',
        'Eclipse_Result', 'Eclipse_Time', # <--- NEW COLUMNS
        'Sampling_Time', 'Sampling_File_Path'
    ]
    
    if os.path.exists(output_csv):
        print(f"Resuming analysis, appending to {output_csv}...")
    else:
        df = pd.DataFrame(columns=columns)
        df.to_csv(output_csv, index=False)

    print(f"Starting Experiment Suite on device: {DEVICE}")
    print(f"Timeout configured to: {timeout_seconds} seconds per method.")
    print(f"Norm Mode: {norm.upper()}")
    print(f"Methods active - LipMIP: {run_lipmip}, LiRPA: {run_lirpa}, Eclipse: {run_eclipse}, Sampling: {run_sampling}")
    
    # --- 1. Loop Datasets ---
    for dataset_name, models_dict in MODEL_ZOO.items():
        
        if dataset_name not in test_image_map:
            print(f"Skipping {dataset_name} (No images provided in map).")
            continue
            
        current_images = test_image_map[dataset_name]
        folder_name = DIR_MAP[dataset_name]
        
        # --- RETRIEVE DATASET SPECIFIC SAMPLING ARGS ---
        dataset_specific_sampling_params = sampling_args.get(dataset_name)
        
        if run_sampling and dataset_specific_sampling_params is None:
            print(f"[Warning] No sampling_args provided for {dataset_name}. Sampling will be skipped.")

        # --- 2. Loop Models ---
        for model_display_name, model_basename in models_dict.items():
            
            # --- PREPARE MODEL CONFIG ---
            weights_filename = f"{model_basename}.pth"
            stat_filename = STAT_FILES[dataset_name]
            
            weights_path = os.path.join(folder_name, weights_filename)

            if not os.path.exists(weights_path):
                print(f"[!] Warning: Model file {weights_filename} not found in {folder_name}. Skipping.")
                continue

            try:
                stat_path = os.path.join(folder_name, stat_filename)
                weight_normalization = torch.load(stat_path)
            except Exception as e:
                print(f"[!] Error loading config for {model_basename}: {e}")
                continue

            # --- 3. Loop Test Images ---
            print(f"--- Processing {dataset_name} | {model_display_name} (Limit: {max_images if max_images else 'All'}) ---")
            
            image_count = 0 

            for img_key, img_filename in current_images.items():
                
                if max_images is not None and image_count >= max_images:
                    print(f"    [Limit Reached] Stopping after {image_count} images for this model.")
                    break

                print(f"\n>>> Analyzing: {dataset_name} | {model_display_name} | {img_key} | Norm: {norm}")
                
                if not os.path.exists(os.path.join(folder_name, img_filename)):
                    print(f"    [!] Image file {img_filename} not found in {folder_name}")
                    continue

                # Setup Config Args
                config_args = {
                    "FOLDER_NAME": folder_name,
                    "MODEL_NAME": model_display_name,
                    "WEIGHTS_FILENAME": weights_filename,
                    "WEIGHT_NORMALISATION": weight_normalization,
                    "IMAGE_FILENAME": img_filename,
                    "EPSILON": epsilon,
                    "GPU_ID": GPU_ID,
                    "DEVICE": DEVICE,
                    "TIMEOUT_SECONDS": timeout_seconds 
                }

                # Dictionary to hold results for this row
                row_data = {
                    'Dataset': dataset_name,
                    'Model_Name': model_display_name,
                    'Image_Name': img_key,
                    'Norm': norm,  # Track the norm used
                    'LipMIP_Result': "SKIPPED", 'LipMIP_Time': "SKIPPED",
                    'LiRPA_Result': "SKIPPED", 'LiRPA_Time': "SKIPPED",
                    'Eclipse_Result': "SKIPPED", 'Eclipse_Time': "SKIPPED",
                    'Sampling_Time': "SKIPPED", 'Sampling_File_Path': "SKIPPED"
                }

                # --- A. RUN LIPMIP ---
                # Typically LipMIP is Linf only, so we might want to skip if norm is L2, 
                # unless LipMIP supports L2 in your blueprint. Assuming standard Linf usage here.
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

                # --- C. RUN ECLIPSE (NEW METHOD) ---
                if run_eclipse and norm == 'l2':
                    try:
                        print("    Running Eclipse (L2 Global)...")
                        # Eclipse requires global domain, so we override epsilon to 1.0 (255/255)
                        eclipse_config = config_args.copy()
                        eclipse_config['EPSILON'] = 1.0 

                        st = time.time()
                        with Timeout(timeout_seconds):
                            # compute l2-norm lipschitz constant across global domain
                            eclipse_res = run_eclipse_analysis(eclipse_config)

                        row_data['Eclipse_Time'] = time.time() - st
                        row_data['Eclipse_Result'] = eclipse_res.item()
                        print(f"    Eclipse: {eclipse_res} ({row_data['Eclipse_Time']:.2f}s)")

                    except TimeoutError:
                        print(f"    [!] Eclipse Timed Out ({timeout_seconds}s)")
                        row_data['Eclipse_Result'] = "TIMEOUT"
                        row_data['Eclipse_Time'] = timeout_seconds

                    except Exception as e:
                        print(f"    [!] Eclipse Failed: {e}")
                        row_data['Eclipse_Result'] = "ERROR"
                        row_data['Eclipse_Time'] = 0

                    _append_and_save(row_data, output_csv, columns)
                elif run_eclipse and norm != 'l2':
                    print("    [Info] Eclipse skipped because Norm is not L2.")

                # --- D. RUN SAMPLING ---
                if run_sampling and dataset_specific_sampling_params is not None:
                    try:
                        # --- FIX START: ADJUST EPSILON FOR L2 ---
                        if norm == 'l2':
                            sampling_epsilon = 1.0
                        else:
                            sampling_epsilon = epsilon
                        
                        sampling_config = config_args.copy()
                        sampling_config['EPSILON'] = sampling_epsilon
                        # --- FIX END ---

                        runtime_sampling_args = {
                            "METHOD": dataset_specific_sampling_params['METHOD'],
                            "STEPS": dataset_specific_sampling_params['STEPS'],
                            "WALKERS": dataset_specific_sampling_params['WALKERS'],
                            "TEMP": dataset_specific_sampling_params['TEMP'],
                            "SOBOL_SAMPLES": dataset_specific_sampling_params['SOBOL_SAMPLES'],
                            "BATCH_SIZE": dataset_specific_sampling_params['BATCH_SIZE'],
                            "STEP_SIZE": 0.05 * sampling_epsilon, 
                            "NORM": norm 
                        }
                        
                        st = time.time()
                        with Timeout(timeout_seconds):
                            # Run the sampling
                            acc_inputs, acc_norms = run_lipschitz_sampling(sampling_config, runtime_sampling_args)
                        
                        row_data['Sampling_Time'] = time.time() - st
                        
                        sample_save_name = f"samples_{model_basename}_{img_key}_{norm}.pt"
                        sample_save_path = os.path.join(folder_name, sample_save_name)
                        
                        # --- FIX: Safe CPU Move ---
                        # 1. Define a helper to check if .cpu() exists (works for Tensor vs Numpy)
                        safe_cpu = lambda x: x.cpu() if hasattr(x, 'cpu') else x

                        # 2. Apply it to your data
                        # If it's a Tensor, it moves to CPU. If it's NumPy, it stays as is.
                        payload = (safe_cpu(acc_inputs), safe_cpu(acc_norms))
                        
                        # 3. Save with Protocol 4 (still needed for large NumPy arrays > 4GB)
                        torch.save(payload, sample_save_path, pickle_protocol=4)
                        
                        row_data['Sampling_File_Path'] = sample_save_path
                        print(f"    Sampling ({norm}): Saved to {sample_save_name} (Eps={sampling_epsilon:.2f}, Time={row_data['Sampling_Time']:.2f}s)")

                        # --- CLEANUP ---
                        del acc_inputs
                        del acc_norms
                        del payload
                        gc.collect()
                        torch.cuda.empty_cache()                       

                        # st = time.time()
                        # with Timeout(timeout_seconds):
                        #     # Run the sampling
                        #     acc_inputs, acc_norms = run_lipschitz_sampling(sampling_config, runtime_sampling_args)
                        
                        # row_data['Sampling_Time'] = time.time() - st
                        
                        # sample_save_name = f"samples_{model_basename}_{img_key}_{norm}.pt"
                        # sample_save_path = os.path.join(folder_name, sample_save_name)
                        
                        # # --- FIX 1 & 2: Move to CPU and use Protocol 4 ---
                        # # We move data to CPU immediately to prevent GPU OOM during the save process
                        # # We use pickle_protocol=4 to allow files larger than 4GB
                        # payload = (acc_inputs.cpu(), acc_norms.cpu())
                        # torch.save(payload, sample_save_path, pickle_protocol=4)
                        
                        # row_data['Sampling_File_Path'] = sample_save_path
                        # print(f"    Sampling ({norm}): Saved to {sample_save_name} (Eps={sampling_epsilon:.2f}, Time={row_data['Sampling_Time']:.2f}s)")

                        # # --- FIX 3: EXPLICIT MEMORY CLEANUP ---
                        # # Delete the massive tensors immediately to free RAM
                        # del acc_inputs
                        # del acc_norms
                        # del payload
                        
                        # # Force Python Garbage Collector and Empty CUDA Cache
                        # gc.collect()
                        # torch.cuda.empty_cache()

                    except TimeoutError:
                        print(f"    [!] Sampling Timed Out ({timeout_seconds}s)")
                        row_data['Sampling_File_Path'] = "TIMEOUT"
                        row_data['Sampling_Time'] = timeout_seconds

                    except Exception as e:
                        print(f"    [!] Sampling Failed: {e}")
                        row_data['Sampling_File_Path'] = "ERROR"
                        row_data['Sampling_Time'] = 0
                        
                        # Ensure cleanup happens even on failure
                        if 'acc_inputs' in locals(): del acc_inputs
                        if 'acc_norms' in locals(): del acc_norms
                        gc.collect()
                        torch.cuda.empty_cache()

                    _append_and_save(row_data, output_csv, columns)



                # # --- D. RUN SAMPLING ---
                # if run_sampling and dataset_specific_sampling_params is not None:
                #     try:
                #         # --- FIX START: ADJUST EPSILON FOR L2 ---
                #         # If we are doing L2, we assume we want to sample the Global Lipschitz 
                #         # constant (epsilon=1.0) to match the Eclipse methodology.
                #         if norm == 'l2':
                #             sampling_epsilon = 1.0
                #         else:
                #             sampling_epsilon = epsilon
                        
                #         # Create a specific config for sampling so we don't mess up other methods
                #         sampling_config = config_args.copy()
                #         sampling_config['EPSILON'] = sampling_epsilon
                #         # --- FIX END ---

                #         runtime_sampling_args = {
                #             "METHOD": dataset_specific_sampling_params['METHOD'],
                #             "STEPS": dataset_specific_sampling_params['STEPS'],
                #             "WALKERS": dataset_specific_sampling_params['WALKERS'],
                #             "TEMP": dataset_specific_sampling_params['TEMP'],
                #             "SOBOL_SAMPLES": dataset_specific_sampling_params['SOBOL_SAMPLES'],
                #             "BATCH_SIZE": dataset_specific_sampling_params['BATCH_SIZE'],
                            
                #             # Update step size to match the sampling epsilon
                #             "STEP_SIZE": 0.05 * sampling_epsilon, 
                #             "NORM": norm 
                #         }
                        
                #         st = time.time()
                #         with Timeout(timeout_seconds):
                #             # USE sampling_config HERE, NOT config_args
                #             acc_inputs, acc_norms = run_lipschitz_sampling(sampling_config, runtime_sampling_args)
                        
                #         row_data['Sampling_Time'] = time.time() - st
                        
                #         # Use basename AND norm for saving
                #         sample_save_name = f"samples_{model_basename}_{img_key}_{norm}.pt"
                #         sample_save_path = os.path.join(folder_name, sample_save_name)
                #         torch.save((acc_inputs, acc_norms), sample_save_path)
                        
                #         row_data['Sampling_File_Path'] = sample_save_path
                #         print(f"    Sampling ({norm}): Saved to {sample_save_name} (Eps={sampling_epsilon:.2f}, Time={row_data['Sampling_Time']:.2f}s)")

                #     except TimeoutError:
                #         print(f"    [!] Sampling Timed Out ({timeout_seconds}s)")
                #         row_data['Sampling_File_Path'] = "TIMEOUT"
                #         row_data['Sampling_Time'] = timeout_seconds

                #     except Exception as e:
                #         print(f"    [!] Sampling Failed: {e}")
                #         row_data['Sampling_File_Path'] = "ERROR"
                #         row_data['Sampling_Time'] = 0

                #     _append_and_save(row_data, output_csv, columns)

                image_count += 1

    print("\n=== Experiment Suite Completed ===")

# def _append_and_save(row_data, filename, columns):
#     """
#     Helper to safely update the CSV.
#     """
#     SKIP_VALUES = ["SKIPPED", None]

#     if not os.path.exists(filename):
#         pd.DataFrame([row_data], columns=columns).to_csv(filename, index=False)
#         return

#     df = pd.read_csv(filename)
    
#     # Updated mask to use 'Model_Name' and 'Norm' to identify unique rows
#     mask = (
#         (df['Dataset'] == row_data['Dataset']) & 
#         (df['Model_Name'] == row_data['Model_Name']) & 
#         (df['Image_Name'] == row_data['Image_Name']) &
#         (df['Norm'] == row_data['Norm']) # Differentiate L2 vs Linf runs in the CSV
#     )
    
#     if mask.any():
#         idx = df[mask].index[0]
#         for col in columns:
#             new_val = row_data.get(col)
#             if new_val not in SKIP_VALUES:
#                 df.at[idx, col] = new_val
#     else:
#         new_row_df = pd.DataFrame([row_data], columns=columns)
#         df = pd.concat([df, new_row_df], ignore_index=True)

#     df[columns].to_csv(filename, index=False)

def _append_and_save(row_data, filename, columns):
    """
    Helper to safely update the CSV.
    Handles adding new columns (like 'Norm') to existing CSV files.
    """
    SKIP_VALUES = ["SKIPPED", None]

    # 1. Create file if it doesn't exist
    if not os.path.exists(filename):
        pd.DataFrame([row_data], columns=columns).to_csv(filename, index=False)
        return

    # 2. Read existing file
    df = pd.read_csv(filename)
    
    # --- FIX: Ensure new columns exist in the loaded DF ---
    # This handles the case where you have an old CSV without 'Norm' or 'Eclipse_Result'
    missing_cols = [c for c in columns if c not in df.columns]
    
    if missing_cols:
        print(f"    [CSV Update] Adding missing columns to existing file: {missing_cols}")
        for col in missing_cols:
            # If 'Norm' is missing, we assume old results were 'linf'
            if col == 'Norm':
                df[col] = 'linf'
            else:
                df[col] = None
    # ----------------------------------------------------

    # Updated mask to use 'Model_Name' and 'Norm' to identify unique rows
    mask = (
        (df['Dataset'] == row_data['Dataset']) & 
        (df['Model_Name'] == row_data['Model_Name']) & 
        (df['Image_Name'] == row_data['Image_Name']) &
        (df['Norm'] == row_data['Norm']) 
    )
    
    if mask.any():
        idx = df[mask].index[0]
        for col in columns:
            new_val = row_data.get(col)
            # Only overwrite if we have a real value (not SKIPPED)
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
    # Note: 'NORM' here is a default, but it gets overridden by run_experiment_suite(norm=...)
    mnist_sampling = {
        "METHOD": 'Langevin', 
        "STEPS": 256,
        "WALKERS": 256,
        "TEMP": 1e-1,
        "STEP_SIZE": 0.05 * EPSILON, 
        "SOBOL_SAMPLES": 4000, 
        "BATCH_SIZE": 64, 
        "NORM" : 'linf' 
    }

    cifar_sampling = {
        "METHOD": 'Langevin', 
        "STEPS": 4096,
        "WALKERS": 256,
        "TEMP": 1e-2,
        "STEP_SIZE": 0.05 * EPSILON, 
        "SOBOL_SAMPLES": 4000, 
        "BATCH_SIZE": 64, 
        "NORM" : 'linf'
    }

    tinyimagenet_sampling = {
        "METHOD": 'Langevin', 
        "STEPS": 8192,
        "WALKERS": 256,
        "TEMP": 1e-3,
        "STEP_SIZE": 0.05 * EPSILON, 
        "SOBOL_SAMPLES": 4000, 
        "BATCH_SIZE": 32, 
        "NORM" : 'linf'
    }

    sampling_args_map = {
        'MNIST': mnist_sampling,
        'CIFAR': cifar_sampling,
        'TinyImageNet': tinyimagenet_sampling
    }

    # 2. Define LiRPA Args
    lirpa_args = {'METHOD': 'CROWN-IBP'}
    
    if os.path.exists('test_suite_dictionary_cnn.pt'):
        test_suite_images = torch.load('test_suite_dictionary_cnn.pt')
        
        # EXAMPLE 1: Standard L_inf Run
        print(">>> Running L_inf Suite")
        run_experiment_suite(
            test_image_map=test_suite_images,
            epsilon = EPSILON,
            sampling_args=sampling_args_map,
            max_images=1,
            timeout_seconds=1200, 
            run_lipmip=False,       
            run_lirpa=True,         
            run_sampling=True,      
            run_eclipse=False,
            norm='linf',
            lirpa_args=lirpa_args
        )

        # EXAMPLE 2: L2 Run with Eclipse and Sampling
        print("\n>>> Running L2 Suite")
        run_experiment_suite(
            test_image_map=test_suite_images,
            epsilon = EPSILON, # For sampling local bounds
            sampling_args=sampling_args_map,
            max_images=1,
            timeout_seconds=1200,
            run_lipmip=False,      # Usually off for L2
            run_lirpa=False,       # Usually off for L2
            run_sampling=True,     # Will use L2 sampling
            run_eclipse=True,      # Will use Global L2 Eclipse
            norm='l2',             # <--- Switches the mode
            lirpa_args=lirpa_args
        )

    else:
        print("Please provide the test_suite_dictionary_cnn.pt file.")