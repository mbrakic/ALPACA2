import pandas as pd
import torch
import os
import time
from tqdm import tqdm

try:
    from running_blueprint import run_alpaca
except ImportError:
    pass

# --- CONFIGURATION ---
RESULTS_CSV = 'experiment_results_relunet.csv'
FINAL_DB_PATH = 'final_collated_results_relunet.csv'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA = 0.01 

alpaca_args = { "GAMMA": GAMMA }

# --- HELPER: SAFE TENSOR EXTRACTION ---
def safe_item(value):
    """Safely extracts a python scalar from a tensor or returns the value itself."""
    if torch.is_tensor(value):
        return value.item()
    return value

def collate_and_save_results(csv_path, output_path):
    """
    Reads the experiment log, runs ALPACA on valid rows, measures execution time,
    and collates all time/success metrics into a new CSV.
    """
    
    # 1. Load Data
    if not os.path.exists(csv_path):
        print(f"[!] Input file {csv_path} not found.")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows. Initializing new columns...")

    # 2. Initialize New Columns (Metrics + Eclipse/Norm features)
    new_cols = [
        'ALPACA_Success', 
        'ALPACA_Endpoint', 
        'ALPACA_Notes', 
        'ALPACA_Compute_Time', 
        'Total_Pipeline_Time'
    ]
    
    # --- FEATURE ADDITION: Norm & Eclipse Support ---
    # Ensure 'Norm' exists (default to 'linf' if missing)
    if 'Norm' not in df.columns:
        df['Norm'] = 'linf'
        
    # Ensure Eclipse columns exist (for comparison if available)
    if 'Eclipse_Result' not in df.columns:
        df['Eclipse_Result'] = None
    if 'Eclipse_Time' not in df.columns:
        df['Eclipse_Time'] = None

    for col in new_cols:
        if col not in df.columns:
            df[col] = None

    # 3. Filter Valid Rows
    #    Must have a valid file path, no previous errors, no timeouts, and not skipped
    mask = (
        (df['Sampling_File_Path'].notna()) & 
        (df['Sampling_File_Path'] != 'ERROR') & 
        (df['Sampling_File_Path'] != 'TIMEOUT') &
        (df['Sampling_File_Path'] != 'SKIPPED')
    )
    valid_rows = df[mask]
    print(f"Found {len(valid_rows)} valid sampling files to process.")

    # 4. Processing Loop
    for idx, row in tqdm(valid_rows.iterrows(), total=len(valid_rows), desc="Processing ALPACA"):
        
        file_path = row['Sampling_File_Path']
        
        # Guard clause: File missing
        if not os.path.exists(file_path):
            df.at[idx, 'ALPACA_Notes'] = "FILE_NOT_FOUND"
            continue

        try:
            # A. Load Data
            data = torch.load(file_path, map_location=DEVICE)
            
            # --- SAFETY CHECK: Data Unpacking ---
            # Handles cases where data might not be a tuple of length 2
            if isinstance(data, (tuple, list)) and len(data) == 2:
                acc_inputs, acc_norms = data
            else:
                raise ValueError(f"Unexpected data format in .pt file: {type(data)}")
            
            # B. Run ALPACA & Measure Time
            start_time = time.time()
            alpaca_result = run_alpaca(alpaca_args, acc_inputs, acc_norms)
            end_time = time.time()
            
            compute_duration = end_time - start_time

            # C. Extract Data (Using safe_item helper to prevent .item() crashes)
            # We use .get() to avoid KeyErrors, then safe_item to handle tensors vs floats
            success = safe_item(alpaca_result.get("Success")).item()
            endpoint = safe_item(alpaca_result.get("Est_Endpoint")).item()
            notes = safe_item(alpaca_result.get("Notes")).item()

            # D. Update DataFrame
            df.at[idx, 'ALPACA_Success'] = success
            df.at[idx, 'ALPACA_Endpoint'] = endpoint
            df.at[idx, 'ALPACA_Notes'] = notes
            df.at[idx, 'ALPACA_Compute_Time'] = compute_duration
            
            # E. Calculate Total Pipeline Time (Sampling + ALPACA)
            sampling_time = row.get('Sampling_Time', 0)
            
            # Sanitize Sampling_Time (handle strings like 'TIMEOUT' or NaNs)
            try:
                s_time_float = float(sampling_time)
            except (ValueError, TypeError):
                s_time_float = 0.0
            
            df.at[idx, 'Total_Pipeline_Time'] = s_time_float + compute_duration

        except Exception as e:
            # Catch unexpected errors (corrupt file, runtime error, CUDA OOM)
            df.at[idx, 'ALPACA_Notes'] = f"ERROR: {str(e)}"
            df.at[idx, 'ALPACA_Success'] = False

    # 5. Save Final Database
    df.to_csv(output_path, index=False)
    print(f"\nSuccess! Collated database saved to: {output_path}")

    return df

if __name__ == "__main__":
    # Pandas formatting for cleaner output
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:.4f}'.format)

    # Run the collation logic
    final_df = collate_and_save_results(RESULTS_CSV, FINAL_DB_PATH)

    # Reload to ensure clean types for display
    if os.path.exists(FINAL_DB_PATH):
        final_df = pd.read_csv(FINAL_DB_PATH)

        # --- UPDATED SELECTION COLUMNS ---
        # Adapted for ReLuNet (uses Model_Size) + New Features (Norm, Eclipse)
        selected_columns = [
            'Dataset', 
            'Model_Size',       # Specific to ReLuNet
            'Norm',             # Added Feature
            'LipMIP_Result', 
            #'LipMIP_Time',     # Optional
            'LiRPA_Result', 
            #'LiRPA_Time',      # Optional
            'Eclipse_Result',   # Added Feature
            'Eclipse_Time',     # Added Feature
            'ALPACA_Endpoint', 
            'Total_Pipeline_Time'
        ]

        # Filter to only show columns that actually exist in the CSV
        valid_cols = [c for c in selected_columns if c in final_df.columns]

        print("\n=== FINAL REPORT (ReLuNet) ===")
        print(final_df[valid_cols])
    else:
        print("No results file generated.")