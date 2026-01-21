import pandas as pd
import torch
import os
import time
from tqdm import tqdm

from running_blueprint import run_alpaca

# --- CONFIGURATION ---
RESULTS_CSV = 'experiment_results_2.csv'
FINAL_DB_PATH = 'final_collated_results.csv'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA = 0.001 

alpaca_args = { "GAMMA": GAMMA }

# Columns to display in the final preview
# (We will save ALL columns, but this is what we print to console)
PREVIEW_COLS = [
    'Dataset', 
    'Model_Size', 
    'Sampling_Time',          # Original Sampling Time
    'ALPACA_Compute_Time',    # New Processing Time
    'Total_Pipeline_Time',    # Sum of both
    'ALPACA_Success', 
    'ALPACA_Endpoint'
]

def collate_and_save_results(csv_path, output_path):
    """
    Reads the experiment log, runs ALPACA on valid rows, measures execution time,
    and collates all time/success metrics into a new CSV.
    """
    
    # 1. Load Data
    if not os.path.exists(csv_path):
        print(f"[!] Input file {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows. Initializing new columns...")

    # 2. Initialize New Columns
    new_cols = [
        'ALPACA_Success', 
        'ALPACA_Endpoint', 
        'ALPACA_Notes', 
        'ALPACA_Compute_Time', 
        'Total_Pipeline_Time'
    ]
    
    for col in new_cols:
        if col not in df.columns:
            df[col] = None

    # 3. Filter Valid Rows
    #    Must have a valid file path, no previous errors, and no timeouts
    mask = (
        (df['Sampling_File_Path'].notna()) & 
        (df['Sampling_File_Path'] != 'ERROR') & 
        (df['Sampling_File_Path'] != 'TIMEOUT')
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
            acc_inputs, acc_norms = data
            
            # B. Run ALPACA & Measure Time
            start_time = time.time()
            alpaca_result = run_alpaca(alpaca_args, acc_inputs, acc_norms)
            end_time = time.time()
            
            compute_duration = end_time - start_time

            # C. Extract Data (Safety check using .get for dictionary)
            success = alpaca_result.get("Success").item()
            endpoint = alpaca_result.get("Est_Endpoint").item()
            notes = alpaca_result.get("Notes").item()

            # D. Update DataFrame
            df.at[idx, 'ALPACA_Success'] = success
            df.at[idx, 'ALPACA_Endpoint'] = endpoint
            df.at[idx, 'ALPACA_Notes'] = notes
            df.at[idx, 'ALPACA_Compute_Time'] = compute_duration
            
            # E. Calculate Total Pipeline Time (Sampling + ALPACA)
            # Ensure Sampling_Time is a number, treat NaN as 0 for summation if needed
            sampling_time = row.get('Sampling_Time', 0)
            if pd.isna(sampling_time): sampling_time = 0
            
            df.at[idx, 'Total_Pipeline_Time'] = float(sampling_time) + compute_duration

        except Exception as e:
            # Catch unexpected errors (corrupt file, runtime error)
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

    final_df = pd.read_csv(FINAL_DB_PATH)

    selected_columns = [
        'Dataset', 'Model_Size', 'LipMIP_Result', 'LipMIP_Time', 'LiRPA_Result',
        'LiRPA_Time', 'ALPACA_Endpoint', 'Total_Pipeline_Time'
    ]

    rel_df = final_df[selected_columns]
    print(rel_df)

    # # Print Preview
    # if final_df is not None:
    #     print("\n--- COLLATED RESULTS PREVIEW ---")
    #     # Filter strictly for columns that exist to prevent KeyErrors
    #     valid_preview_cols = [c for c in PREVIEW_COLS if c in final_df.columns]
    #     print(final_df[valid_preview_cols].head(10))