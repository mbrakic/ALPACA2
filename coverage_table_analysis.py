import pandas as pd

CSV_PATH = 'coverage_test.csv'

df = pd.read_csv(CSV_PATH)

# Coerce LipMIP columns to numeric (handles "TIMEOUT" / "ERROR" strings)
df['LipMIP_Result'] = pd.to_numeric(df['LipMIP_Result'], errors='coerce')
df['LipMIP_Time']   = pd.to_numeric(df['LipMIP_Time'],   errors='coerce')

# Only rows where LipMIP actually returned a numeric result
valid = df.dropna(subset=['LipMIP_Result', 'LipMIP_Time'])

# --- FULL DATASET ---
total_successful   = len(valid)
total_all_runs     = valid['ALPACA_Attempts'].sum()
underestimates     = valid[valid['ALPACA_Endpoint'] < valid['LipMIP_Result']]
n_under            = len(underestimates)

pct_of_successful  = 100 * n_under / total_successful if total_successful else float('nan')
pct_of_all_runs    = 100 * n_under / total_all_runs   if total_all_runs   else float('nan')

# --- FILTER A: remove timed-out rows where LipMIP_Result > ALPACA_Endpoint ---
timed_out_and_over  = (valid['LipMIP_Time'] > 150.0) & (valid['LipMIP_Result'] > valid['ALPACA_Endpoint'])
filtered_a          = valid[~timed_out_and_over]

total_successful_a  = len(filtered_a)
total_all_runs_a    = filtered_a['ALPACA_Attempts'].sum()
underestimates_a    = filtered_a[filtered_a['ALPACA_Endpoint'] < filtered_a['LipMIP_Result']]
n_under_a           = len(underestimates_a)

pct_of_successful_a = 100 * n_under_a / total_successful_a if total_successful_a else float('nan')
pct_of_all_runs_a   = 100 * n_under_a / total_all_runs_a   if total_all_runs_a   else float('nan')

n_removed_a = total_successful - total_successful_a

# --- FILTER B: remove ALL timed-out rows (regardless of LipMIP_Result vs ALPACA) ---
timed_out_any       = valid['LipMIP_Time'] > 150.0
filtered_b          = valid[~timed_out_any]

total_successful_b  = len(filtered_b)
total_all_runs_b    = filtered_b['ALPACA_Attempts'].sum()
underestimates_b    = filtered_b[filtered_b['ALPACA_Endpoint'] < filtered_b['LipMIP_Result']]
n_under_b           = len(underestimates_b)

pct_of_successful_b = 100 * n_under_b / total_successful_b if total_successful_b else float('nan')
pct_of_all_runs_b   = 100 * n_under_b / total_all_runs_b   if total_all_runs_b   else float('nan')

n_removed_b = total_successful - total_successful_b

# --- OUTPUT ---
print("=" * 55)
print("FULL DATASET")
print("=" * 55)
print(f"  Successful runs (rows with LipMIP result): {total_successful}")
print(f"  Total ALPACA runs (sum of attempts):       {total_all_runs}")
print(f"  Underestimates (ALPACA < LipMIP):          {n_under}")
print(f"  Underestimates / successful runs:          {pct_of_successful:.2f}%")
print(f"  Underestimates / all ALPACA runs:          {pct_of_all_runs:.2f}%")

print()
print("=" * 55)
print(f"FILTER A — removed {n_removed_a} rows: timed-out LipMIP > ALPACA")
print("=" * 55)
print(f"  Successful runs remaining:                 {total_successful_a}")
print(f"  Total ALPACA runs (sum of attempts):       {total_all_runs_a}")
print(f"  Underestimates (ALPACA < LipMIP):          {n_under_a}")
print(f"  Underestimates / successful runs:          {pct_of_successful_a:.2f}%")
print(f"  Underestimates / all ALPACA runs:          {pct_of_all_runs_a:.2f}%")

print()
print("=" * 55)
print(f"FILTER B — removed {n_removed_b} rows: all timed-out LipMIP")
print("=" * 55)
print(f"  Successful runs remaining:                 {total_successful_b}")
print(f"  Total ALPACA runs (sum of attempts):       {total_all_runs_b}")
print(f"  Underestimates (ALPACA < LipMIP):          {n_under_b}")
print(f"  Underestimates / successful runs:          {pct_of_successful_b:.2f}%")
print(f"  Underestimates / all ALPACA runs:          {pct_of_all_runs_b:.2f}%")
