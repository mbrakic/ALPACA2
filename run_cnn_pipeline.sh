#!/bin/bash

# 1. Initialize Conda using your specific path
source /home/milan/anaconda3/etc/profile.d/conda.sh

# 2. Activate the base environment
conda activate base

# 3. Run the scripts (Force Run: no error checking between steps)
echo "--- Step 1: Running train_all_models.py ---"
python train_all_models.py

echo "--- Step 2: Running experiment_runner_cnn.py ---"
# This runs even if Step 1 crashes
python experiment_runner_cnn.py