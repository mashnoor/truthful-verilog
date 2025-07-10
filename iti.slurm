#!/bin/bash

#SBATCH --time=3:00:00 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --output=iti-verilog%J.out
#SBATCH --job-name=iti-verilog


# Load the Python and CUDA modules
module load python



# Activate the GPU version of PyTorch
source venv/bin/activate

# Run PyTorch Training
echo "Training Start:"
time python3 iti_verilog.py
echo


# You're done!
echo "Ending script..."
date
