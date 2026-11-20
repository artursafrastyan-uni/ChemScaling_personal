#!/bin/bash -l
#SBATCH --job-name=belka_13_max_grid_opt
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=250G
#SBATCH --time=30:00:00
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
#SBATCH --mail-user=<YOUR_EMAIL>
#SBATCH --mail-type=BEGIN,END,FAIL

source /path/to/conda/etc/profile.d/conda.sh
conda activate <YOUR_CONDA_ENV>

export HF_TOKEN="${HF_TOKEN:-<YOUR_HF_TOKEN>}"
export OMP_NUM_THREADS=1

cd /path/to/project/Belka/<MODEL_SIZE>/new
mkdir -p logs

NGPUS=8
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

torchrun \
  --standalone \
  --nproc_per_node=$NGPUS \
  --master_port=$MASTER_PORT \
  belka_13_max_grid_opt.py
