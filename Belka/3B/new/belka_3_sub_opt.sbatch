#!/bin/bash -l
#SBATCH --job-name=belka_3_full_opt
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=15:00:00
#SBATCH --array=0-4
#SBATCH --output=./logs/%x_%A_%a.out
#SBATCH --error=./logs/%x_%A_%a.err
#SBATCH --mail-user=<YOUR_EMAIL>
#SBATCH --mail-type=BEGIN,END,FAIL

source /path/to/conda/etc/profile.d/conda.sh
conda activate <YOUR_CONDA_ENV>

export HF_TOKEN="${HF_TOKEN:-<YOUR_HF_TOKEN>}"
export OMP_NUM_THREADS=1

SCRIPTS=(
  "/path/to/project/Belka/<MODEL_SIZE>/new/belka_3_sub_opt_42.py"
  "/path/to/project/Belka/<MODEL_SIZE>/new/belka_3_sub_opt_75.py"
  "/path/to/project/Belka/<MODEL_SIZE>/new/belka_3_sub_opt_85.py"
  "/path/to/project/Belka/<MODEL_SIZE>/new/belka_3_sub_opt_5.py"
  "/path/to/project/Belka/<MODEL_SIZE>/new/belka_3_sub_opt_97.py"
)
CURRENT_SCRIPT=${SCRIPTS[$SLURM_ARRAY_TASK_ID]}
SCRIPT_DIR=$(dirname "$CURRENT_SCRIPT")

cd "$SCRIPT_DIR"
mkdir -p logs

NGPUS=8
MASTER_PORT=$((29500 + (SLURM_ARRAY_JOB_ID + SLURM_ARRAY_TASK_ID * 100) % 1000))

echo "Running array task $SLURM_ARRAY_TASK_ID: $CURRENT_SCRIPT in $SCRIPT_DIR on $NGPUS GPUs (Port: $MASTER_PORT)"

torchrun \
  --standalone \
  --nproc_per_node=$NGPUS \
  --master_port=$MASTER_PORT \
  "$CURRENT_SCRIPT"
