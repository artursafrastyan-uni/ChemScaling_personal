#!/bin/bash -l
#SBATCH --job-name=belka_3_full_opt
#SBATCH --partition=research8
#SBATCH --account=research
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=15:00:00
#SBATCH --array=0-4
#SBATCH --output=/mnt/weka/asafrastyan/ChemScaling_personal/Belka/3B/new/logs/%x_%A_%a.out
#SBATCH --error=/mnt/weka/asafrastyan/ChemScaling_personal/Belka/3B/new/logs/%x_%A_%a.err
#SBATCH --mail-user=arthsafrastyan@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

source /home/asafrastyan/miniforge3/etc/profile.d/conda.sh
conda activate chemproject

export HF_TOKEN=$(cat ~/.hf_token)
export OMP_NUM_THREADS=1

SCRIPTS=(
  "/mnt/weka/asafrastyan/ChemScaling_personal/Belka/3B/new/belka_3_sub_opt_42.py"
  "/mnt/weka/asafrastyan/ChemScaling_personal/Belka/3B/new/belka_3_sub_opt_75.py"
  "/mnt/weka/asafrastyan/ChemScaling_personal/Belka/3B/new/belka_3_sub_opt_85.py"
  "/mnt/weka/asafrastyan/ChemScaling_personal/Belka/3B/new/belka_3_sub_opt_5.py"
  "/mnt/weka/asafrastyan/ChemScaling_personal/Belka/3B/new/belka_3_sub_opt_97.py"
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
