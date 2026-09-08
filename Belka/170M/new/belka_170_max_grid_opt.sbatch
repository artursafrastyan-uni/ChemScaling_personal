#!/bin/bash -l
#SBATCH --job-name=belka_170_max_grid_opt
#SBATCH --partition=research8
#SBATCH --account=research
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=250G
#SBATCH --time=14:00:00
#SBATCH --output=/mnt/weka/asafrastyan/ChemScaling_personal/Belka/170M/new/logs/%x_%j.out
#SBATCH --error=/mnt/weka/asafrastyan/ChemScaling_personal/Belka/170M/new/logs/%x_%j.err
#SBATCH --mail-user=arthsafrastyan@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

source /home/asafrastyan/miniforge3/etc/profile.d/conda.sh
conda activate chemproject

export HF_TOKEN=$(cat ~/.hf_token)
export OMP_NUM_THREADS=1

cd /mnt/weka/asafrastyan/ChemScaling_personal/Belka/170M/new
mkdir -p logs

NGPUS=8
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

torchrun \
  --standalone \
  --nproc_per_node=$NGPUS \
  --master_port=$MASTER_PORT \
  belka_170_max_grid_opt.py

