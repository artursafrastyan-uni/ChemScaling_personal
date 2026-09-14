#!/bin/bash
#SBATCH --job-name=polaris_inference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/weka/asafrastyan/ChemScaling_personal/logs/%x_%j.out
#SBATCH --error=/mnt/weka/asafrastyan/ChemScaling_personal/logs/%x_%j.err
#SBATCH --account=research
#SBATCH --partition=research

source /home/asafrastyan/miniforge3/bin/activate
conda activate chemproject

cd /mnt/weka/asafrastyan/ChemScaling_personal
python polaris_inference_all_sizes.py
