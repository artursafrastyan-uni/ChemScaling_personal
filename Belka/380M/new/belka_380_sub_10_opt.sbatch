#!/bin/bash -l
#SBATCH --job-name=belka_380_sub_10_opt
#SBATCH --partition=research
#SBATCH --account=research
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/weka/asafrastyan/ChemScaling_personal/Belka/380M/new/logs/%x_%j.out
#SBATCH --error=/mnt/weka/asafrastyan/ChemScaling_personal/Belka/380M/new/logs/%x_%j.err
#SBATCH --mail-user=arthsafrastyan@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

source /home/asafrastyan/miniforge3/etc/profile.d/conda.sh
conda activate chemproject

export HF_TOKEN=$(cat ~/.hf_token)
export OMP_NUM_THREADS=1

cd /mnt/weka/asafrastyan/ChemScaling_personal/Belka/380M/new
mkdir -p /mnt/weka/asafrastyan/ChemScaling_personal/Belka/380M/new/logs

python3 belka_380_sub_10_opt.py
