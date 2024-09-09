#!/bin/bash

# The project ID which this job should run under:
#SBATCH --account="punim0512"
# The name of the job:
#SBATCH --job-name="time-vqvae-train"
# Partition for the job:
#SBATCH --partition deeplearn
#SBATCH --qos gpgpudeeplearn
# Number of GPUs requested per node:
#SBATCH --gres=gpu:1
# Maximum number of tasks/CPU cores used by the job:
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=00-05:00:00
# Send yourself an email when the job:
#SBATCH --mail-user=dpoddenige@student.unimelb.edu.au
#SBATCH --mail-type=BEGIN,FAIL,END

# The modules to load:
source ../envs/time_vqvae_venv/bin/activate
module load foss/2022a
module load GCCcore/11.3.0
module load Python/3.10.4
module load PyTorch/2.1.2-CUDA-11.7.0
module load torchvision/0.13.1-CUDA-11.7.0
module load cuDNN/8.9.3.28-CUDA-12.2.0
pip install einops
pip install numpy==1.26.4
pip install matplotlib
pip install scikit-learn
pip install wandb
pip install pandas
pip install supervised-fcn-2
pip install numba
pip install x-transformers==1.31.6
pip install pytorch-lightning==2.3.1

python stage1.py \
    --dataset_names Wafer \
    --gpu_device_ind 0 \
