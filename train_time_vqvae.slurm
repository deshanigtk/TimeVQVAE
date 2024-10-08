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
source ../envs/timevq_venv/bin/activate
module load foss/2022a
module load GCCcore/11.3.0
module load Python/3.10.4
module load cuDNN/8.9.3.28-CUDA-12.2.0
module load TensorFlow/2.11.0-CUDA-11.7.0-deeplearn
pip install numpy==1.24.0
pip install pandas
pip install matplotlib
pip install torch==2.2.1
pip install lightning==2.3.1


python stage1.py \
    --dataset_names Wafer \
