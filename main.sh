#!/bin/bash

### TC2 Job Script ###

#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1

### Specify Memory allocate to this job ###
#SBATCH --mem=1G

### Specify number of core (CPU) to allocate to per task ###
#SBATCH --cpus-per-task=1

### Specify number of node to compute ###
#SBATCH --nodes=1

### Optional: Specify node to execute the job ###
### Remove 1st # at next line for the option to take effect ###
##SBATCH --nodelist=TC2N01

### Specify Time Limit, format: <min> or <min>:<sec> or <hr>:<min>:<sec> or <days>-<hr>:<min>:<sec> or <days>-<hr> ###
#SBATCH --time=6:00:00

### Specify name for the job, filename format for output and error ###
#SBATCH --job-name=HSTU_b
#SBATCH --output=output_HSTU_b.out
#SBATCH --error=error_HSTU_b.err

### Your script for computation ###
module load anaconda
eval "$(conda shell.bash hook)"
conda activate PYTHON3.10_TORCH2.8
python CUDA_VISIBLE_DEVICES=0 python3 main.py --gin_config_file=configs/amzn23_office/hstu-sampled-softmax-n512-blair.gin --master_port=12345
