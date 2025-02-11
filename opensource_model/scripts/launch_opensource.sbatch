#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=1
#SBATCH --job-name fuyu_n2
#SBATCH --time=48:00:00  # Increased time for long tasks

# gpt4o_n1000
# Load the Conda commands   into the environment
#source /path/to/conda.sh  # Update this path to the location of your conda.sh
# Activate the Conda environment
source /data/hengyi_wang/miniconda3/bin/conda
conda init
# Activate the Conda environment
conda activate fuyu
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
# Set your environment variable here
export BEGIN=0 # 200     
export N_SEQ=100 # number of samples to test
export N_NEEDLES=2 # number of needles
# exclusively for Gemini-1.5-pro
#export MODEL_VERSION=gemini-1.5-pro-latest
#export RATE=2

echo $MASTER_ADDR


srun bash opensource_model/scripts/test_opensource.sh
