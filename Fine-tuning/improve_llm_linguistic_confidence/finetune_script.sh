
#!/bin/bash
#SBATCH --job-name=unc_tuning
#SBATCH --gres=gpu
#SBATCH --output=unc_tuning.out
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --time=07:30:00
#SBATCH --partition=acltr
#SBATCH --exclusive

set -e
set -x

module load CUDA/12.1.1
module load NCCL/2.18.3-GCCcore-12.3.0-CUDA-12.1.1
module load Miniconda3/25.5.1-1

export CONDA_PKGS_DIRS=/home/pavd/miniconda3-25-5-1-1/cache

# Load the conda shell function (required in batch jobs)
source /opt/itu/easybuild/software/Miniconda3/25.5.1-1/etc/profile.d/conda.sh

# Correct activation
conda activate /home/pavd/miniconda3-25-5-1-1/envs/gpu312_pip

export HF_TOKEN=$(cat hf_token.txt)

CUDA_VISIBLE_DEVICES=0 python -m improve_llm_linguistic_confidence mapper=Gemma_mapper_4B