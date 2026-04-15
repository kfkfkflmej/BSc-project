#!/bin/bash
#SBATCH --job-name=bench
#SBATCH --output=bench.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --time=07:30:00
#SBATCH --partition=acltr
# #SBATCH --exclusive
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END          # Send an email when the job finishes

set -e
set -x

module load Anaconda3
module load CUDA
module load GCC
source "/opt/itu/easybuild/software/Anaconda3/2025.06-1/etc/profile.d/conda.sh"

export CONDA_PKGS_DIRS=/home/pavd/conda_pkgs_cache/


conda activate unc_com

export HF_TOKEN=$(cat hf_token.txt)

models=(
'google/gemma-3-4b-it'
'OOOss/gemma3-4B-it-uncertain'
)

datasets=(
    'truthfulQA'
    'commonsenseQA'
    'gsm8k'
    'triviaQA'
    'nq_open'
    'pop_qa_subset'
    'simple_qa_test_set'
)

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        model_name=$(basename "$model")
        data_path="data/${dataset}.csv"

        PYTHONPATH=lib python3 bin/gen_estimator_tree.py \
            --model "$model" \
            --data_path "$data_path" \
            --dataset "$dataset" \
            --branch_num 8 \
            --sampling_num 16 \
            --max_tokens 2048 \
            --temperature 1.0 \
            --tensor_parallel_size 1 \
            --gpu_memory_utilization 0.9
    done
done