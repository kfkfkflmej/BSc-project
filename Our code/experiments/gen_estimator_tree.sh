
datasets=(
    'truthfulQA'
    'commonsenseQA'
    'math'
    #'triviaQA'
    'nq_open'
    #'popQA'
    'simpleQA'
)

for dataset in "${datasets[@]}"; do
    
    model_name=$(basename "")
    data_path="datasets/data/${dataset}.csv"

    
    if [ -f "$output" ]; then
        echo "Skipping $model_name on $dataset"
        continue        
    fi

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
