#!/bin/bash

# Define the list of methods
methods=("l1" "l2" "scalar")

# Loop through each method and execute the python script
for method_name in "${methods[@]}"; do
    echo "------------------------------------------------"
    echo "Running evaluation for method: $method_name"
    echo "------------------------------------------------"
    
    python run_eval_pg_sentiment_ablation.py \
        --dataset sst2 \
        --model bert \
        --method "$method_name"
done

echo "All evaluations complete!"