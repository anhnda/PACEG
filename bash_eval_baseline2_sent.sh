#!/bin/bash

# Define the list of scripts to run
scripts=(
    "run_eval_attcat_sentiment.py"
    "run_eval_ig_sentiment.py"
    "run_eval_pg_sentiment.py"
    "run_slalom_eval_sentiment.py"
    "run_eval_reagent_sentiment.py"
)

# Define the list of models
models=("bert" "roberta" "distillbert")

# Define the list of possible methods
methods=("mask" "pad" "zero" "mean")

# 1. Iterate through each script
for script in "${scripts[@]}"; do
    # 2. Iterate through each model
    for model in "${models[@]}"; do
        # 3. Iterate through each baseline method
        for baseline in "${methods[@]}"; do
            # 4. Iterate through each evaluation baseline method
            for eval_baseline in "${methods[@]}"; do
                
                echo "=========================================================="
                echo "SCRIPT:   $script"
                echo "MODEL:    $model"
                echo "BASELINE: $baseline | EVAL: $eval_baseline"
                echo "=========================================================="
                
                # Execute the python script
                python "$script" \
                    --model "$model" \
                    --eval-baseline "$eval_baseline" \
                    --dataset sst2
                
                # Check if the command succeeded
                if [ $? -ne 0 ]; then
                    echo "ERROR: $script failed for $model (BL: $baseline, EV: $eval_baseline)"
                fi
                
                echo -e "\n"
            done
        done
    done
done

echo "All tests completed across all models and scripts."