#!/bin/sh
# Base model configuration
MODEL="./data/OpenRS-GRPO/checkpoint-300"
BASE_MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

OUTPUT_DIR="logs/evals/full-math-bench-tasks/$MODEL"

mkdir -p "$OUTPUT_DIR"

# Define evaluation tasks
TASKS="aime24 math_500 amc23 minerva olympiadbench"
# TASKS="aime24 math_500 amc23"

# Run evaluations for each task
for task in $TASKS; do
    echo "Evaluating task: $task"
    lighteval vllm "$BASE_MODEL_ARGS" "custom|$task|0|0" \
        --custom-tasks src/open_r1/evaluate.py \
        --use-chat-template \
        --output-dir "$OUTPUT_DIR"
done
