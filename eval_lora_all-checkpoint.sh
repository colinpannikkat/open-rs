#!/bin/sh

export ACCELERATE_LOG_LEVEL=info
export TRITON_CACHE_DIR=".triton"

# Base model configuration
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
LORA_MODEL=$1

for checkpoint in $(seq 50 50 500); do
    LORA="./data/$LORA_MODEL/checkpoint-$checkpoint"
    echo "Evaluating LORA adapter: $LORA"

    MODEL_ARGS="pretrained=$MODEL,lora_path=$LORA,max_lora_rank=64,dtype=bfloat16,max_model_length=32768,max_num_batched_tokens=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

    OUTPUT_DIR="logs/evals/full-math-bench-tasks/$LORA"

    mkdir -p "$OUTPUT_DIR"

    # Define evaluation tasks
    TASKS="aime24 math_500 amc23 minerva olympiadbench"
    TASKS_BBH="harness|bbh:boolean_expressions|0|0,harness|bbh:causal_judgment|0|0,harness|bbh:date_understanding|0|0,harness|bbh:disambiguation_qa|0|0,harness|bbh:dyck_languages|0|0,harness|bbh:formal_fallacies|0|0,harness|bbh:geometric_shapes|0|0,harness|bbh:hyperbaton|0|0,harness|bbh:logical_deduction_five_objects|0|0,harness|bbh:logical_deduction_seven_objects|0|0,harness|bbh:logical_deduction_three_objects|0|0,harness|bbh:movie_recommendation|0|0,harness|bbh:multistep_arithmetic_two|0|0,harness|bbh:navigate|0|0,harness|bbh:object_counting|0|0,harness|bbh:penguins_in_a_table|0|0,harness|bbh:reasoning_about_colored_objects|0|0,harness|bbh:ruin_names|0|0,harness|bbh:salient_translation_error_detection|0|0,harness|bbh:snarks|0|0,harness|bbh:sports_understanding|0|0,harness|bbh:temporal_sequences|0|0,harness|bbh:tracking_shuffled_objects_five_objects|0|0,harness|bbh:tracking_shuffled_objects_seven_objects|0|0,harness|bbh:tracking_shuffled_objects_three_objects|0|0,harness|bbh:web_of_lies|0|0,harness|bbh:word_sorting|0|0"
    TASKS_MMLU="helm|mmlu|0|0"

    # Run evaluations for each task
    for task in $TASKS; do
        echo "Evaluating task: $task"
        lighteval vllm "$MODEL_ARGS" "custom|$task|0|0" \
            --custom-tasks src/open_r1/evaluate.py \
            --use-chat-template \
            --output-dir "$OUTPUT_DIR"
    done

    echo "Evaluating BBH"
    lighteval vllm $MODEL_ARGS "$TASKS_BBH" \
        --use-chat-template \
        --output-dir $OUTPUT_DIR

    echo "Evaluating MMLU"
    lighteval vllm $MODEL_ARGS "$TASKS_MMLU" \
        --use-chat-template \
        --output-dir $OUTPUT_DIR

done