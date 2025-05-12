export ACCELERATE_LOG_LEVEL=info
export TRITON_CACHE_DIR=".triton"

TASK="helm|mmlu|0|0"
MODEL="./data/OpenRS-GRPO/checkpoint-300"
# MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
OUTPUT_DIR="logs/evals/mmlu/$MODEL"
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

lighteval vllm $MODEL_ARGS "$TASK" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR