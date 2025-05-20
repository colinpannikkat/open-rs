export ACCELERATE_LOG_LEVEL=info
export TRITON_CACHE_DIR=".triton"

TASK="helm|mmlu|0|0"
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" #"./data/OpenRS-RLoRA-LoftQ"
LORA="./data/OpenRS-RLoRA-LoftQ"
MODEL_ARGS="pretrained=$MODEL,lora_path=$LORA,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

OUTPUT_DIR="logs/evals/full-math-bench-tasks/$LORA"

lighteval vllm $MODEL_ARGS "$TASK" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR