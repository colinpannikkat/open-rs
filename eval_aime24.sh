# MODEL=knoveleng/Open-RS3
MODEL="./data/OpenRS-GRPO/checkpoint-300"
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/aime24/$MODEL

# Example: AIME 2024
TASK=aime24
lighteval vllm "$MODEL_ARGS" "custom|$TASK|0|0" \
  --custom-tasks src/open_r1/evaluate.py \
  --use-chat-template \
  --output-dir "$OUTPUT_DIR"
