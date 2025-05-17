export ACCELERATE_LOG_LEVEL=info
export TRITON_CACHE_DIR=".triton"
nohup accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes=1 src/open_r1/grpo.py --config recipes/rlora-loftq.yaml --cosine_max_len 3584 2>&1 >> rlora-loftq-train.out &