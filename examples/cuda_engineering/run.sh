CUDA_VISIBLE_DEVICES=1 optimizer-agent \
    --baseline ./examples/cuda_engineering/baseline.py \
    --reference ./examples/cuda_engineering/reference.py \
    --evaluate ./examples/cuda_engineering/evaluate.py \
    --prompt-config-file ./examples/cuda_engineering/prompt_config.json


# optimizer-agent --baseline baseline.py --evaluate evaluate.py