# Example: run the optimizer agent on the real CUDA softplus kernel.
#
# The agent is generic now: it reads the reference + baseline from the files
# below and scores candidates with the CUDA evaluator (examples/cuda_engineering/
# evaluate.py). GPU + torch are required by that evaluator.
#
# Set CUDA_VISIBLE_DEVICES to the GPU you want to benchmark on.
CUDA_VISIBLE_DEVICES=1 optimizer-agent \
    --baseline ./examples/cuda_engineering/baseline.py \
    --reference ./examples/cuda_engineering/reference.py \
    --evaluate ./examples/cuda_engineering/evaluate.py \
    --prompt-config-file ./examples/cuda_engineering/prompt_config.json \
    --num-iterations 2 \
    --num-islands 1 \
    --predecessor-selection-method map_elites \
    --num-map-elites 3 \
    --use-debug

# Quick CPU-only smoke test (random scorer, FakeLLMServer, no GPU):
# optimizer-agent \
#     --baseline ./examples/test_example/baseline.py \
#     --reference ./examples/test_example/reference.py \
#     --evaluate ./examples/test_example/evaluate.py \
#     --test
