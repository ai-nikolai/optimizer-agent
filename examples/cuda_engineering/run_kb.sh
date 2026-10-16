# Example: run the optimizer agent with the KernelBench (TOML-driven) prompt
# style. This needs the `robust_kernelbench` repo checked out next to the
# optimizer_agent package (the KB prompt builder resolves kb_prompts from there).
# GPU + torch are required by the real evaluator (examples/cuda_engineering/evaluate.py).

CUDA_VISIBLE_DEVICES=1 optimizer-agent \
    --use-kb-prompt \
    --prompt-config-file ./examples/cuda_engineering/prompt_config_kb.json \
    --baseline ./examples/cuda_engineering/baseline.py \
    --reference ./examples/cuda_engineering/reference.py \
    --evaluate ./examples/cuda_engineering/evaluate.py \
    --num-iterations 2 \
    --num-islands 1 \
    --predecessor-selection-method map_elites \
    --num-map-elites 3 \
    --feedback-strategy error_case

# Quick CPU-only smoke test (no GPU), using the random-scorer example:
optimizer-agent \
    --use-kb-prompt \
    --prompt-config-file ./examples/cuda_engineering/prompt_config_kb.json \
    --baseline ./examples/test_example/baseline.py \
    --reference ./examples/test_example/reference.py \
    --evaluate ./examples/test_example/evaluate.py \
    --test
