optimizer-agent \
    --baseline ./examples/generic_task/baseline.txt \
    --evaluate ./examples/generic_task/evaluate.py \
    --prompt-config-file ./examples/generic_task/prompt_config.json \
    --feedback-config-file ./examples/feedback/evaluator_supplied.json \
    --num-iterations 3 \
    --num-samples 2 \
    --experiments-dir opt_agent_generic_experiment

# Compare strategies by swapping the feedback config, e.g.:
#   --feedback-config-file ./examples/feedback/generic_rules.json
# or select directly:
#   --feedback-strategy none | prompt | rules | rules_llm | llm | evaluator

# Smoke test without a model server:
# optimizer-agent ... --test
