optimizer-agent \
    --baseline ./examples/test_example/baseline.py \
    --reference ./examples/test_example/reference.py \
    --evaluate ./examples/test_example/evaluate.py \
    --prompt-config-file ./examples/test_example/prompt_config.json \
    --num-iterations 2 \
    --num-map-elites 3 \
    --predecessor-selection-method map_elites \
    --test

# ", type=str, default="bestN_last", help="Selection Algorithm", choices=["bestN_last","bestN_all","map_elites"]
# optimizer-agent --baseline baseline.py --evaluate evaluate.py