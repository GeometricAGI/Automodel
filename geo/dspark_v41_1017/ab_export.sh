#!/usr/bin/env bash
# Serve an exported drafter and the original checkpoint in turn; replay the same prompts greedily on each.
#   SERVE_SCRIPT=<serve script taking extra vllm args> ./ab_export.sh <exported checkpoint dir>
# SERVE_SCRIPT must serve "$DSPARK_CKPT" on port 8000 with the production engine arguments.
set -u
cd "$(dirname "$0")"
EXPORT=$1
ORIGINAL=${DSPARK_CKPT:-$HOME/models/DeepSeek-V4.1-Flash-1017-vLLM-B300}
for name in export original; do
  model=$([ $name = export ] && echo "$EXPORT" || echo "$ORIGINAL")
  DSPARK_CKPT=$model bash "$SERVE_SCRIPT" --safetensors-load-strategy lazy \
    --speculative-config '{"method": "dspark", "num_speculative_tokens": 3, "draft_sample_method": "probabilistic"}' \
    > serve_$name.log 2>&1 &
  pid=$!
  until curl -sf localhost:8000/v1/models >/dev/null; do sleep 10; kill -0 $pid 2>/dev/null || { echo "$name server died"; tail -20 serve_$name.log; exit 1; }; done
  echo "== $name"
  python replay.py prompts --n 240 --concurrency 24 --temperature 0 --max-tokens 2048 --seed 7 --out data/ab_$name.jsonl 2>&1 | grep -E "^\[|^ "
  kill $pid; wait $pid 2>/dev/null
  while pgrep -f "^VLLM::EngineCore" >/dev/null; do sleep 5; done
done
