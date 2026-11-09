#!/usr/bin/env bash
# The prefill/decode feature comparison with the Lattice INT3 kernels wired (production) and without them.
# Run from the vLLM environment.
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 GEOQUANT_INT3_TKDSL=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
for v in lattice geoquant-only; do
  echo "== $v"
  if [ $v = geoquant-only ]; then export VLLM_PLUGINS=geoquant_v41_mixed; fi
  python capture_paths.py "$1" paths_$v.pt > paths_$v.log 2>&1
  grep -E "^generation|^prefill|^prompt|^decode|^per-layer|Error" paths_$v.log
done
