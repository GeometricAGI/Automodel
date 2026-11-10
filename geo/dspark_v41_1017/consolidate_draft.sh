#!/usr/bin/env bash
# Consolidate a DSpark fine-tune checkpoint's sharded draft into one BF16 safetensors file for export.py.
#   consolidate_draft.sh <checkpoint dir (…/checkpoints/epoch_E_step_S)> <out dir>
# Run from the AutoModel repo root in the training container. NeMo's offline tool writes each tensor to the file
# its checkpoint's .hf_metadata mapping names, but the trainer's mapping covers only 80 of the draft's ~1,200
# tensors; this rewrites it (keeping the original as .orig) to put every saved tensor in one file.
set -euo pipefail
CKPT=$1
OUT=$2
python3 - "$CKPT/model" <<'EOF'
import glob, json, os, struct, sys
model = sys.argv[1]
keys = set()
for f in glob.glob(f"{model}/shard-*.safetensors"):
    with open(f, "rb") as fh:
        keys |= {k for k in json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0])) if k != "__metadata__"}
path = f"{model}/.hf_metadata/fqn_to_file_index_mapping.json"
if not os.path.exists(path + ".orig"):
    os.rename(path, path + ".orig")
json.dump({k: 1 for k in sorted(keys)}, open(path, "w"))
print(f"{len(keys)} tensors -> one file")
EOF
python3 tools/offline_hf_consolidation.py --backend gloo --num-threads 8 \
  --model-name /data/models/DeepSeek-V4.1-Flash-1017-bf16 --input-dir "$CKPT/model" --output-dir "$OUT" --cast-dtype bf16
ls -la "$OUT"
