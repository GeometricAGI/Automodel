#!/usr/bin/env bash
# Consolidate a DSpark fine-tune checkpoint's sharded draft into one BF16 safetensors file for export.py.
#   consolidate_draft.sh <checkpoint dir (…/checkpoints/epoch_E_step_S)> <out dir>
# Run from the AutoModel repo root in the training container. NeMo's offline tool takes its file layout from a
# reference index; the draft's tensors all go into one file, so the reference lists every saved name in file 1.
set -euo pipefail
CKPT=$1
OUT=$2
REF=$(mktemp -d)
python3 - "$CKPT/model" "$REF" <<'EOF'
import glob, json, struct, sys
src, ref = sys.argv[1], sys.argv[2]
keys = set()
for f in glob.glob(f"{src}/shard-*.safetensors"):
    with open(f, "rb") as fh:
        keys |= {k for k in json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0])) if k != "__metadata__"}
json.dump({"metadata": {}, "weight_map": {k: "model-00001-of-00001.safetensors" for k in sorted(keys)}},
          open(f"{ref}/model.safetensors.index.json", "w"))
json.dump({}, open(f"{ref}/config.json", "w"))
print(f"{len(keys)} tensors")
EOF
python3 tools/offline_hf_consolidation.py --backend gloo --num-threads 8 --model-name "$REF" \
  --input-dir "$CKPT/model" --output-dir "$OUT" --cast-dtype bf16
rm -rf "$REF"
ls -la "$OUT"
