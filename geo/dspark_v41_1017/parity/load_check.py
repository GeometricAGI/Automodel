"""Load the released mtp.* weights into NeMo's native draft and check they equal a direct dequantization."""
import pathlib
import sys
import time

import torch
from safetensors import safe_open

from build import build

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from roundtrip import fp8_dequant  # noqa: E402

from paths import CKPT  # noqa: E402

ckpt = CKPT
t0 = time.time()
model = build(ckpt)
model.load_released_draft(ckpt)
print(f"loaded in {time.time() - t0:.0f} s")
sd = model.state_dict()
with safe_open(ckpt / "model-00044-of-00048.safetensors", "pt") as h:
    keys = set(h.keys())
    w = h.get_tensor("mtp.0.attn.wq_a.weight") if "mtp.0.attn.wq_a.weight" in keys else None
    if w is not None:
        ref = fp8_dequant(w, h.get_tensor("mtp.0.attn.wq_a.scale")).bfloat16()
        print("mtp.0.attn.wq_a equal to direct dequant:", torch.equal(sd["mtp.0.attn.wq_a.weight"], ref))
print("confidence head", sd["mtp.2.confidence_head.proj.weight"].dtype, sd["mtp.2.confidence_head.proj.weight"].abs().mean().item())
print("expert grouped", sd["mtp.1.ffn.experts.gate_and_up_projs"].shape, sd["mtp.1.ffn.experts.gate_and_up_projs"].float().abs().mean().item())
