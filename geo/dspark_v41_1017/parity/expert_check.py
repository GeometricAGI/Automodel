"""Compare NeMo's dequantized grouped draft experts with a direct MXFP4 dequantization of the checkpoint."""
import json
import pathlib
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from roundtrip import fp4_dequant  # noqa: E402

from nemo_automodel.components.models.deepseek_v41.state_dict_adapter import DeepseekV41DSparkStateDictAdapter  # noqa: E402
from build import build  # noqa: E402

from paths import CKPT  # noqa: E402

ckpt = CKPT
wm = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
with torch.device("meta"):
    model = build(ckpt)
names = [f"mtp.0.ffn.experts.{e}.{w}.{s}" for e in range(128) for w in ("w1", "w2", "w3") for s in ("weight", "scale")]
names += ["mtp.0.ffn.gate.weight", "mtp.0.ffn.gate.bias"] + [k for k in wm if k.startswith("mtp.0.ffn.") and "experts." not in k]
raw = {}
for n in sorted(set(names)):
    if n in wm:
        with safe_open(ckpt / wm[n], "pt") as h:
            raw[n] = h.get_tensor(n)
print({k: (tuple(v.shape), str(v.dtype)) for k, v in raw.items() if "experts.0." in k or "experts." not in k})
ref = {k: v.clone() for k, v in raw.items()}
native = model.state_dict_adapter.from_hf(dict(raw))
gu = native["mtp.0.ffn.experts.gate_and_up_projs"]
dn = native["mtp.0.ffn.experts.down_projs"]
print("native", gu.shape, dn.shape)
w1 = fp4_dequant(ref["mtp.0.ffn.experts.0.w1.weight"], ref["mtp.0.ffn.experts.0.w1.scale"])
w3 = fp4_dequant(ref["mtp.0.ffn.experts.0.w3.weight"], ref["mtp.0.ffn.experts.0.w3.scale"])
w2 = fp4_dequant(ref["mtp.0.ffn.experts.0.w2.weight"], ref["mtp.0.ffn.experts.0.w2.scale"])
inter = w1.shape[0]
g = gu[0].float()
print("w1 shape", tuple(w1.shape), "| gate == w1.T:", torch.equal(g[:, :inter], w1.T.bfloat16().float()),
      "| up == w3.T:", torch.equal(g[:, inter:], w3.T.bfloat16().float()),
      "| interleaved?", torch.equal(g[:, 0::2], w1.T.bfloat16().float()))
print("down == w2.T:", torch.equal(dn[0].float(), w2.T.bfloat16().float()))
print("max |gate - w1.T|", (g[:, :inter] - w1.T).abs().max().item(), "w1 absmax", w1.abs().max().item())
print([k for k in native if "experts" not in k])
