"""Write a trained DSpark drafter into a servable copy of DeepSeek-V4.1-Flash-1017-vLLM-B300.

The input is the drafter in BF16/FP32 under released names (`mtp.*`, one tensor per expert), as NeMo AutoModel's
DeepseekV41DSparkStateDictAdapter.to_hf exports it. Every weight the checkpoint stores quantized is quantized back to
the checkpoint's format (MXFP4 experts, FP8 block-32 elsewhere, power-of-two E8M0 scales, as roundtrip.py checks); the
rest is cast to the checkpoint's dtype. The output directory links every other file of the checkpoint and holds new
drafter shards with the original names and tensor grouping, so the index is unchanged.

  python export.py <checkpoint dir> <drafter .safetensors file or dir> <out dir>
  python export.py <checkpoint dir> --dequantized-self <out dir>   # test: re-export the checkpoint's own drafter
"""
import json
import os
import pathlib
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from roundtrip import fp4_dequant, fp4_quant, fp8_dequant, fp8_quant


def load_dir(src: pathlib.Path) -> dict[str, torch.Tensor]:
    files = [src] if src.is_file() else sorted(src.glob("*.safetensors"))
    out = {}
    for f in files:
        with safe_open(f, "pt") as h:
            out.update({k: h.get_tensor(k) for k in h.keys() if k.startswith("mtp.")})
    return out


def main(ckpt: pathlib.Path, src: str, out: pathlib.Path) -> None:
    wm = json.load(open(ckpt / "model.safetensors.index.json"))["weight_map"]
    shards = sorted({v for k, v in wm.items() if k.startswith("mtp.")})
    if any(not k.startswith("mtp.") for k, v in wm.items() if v in shards):
        raise SystemExit("drafter shards also hold target tensors; this script only replaces whole drafter shards")
    orig = {}
    for f in shards:
        with safe_open(ckpt / f, "pt") as h:
            orig.update({k: h.get_tensor(k) for k in h.keys()})
    quantized = {k for k in orig if k.endswith(".weight") and k.removesuffix(".weight") + ".scale" in orig}
    if src == "--dequantized-self":
        trained = {}
        for k, v in orig.items():
            if k in quantized:
                s = orig[k.removesuffix(".weight") + ".scale"]
                trained[k] = (fp4_dequant(v, s) if v.dtype == torch.int8 else fp8_dequant(v, s)).bfloat16()
            elif not k.endswith(".scale") or k.removesuffix(".scale") + ".weight" not in quantized:
                trained[k] = v.float() if v.is_floating_point() else v
    else:
        trained = load_dir(pathlib.Path(src))
    expected = {k for k in orig if not (k.endswith(".scale") and k.removesuffix(".scale") + ".weight" in quantized)}
    missing, extra = expected - trained.keys(), trained.keys() - expected
    if missing or extra:
        raise SystemExit(f"drafter names differ from the checkpoint: missing {sorted(missing)[:5]} "
                         f"({len(missing)}), unexpected {sorted(extra)[:5]} ({len(extra)})")

    new = {}
    for k in expected:
        w, ref = trained[k], orig[k]
        if tuple(w.shape) != tuple(ref.shape) and k not in quantized:
            raise SystemExit(f"{k}: shape {tuple(w.shape)} != checkpoint {tuple(ref.shape)}")
        if k in quantized:
            base = k.removesuffix(".weight")
            q, s = fp4_quant(w) if ref.dtype == torch.int8 else fp8_quant(w)
            if q.shape != ref.shape or s.shape != orig[base + ".scale"].shape:
                raise SystemExit(f"{k}: quantized shape {tuple(q.shape)} != checkpoint {tuple(ref.shape)}")
            new[k], new[base + ".scale"] = q, s.to(orig[base + ".scale"].dtype)
        else:
            new[k] = w.to(ref.dtype)

    out.mkdir(parents=True, exist_ok=True)
    for entry in ckpt.iterdir():
        dst = out / entry.name
        if entry.name not in shards and not dst.exists():
            os.symlink(entry.resolve(), dst)
    for f in shards:
        save_file({k: new[k].contiguous() for k in orig if wm[k] == f}, str(out / f), metadata={"format": "pt"})
    raw = lambda t: t.contiguous().view(torch.uint8)
    changed = sum(new[k].dtype != orig[k].dtype or not torch.equal(raw(new[k]), raw(orig[k])) for k in orig)
    print(f"wrote {len(new)} drafter tensors into {len(shards)} shards at {out}; {changed} differ in bytes from the checkpoint")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3]))
