"""Write a BF16 copy of the released DeepSeek-V4.1-Flash-1017 checkpoint for the frozen training target.

AutoModel dequantizes the FP8/FP4 checkpoint while loading it, holding each rank's quantized tensors and
conversion buffers on the GPU beside the BF16 model; on 8x B300 that runs out of memory. A BF16 checkpoint loads
straight into the model. Decoding follows DeepseekV41StateDictAdapter: FP8 with 32x32 block scales, packed FP4
(low nibble = even column) with per-row 32-wide scales, and Engram FP8 tables (".engram.embed.") with per-row
32-wide scales; products are formed in FP32 and stored as BF16. Unscaled tensors are copied unchanged, and
config.json drops its quantization_config.

  python dequantize_target.py <released checkpoint> <out dir> [--workers 12]
"""
import argparse
import json
import multiprocessing as mp
import pathlib
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP4_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
BLOCK = 32
ROWS_PER_CHUNK = 1 << 20


def dequantize(name: str, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    rows = weight.shape[0]
    out = torch.empty(rows, weight.shape[1] * (2 if weight.dtype == torch.int8 else 1), dtype=torch.bfloat16)
    for r0 in range(0, rows, ROWS_PER_CHUNK):
        w = weight[r0 : r0 + ROWS_PER_CHUNK]
        if weight.dtype == torch.int8:  # FP4 experts: per-row scales
            b = w.view(torch.uint8)
            x = FP4_TABLE[torch.stack([b & 0x0F, b >> 4], dim=-1).flatten(-2).long()]
            s = scale[r0 : r0 + ROWS_PER_CHUNK].float().repeat_interleave(BLOCK, 1)
        elif ".engram.embed." in name:  # FP8 Engram tables: per-row scales
            x = w.float()
            s = scale[r0 : r0 + ROWS_PER_CHUNK].float().repeat_interleave(BLOCK, 1)
        else:  # dense FP8: 32x32 block scales
            x = w.float()
            s = scale[r0 // BLOCK : (r0 + len(w) + BLOCK - 1) // BLOCK].float()
            s = s.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)[: len(w)]
        out[r0 : r0 + len(w)] = (x * s[:, : x.shape[1]]).to(torch.bfloat16)
    return out


def convert_shard(args):
    src, dst, shard = args
    if (dst / shard).exists():
        return shard, "exists"
    tensors = {}
    with safe_open(src / shard, "pt") as h:
        keys = list(h.keys())
        for k in keys:
            if k.endswith(".scale") and k[: -len(".scale")] + ".weight" in keys:
                continue
            t = h.get_tensor(k)
            scale_key = k[: -len(".weight")] + ".scale" if k.endswith(".weight") else None
            if scale_key in keys:
                t = dequantize(k, t, h.get_tensor(scale_key))
            elif t.dtype in (torch.int8, torch.float8_e4m3fn):
                raise ValueError(f"{shard}: {k} is quantized but its scale is in another shard")
            tensors[k] = t
    save_file(tensors, str(dst / f".{shard}.tmp"), metadata={"format": "pt"})
    (dst / f".{shard}.tmp").rename(dst / shard)  # a stopped run leaves no partial shard behind
    return shard, f"{len(tensors)} tensors"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=pathlib.Path)
    ap.add_argument("dst", type=pathlib.Path)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    a.dst.mkdir(parents=True, exist_ok=True)
    index = json.loads((a.src / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]
    for f in a.src.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name not in ("config.json", "model.safetensors.index.json"):
            shutil.copy2(f, a.dst / f.name)
    config = json.loads((a.src / "config.json").read_text())
    config.pop("quantization_config", None)
    config.get("text_config", {}).pop("quantization_config", None)
    (a.dst / "config.json").write_text(json.dumps(config, indent=2))
    index["weight_map"] = {k: v for k, v in wm.items() if not (k.endswith(".scale") and k[:-6] + ".weight" in wm)}
    (a.dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    shards = sorted(set(wm.values()))
    engram = sorted({wm[k] for k in wm if ".engram.embed." in k})
    jobs = [(a.src, a.dst, s) for s in shards if s not in engram]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for shard, note in pool.imap_unordered(convert_shard, jobs):
            print(shard, note, flush=True)
    for s in engram:  # ~200 GB each as BF16: one at a time
        print(*convert_shard((a.src, a.dst, s)), flush=True)
    print("done")


if __name__ == "__main__":
    main()
