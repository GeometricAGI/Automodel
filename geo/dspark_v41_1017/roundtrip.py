"""Export round-trip for DeepSeek-V4.1-Flash-1017's DSpark drafter (mtp.*): dequantize every quantized tensor to
BF16 (what a trainer sees), quantize it back with DeepSeek's rules (inference/convert.py, kernel.py), and check the
result dequantizes to the same values, survives a safetensors write and read with the original dtype tags, and, for
a perturbed copy (a stand-in for trained weights), stays within each format's rounding error.

  python roundtrip.py <checkpoint dir> <out dir>
"""
import json
import pathlib
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP4_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
FP4_MAX, FP8_MAX, BLOCK = 6.0, 448.0, 32


def pow2_scale(amax: torch.Tensor, qmax: float) -> torch.Tensor:
    """DeepSeek's fast_round_scale: 2^ceil(log2(amax / qmax)); an all-zero block keeps the smallest normal scale."""
    e = torch.ceil(torch.log2(torch.clamp(amax / qmax, min=torch.finfo(torch.float32).tiny)))
    return torch.exp2(e)


def fp4_dequant(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    b = packed.view(torch.uint8)
    codes = torch.stack([b & 0x0F, b >> 4], dim=-1).flatten(-2).long()  # low nibble = even element
    x = FP4_TABLE[codes]
    return (x.unflatten(-1, (-1, BLOCK)) * scale.float().unsqueeze(-1)).flatten(-2)


def fp4_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = x.float().unflatten(-1, (-1, BLOCK))
    s = pow2_scale(blocks.abs().amax(-1), FP4_MAX)
    q = torch.clamp(blocks / s.unsqueeze(-1), -FP4_MAX, FP4_MAX).flatten(-2)
    mags = FP4_TABLE[:8]
    # nearest magnitude, ties to the even code (as a round-to-nearest-even cast does)
    d = (q.abs().unsqueeze(-1) - mags).abs()
    idx = torch.argmin(d + (torch.arange(8) % 2) * 1e-6, dim=-1)
    code = idx + torch.signbit(q).long() * 8  # a negative value that rounds to zero is -0 (code 8), as DeepSeek writes
    packed = (code[..., 0::2] | (code[..., 1::2] << 4)).to(torch.uint8).view(torch.int8)
    return packed, s.to(torch.float8_e8m0fnu)


def fp8_dequant(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    s = scale.float().repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)[: w.shape[0], : w.shape[1]]
    return w.float() * s


def fp8_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n, k = x.shape
    pn, pk = -n % BLOCK, -k % BLOCK
    xp = torch.nn.functional.pad(x.float(), (0, pk, 0, pn))
    blocks = xp.unflatten(0, (-1, BLOCK)).unflatten(2, (-1, BLOCK))  # [bn, 32, bk, 32]
    s = pow2_scale(blocks.abs().amax(dim=(1, 3)), FP8_MAX)  # [bn, bk]
    q = (blocks / s[:, None, :, None]).to(torch.float8_e4m3fn).flatten(2, 3).flatten(0, 1)[:n, :k]
    return q, s.to(torch.float8_e8m0fnu)


def main(ckpt: pathlib.Path, out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    wm = json.load(open(ckpt / "model.safetensors.index.json"))["weight_map"]
    names = sorted(k for k in wm if k.startswith("mtp."))
    tensors = {}
    for f in sorted({wm[k] for k in names}):
        with safe_open(ckpt / f, "pt") as h:
            tensors.update({k: h.get_tensor(k) for k in h.keys() if k.startswith("mtp.")})
    stats = {"fp4": [0, 0, 0], "fp8": [0, 0, 0]}  # tensors, value mismatches, byte-differing tensors
    rebuilt, worst = {}, {"fp4": 0.0, "fp8": 0.0}
    for k in names:
        if not k.endswith(".weight") or k[: -len(".weight")] + ".scale" not in tensors:
            rebuilt.setdefault(k, tensors[k])
            continue
        base = k[: -len(".weight")]
        w, s = tensors[k], tensors[base + ".scale"]
        kind = "fp4" if w.dtype == torch.int8 else "fp8"
        deq = fp4_dequant(w, s) if kind == "fp4" else fp8_dequant(w, s)
        bf16 = deq.to(torch.bfloat16)
        assert torch.equal(bf16.float(), deq), f"{k}: dequantized values are not exact in bf16"
        q, qs = fp4_quant(bf16) if kind == "fp4" else fp8_quant(bf16)
        back = fp4_dequant(q, qs) if kind == "fp4" else fp8_dequant(q, qs)
        stats[kind][0] += 1
        stats[kind][1] += int(not torch.equal(back, deq))
        stats[kind][2] += int(not (torch.equal(q.view(torch.uint8), w.view(torch.uint8))
                                   and torch.equal(qs.view(torch.uint8), s.view(torch.uint8))))
        rebuilt[k], rebuilt[base + ".scale"] = q, qs
        # a trained weight is not on the grid: quantization error must stay within half a step of the block scale
        noisy = bf16.float() * (1 + 0.01 * torch.randn_like(bf16.float()))
        q2, qs2 = fp4_quant(noisy.bfloat16()) if kind == "fp4" else fp8_quant(noisy.bfloat16())
        err = ((fp4_dequant(q2, qs2) if kind == "fp4" else fp8_dequant(q2, qs2)) - noisy.bfloat16().float()).abs()
        step = (qs2.float().repeat_interleave(BLOCK, -1) if kind == "fp4" else
                qs2.float().repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)[: err.shape[0], : err.shape[1]])
        worst[kind] = max(worst[kind], float((err / step).max()))
    path = out / "mtp-roundtrip.safetensors"
    save_file(rebuilt, str(path))
    with open(path, "rb") as fh:
        hdr = json.loads(fh.read(int.from_bytes(fh.read(8), "little")))
    orig_types = {}
    for f in sorted({wm[k] for k in names}):
        with open(ckpt / f, "rb") as fh:
            h = json.loads(fh.read(int.from_bytes(fh.read(8), "little")))
        orig_types.update({k: v["dtype"] for k, v in h.items() if k.startswith("mtp.")})
    bad_types = [k for k in names if hdr[k]["dtype"] != orig_types[k] or hdr[k]["shape"] != list(tensors[k].shape)]
    for kind, (n, vals, byts) in stats.items():
        print(f"{kind}: {n} tensors; dequantized values differ in {vals}; bytes differ in {byts}; "
              f"perturbed copy: max error {worst[kind]:.3f} x block scale")
    print(f"written {len(hdr) - ('__metadata__' in hdr)} tensors; dtype or shape tag mismatches: {len(bad_types)} {bad_types[:3]}")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
