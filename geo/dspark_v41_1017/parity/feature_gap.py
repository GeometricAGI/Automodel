"""How much does training on BF16 target features, when serving computes them with the INT3 model, cost?

For capture_decode.py's sequences (INT3 features from vLLM decode steps) and bf16_features.py's recomputation of
the same positions with the training target: the cosine between the two, and each drafter's greedy acceptance on
the generated tokens with either set. BF16 features are zeroed wherever vLLM had none (prompt positions before
its 128-token window), so the two sets differ only in values.

  python feature_gap.py <capture .pt> <bf16 features .pt> <drafter checkpoint>... [--drafts 5]
"""
import argparse
import pathlib

import torch
import torch.nn.functional as F
from nemo_accept import load_model


def acceptance(model, records, features, drafts):
    torch.manual_seed(0)
    blocks = tokens = 0
    hits = torch.zeros(drafts)
    for rec, feats in zip(records, features):
        ids = rec["input_ids"].cuda()[None]
        mask = torch.zeros_like(ids)
        mask[:, rec["prompt_len"] :] = 1
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(ids, feats.cuda()[None], mask)
        pred = out.draft_logits.argmax(-1)[..., :drafts]
        hit = (pred == out.target_ids[..., :drafts]) & out.eval_mask[..., :drafts].bool()
        keep = out.block_keep_mask.bool() & out.eval_mask[..., :drafts].bool().all(-1)
        accepted = hit.int().cumprod(-1).sum(-1)[keep]
        blocks += accepted.numel()
        tokens += (accepted + 1).sum().item()
        hits += hit[keep].float().sum(0).cpu()
    return tokens / blocks, [round(x, 3) for x in (hits / blocks).tolist()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=pathlib.Path)
    ap.add_argument("bf16", type=pathlib.Path)
    ap.add_argument("drafters", type=pathlib.Path, nargs="+")
    ap.add_argument("--drafts", type=int, default=5)
    a = ap.parse_args()
    records = torch.load(a.capture, weights_only=False)
    bf16 = torch.load(a.bf16, weights_only=False)
    assert len(bf16) == len(records)
    int3 = [r["features"] for r in records]
    bf16 = [f * r["valid"][:, None].to(f.dtype) for f, r in zip(bf16, records)]

    for part in ("prompt window", "generated"):
        cos = []
        for r, x, y in zip(records, int3, bf16):
            sel = r["valid"].clone()
            if part == "generated":
                sel[: r["prompt_len"]] = False
            else:
                sel[r["prompt_len"] :] = False
            cos.append(F.cosine_similarity(x[sel].float(), y[sel].float(), dim=-1))
        cos = torch.cat(cos)
        q = torch.quantile(cos, torch.tensor([0.01, 0.1, 0.5]))
        print(f"cosine INT3 vs BF16, {part}: {cos.numel()} positions, mean {cos.mean():.4f}, "
              f"median {q[2]:.4f}, p10 {q[1]:.4f}, p1 {q[0]:.4f}")

    for ckpt in a.drafters:
        model = load_model(ckpt, anchors=512)
        for name, feats in (("INT3 (serving)", int3), ("BF16 (training)", bf16)):
            tps, pos = acceptance(model, records, feats, a.drafts)
            for k in sorted({3, a.drafts}):
                tps_k, _ = acceptance(model, records, feats, k) if k != a.drafts else (tps, pos)
                print(f"{ckpt.name}, {name} features, {k} drafts: tokens per step {tps_k:.3f}")
            print(f"{ckpt.name}, {name} features: per-position top-1 accuracy {pos}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
