"""Run NeMo AutoModel's native V4.1 DSpark draft, loaded with the released weights, on target features captured
from vLLM, and measure greedy acceptance against the tokens the engine generated.

Teacher forcing is exact for acceptance: draft position k only matters if positions 1..k-1 were accepted, in
which case its Markov input (the previous token) is the generated token the trainer feeds.

  python nemo_accept.py <capture .pt> [--drafts 3]
"""
import argparse
import json
import pathlib
import sys

import torch
from safetensors import safe_open

from build import build

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from paths import CKPT  # noqa: E402



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=pathlib.Path)
    ap.add_argument("--drafts", type=int, default=3)
    ap.add_argument("--anchors", type=int, default=512)
    ap.add_argument("--zero-features", action="store_true", help="diagnostic: drop the target features")
    ap.add_argument("--shifts", type=int, nargs="+", default=[0],
                    help="diagnostic: move features this many positions later (+) or earlier (-)")
    a = ap.parse_args()
    torch.manual_seed(0)
    model = build(CKPT, num_anchors=a.anchors)
    model.load_released_draft(CKPT)

    wm = json.loads((CKPT / "model.safetensors.index.json").read_text())["weight_map"]
    for name, module in (("embed.weight", model.embed_tokens), ("head.weight", model.lm_head)):
        with safe_open(CKPT / wm[name], "pt") as h:
            module.weight.data.copy_(h.get_tensor(name))
    model = model.cuda().eval()
    seqs = torch.load(a.capture)
    global ZERO
    ZERO = a.zero_features
    for shift in a.shifts:
        measure(model, seqs, a.drafts, shift)


ZERO = False


def measure(model, seqs, drafts, shift):
    torch.manual_seed(0)
    total_steps = total_tokens = 0
    position_hits = torch.zeros(drafts)
    for seq in seqs:
        ids = seq["input_ids"].cuda()[None]
        mask = torch.zeros_like(ids)
        mask[:, seq["prompt_len"]:] = 1
        feats = torch.roll(seq["features"], shift, 0).cuda()[None]
        if ZERO:
            feats = torch.zeros_like(feats)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(ids, feats, mask)
        pred = out.draft_logits.argmax(-1)[..., : drafts]
        hit = (pred == out.target_ids[..., : drafts]) & out.eval_mask[..., : drafts].bool()
        keep = out.block_keep_mask.bool() & out.eval_mask[..., : drafts].bool().all(-1)
        accepted = hit.int().cumprod(-1).sum(-1)[keep]
        total_steps += accepted.numel()
        total_tokens += (accepted + 1).sum().item()
        position_hits += hit[keep].float().sum(0).cpu()
    print(f"shift {shift:+d}: blocks {total_steps}; tokens per step {total_tokens / total_steps:.2f} (1 + accepted of {drafts}); "
          f"per-position top-1 accuracy {[round(x, 3) for x in (position_hits / total_steps).tolist()]}")


if __name__ == "__main__":
    main()
