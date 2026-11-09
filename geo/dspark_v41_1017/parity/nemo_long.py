"""Compare the drafter's target features under full and truncated contexts (long_context.py captures).

For each truncated arm: how long its greedy output agrees with the full context's, the cosine of its features
to the full context's at the same positions (the last 128 prompt tokens, which both share, and the agreed
generated tokens), and NeMo's released-draft acceptance on those tokens with either set of features.

  python nemo_long.py <long_context .pt>
"""
import argparse
import collections
import pathlib

import torch
import torch.nn.functional as F
from nemo_accept import load_model

from build import CKPT


DRAFTS = 3


def acceptance(model, ids, feats, mask):
    drafts = DRAFTS
    torch.manual_seed(0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(ids.cuda()[None], feats.cuda()[None], mask.cuda()[None])
    pred = out.draft_logits.argmax(-1)[..., :drafts]
    valid = out.eval_mask[..., :drafts].bool()
    hit = (pred == out.target_ids[..., :drafts]) & valid
    keep = out.block_keep_mask.bool() & valid.all(-1)
    accepted = hit.int().cumprod(-1).sum(-1)[keep]
    return accepted.numel(), int((accepted + 1).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=pathlib.Path)
    ap.add_argument("--drafts", type=int, default=3, help="draft positions scored (the drafter makes 5)")
    a = ap.parse_args()
    global DRAFTS
    DRAFTS = a.drafts
    model = load_model(CKPT, anchors=512)
    records = torch.load(a.capture)
    agg = collections.defaultdict(lambda: collections.defaultdict(float))
    for rec in records:
        full = rec["full"]
        g = len(full["generated"])
        mask = torch.zeros(len(full["input_ids"]), dtype=torch.long)
        mask[128:-1] = 1
        b, t = acceptance(model, full["input_ids"], full["features"], mask)
        agg["full"]["blocks"] += b
        agg["full"]["tokens"] += t
        for name in [k for k in rec if k.startswith("last")]:
            arm = rec[name]
            n = next((i for i, (x, y) in enumerate(zip(full["generated"], arm["generated"])) if x != y),
                     min(g, len(arm["generated"])))
            s = agg[name]
            s["samples"] += 1
            s["agree"] += n
            s["agree_all"] += n == min(g, len(arm["generated"]))
            both = full["valid"][: 128 + n] & arm["valid"][: 128 + n]
            cos = F.cosine_similarity(full["features"][: 128 + n][both].float(),
                                      arm["features"][: 128 + n][both].float(), dim=-1)
            s["cos_sum"] += cos.sum().item()
            s["cos_n"] += cos.numel()
            s["cos_min"] = min(s.get("cos_min", 1.0), cos.min().item())
            if n > 8:
                # The same tokens, with full-context or truncated-context features.
                m = torch.zeros(len(full["input_ids"]), dtype=torch.long)
                m[128 : 128 + n - 1] = 1
                rows = len(full["features"])
                trunc = arm["features"][:rows]
                if len(trunc) < rows:
                    trunc = torch.cat([trunc, trunc.new_zeros(rows - len(trunc), trunc.shape[1])])
                for which, f in (("on_full", full["features"]), ("on_trunc", trunc)):
                    b, t = acceptance(model, full["input_ids"], f, m)
                    s[f"{which}_blocks"] += b
                    s[f"{which}_tokens"] += t
            # The truncated arm on its own continuation, as training data from it would be.
            m = torch.zeros(len(arm["input_ids"]), dtype=torch.long)
            m[128:-1] = 1
            b, t = acceptance(model, arm["input_ids"], arm["features"], m)
            s["own_blocks"] += b
            s["own_tokens"] += t
    print(f"full context: {len(records)} turns, tokens per step {agg['full']['tokens'] / agg['full']['blocks']:.2f}")
    for name, s in agg.items():
        if name == "full":
            continue
        print(f"{name}: {int(s['samples'])} turns; greedy output agrees with full for {s['agree'] / s['samples']:.0f} "
              f"tokens on average ({int(s['agree_all'])} fully); feature cosine on shared positions mean "
              f"{s['cos_sum'] / s['cos_n']:.4f}, min {s['cos_min']:.3f}; acceptance on the agreed tokens "
              f"{s['on_full_tokens'] / max(1, s['on_full_blocks']):.2f} with full features vs "
              f"{s['on_trunc_tokens'] / max(1, s['on_trunc_blocks']):.2f} with truncated; on its own output "
              f"{s['own_tokens'] / s['own_blocks']:.2f}")


if __name__ == "__main__":
    main()
