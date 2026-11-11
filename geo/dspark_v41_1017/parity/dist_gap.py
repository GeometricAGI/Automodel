"""Is the drafter matched to the BF16 target's output distribution rather than the served INT3 model's?

At temperature 1, a probabilistic draft at a position is accepted with probability sum(min(p, q)) for the
verifying model's distribution p and the draft's q. For a temperature-1 capture_decode.py run, this compares, per
draft position: q against the BF16 target (bf16_features.py's final-norm states, what training matches), q against
the served INT3 model (int3_logprobs.py, top-k: a lower bound missing at most the tail mass), and the two targets.

  python dist_gap.py <capture .pt> <bf16 .last.pt> <int3 logprobs .pt> <drafter checkpoint>...
"""
import argparse
import pathlib

import torch

import nemo_automodel.components.models.deepseek_v41.dspark as nemo_dspark
from nemo_accept import load_model

_anchors = []
_sample = nemo_dspark.sample_anchor_positions


def _recording_sample(*args, **kwargs):
    anchors, keep = _sample(*args, **kwargs)
    _anchors.append(anchors)
    return anchors, keep


nemo_dspark.sample_anchor_positions = _recording_sample


def sparse_overlap(dense, tok, lp):
    """sum over the sparse distribution's support of min(dense, sparse); dense [..., V], tok/lp [..., K]."""
    valid = tok >= 0
    d = dense.gather(-1, tok.clamp(min=0))
    return (torch.minimum(d, lp.exp()) * valid).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=pathlib.Path)
    ap.add_argument("last", type=pathlib.Path)
    ap.add_argument("int3", type=pathlib.Path)
    ap.add_argument("drafters", type=pathlib.Path, nargs="+")
    a = ap.parse_args()
    records = torch.load(a.capture, weights_only=False)
    last = torch.load(a.last, weights_only=False)
    int3 = torch.load(a.int3, weights_only=False)
    for ckpt in a.drafters:
        model = load_model(ckpt, anchors=512)
        sums = {k: [] for k in ("q_bf16", "q_int3", "bf16_int3", "int3_mass")}
        torch.manual_seed(0)
        for rec, h, d in zip(records, last, int3):
            ids = rec["input_ids"].cuda()[None]
            mask = torch.zeros_like(ids)
            mask[:, rec["prompt_len"] :] = 1
            _anchors.clear()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(ids, rec["features"].cuda()[None], mask, target_last_hidden_states=h.cuda()[None])
            keep = out.block_keep_mask.bool() & out.eval_mask.bool().all(-1)
            block = out.draft_logits.shape[2]
            pos = (_anchors[0][..., None] + torch.arange(1, block + 1, device=ids.device))[keep]  # [blocks, 5]
            row = pos - rec["prompt_len"]
            assert (row >= 0).all() and (row < len(d["token_ids"])).all()
            q = out.draft_logits[keep].float().softmax(-1)
            pb = out.aligned_target_logits[keep].float().softmax(-1)
            tok, lp = d["token_ids"].cuda()[row], d["logprobs"].cuda()[row]
            sums["q_bf16"].append(torch.minimum(q, pb).sum(-1).cpu())
            sums["q_int3"].append(sparse_overlap(q, tok, lp).cpu())
            sums["bf16_int3"].append(sparse_overlap(pb, tok, lp).cpu())
            sums["int3_mass"].append(lp.exp().sum(-1).cpu())
        s = {k: torch.cat(v) for k, v in sums.items()}
        print(f"== {ckpt.name}: {len(s['q_bf16'])} draft blocks; INT3 top-k mass {s['int3_mass'].mean():.4f}")
        for name, label in (("q_bf16", "draft vs BF16 target (training)"), ("q_int3", "draft vs INT3 (serving)"),
                            ("bf16_int3", "BF16 target vs INT3")):
            x = s[name]
            chain3 = 1 + x[:, :3].cumprod(-1).sum(-1).mean()
            chain5 = 1 + x.cumprod(-1).sum(-1).mean()
            print(f"  {label}: overlap by position {[round(v, 3) for v in x.mean(0).tolist()]}; "
                  f"chain acceptance {[round(v, 3) for v in x.cumprod(-1).mean(0).tolist()]}; "
                  f"tokens per step {chain3:.3f} (3 drafts), {chain5:.3f} (5)")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
