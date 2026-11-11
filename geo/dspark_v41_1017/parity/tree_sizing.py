"""How many tokens per step could tree verification of DSpark's drafts accept at the serving temperature?

Inputs are a temperature-1 capture_decode.py run (vLLM's INT3 features and its sampled tokens) and
bf16_features.py's final-norm states for the same sequences (the target's output distribution). The generated tokens
are samples from the target, so with teacher forcing:

- chain, probabilistic drafting (serving today): position k is accepted with probability sum(min(p_k, q_k)); the
  expected tokens per step are 1 + sum over k of the product of those up to k.
- tree of draft candidates, verified by sampling the target and walking the tree (lossless): the step accepts as
  many positions as the true tokens' rank path stays inside the tree. DSpark drafts every candidate path from one
  backbone pass; only the Markov head's previous-token term differs per branch, and teacher forcing scores the true
  path's. For a budget of N tree nodes the best tree holds the N most frequent rank-path prefixes; it is chosen on
  half the sequences and scored on the other half (and the reverse).

  python tree_sizing.py <capture .pt> <bf16 .last.pt> <drafter checkpoint>... [--budgets 3 5 8 16 32 64]
"""
import argparse
import collections
import pathlib

import torch
from nemo_accept import load_model

MAX_RANK = 16  # deeper ranks never make a useful tree


def block_stats(model, records, last, bf16_feats=None):
    """Per kept draft block: the true tokens' ranks under the draft [blocks, 5] and sum(min(p, q)) [blocks, 5]."""
    ranks, overlap, seq_of = [], [], []
    torch.manual_seed(0)
    for i, rec in enumerate(records):
        ids = rec["input_ids"].cuda()[None]
        mask = torch.zeros_like(ids)
        mask[:, rec["prompt_len"] :] = 1
        feats = rec["features"] if bf16_feats is None else bf16_feats[i]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(ids, feats.cuda()[None], mask, target_last_hidden_states=last[i].cuda()[None])
        keep = out.block_keep_mask.bool() & out.eval_mask.bool().all(-1)
        draft = out.draft_logits[keep].float()  # [blocks, 5, vocab]
        target = out.aligned_target_logits[keep].float()
        true = out.target_ids[keep]
        ranks.append((draft > draft.gather(-1, true[..., None])).sum(-1).clamp(max=MAX_RANK).cpu())
        overlap.append(torch.minimum(draft.softmax(-1), target.softmax(-1)).sum(-1).cpu())
        seq_of.append(torch.full((len(true),), i))
    return torch.cat(ranks), torch.cat(overlap), torch.cat(seq_of)


def prefix_counts(ranks):
    counts = collections.Counter()
    for row in ranks.tolist():
        for k in range(len(row)):
            if row[k] >= MAX_RANK:
                break
            counts[tuple(row[: k + 1])] += 1
    return counts


def tree_tokens(fit, score, budget):
    """Expected tokens per step for the budget-node tree chosen on `fit`, scored on `score`."""
    # A prefix is never rarer than its extensions, so the top-N by (count, shorter first) is prefix-closed.
    tree = {p for p, _ in sorted(prefix_counts(fit).items(), key=lambda x: (-x[1], len(x[0])))[:budget]}
    counts = prefix_counts(score)
    return 1 + sum(c for p, c in counts.items() if p in tree) / len(score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=pathlib.Path)
    ap.add_argument("last", type=pathlib.Path)
    ap.add_argument("drafters", type=pathlib.Path, nargs="+")
    ap.add_argument("--budgets", type=int, nargs="+", default=[3, 5, 8, 16, 32, 64])
    a = ap.parse_args()
    records = torch.load(a.capture, weights_only=False)
    last = torch.load(a.last, weights_only=False)
    n_gen = sum(len(r["input_ids"]) - r["prompt_len"] for r in records)
    print(f"{len(records)} sequences, {n_gen} sampled tokens")
    for ckpt in a.drafters:
        model = load_model(ckpt, anchors=512)
        ranks, overlap, seq = block_stats(model, records, last)
        print(f"== {ckpt.name}: {len(ranks)} draft blocks")
        for d in (3, 5):
            chain = 1 + overlap[:, :d].cumprod(-1).sum(-1).mean()
            greedy = 1 + (ranks[:, :d] == 0).int().cumprod(-1).sum(-1).float().mean()
            print(f"  chain of {d}: probabilistic drafting {chain:.3f} tokens per step (acceptance by position "
                  f"{[round(x, 3) for x in overlap[:, :d].cumprod(-1).mean(0).tolist()]}); top-1 drafts {greedy:.3f}")
        print(f"  share of true tokens at draft rank 0/1/2/3/4-15/16+, by position:")
        for k in range(ranks.shape[1]):
            r = ranks[:, k]
            share = [(r == 0), (r == 1), (r == 2), (r == 3), (r >= 4) & (r < MAX_RANK), r >= MAX_RANK]
            print(f"    {k + 1}: {[round(s.float().mean().item(), 3) for s in share]}")
        even = seq % 2 == 0
        for b in a.budgets:
            t = (tree_tokens(ranks[even], ranks[~even], b) * (~even).sum()
                 + tree_tokens(ranks[~even], ranks[even], b) * even.sum()) / len(ranks)
            print(f"  best tree of {b} nodes: {t:.3f} tokens per step")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
