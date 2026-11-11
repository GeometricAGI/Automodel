"""Record the served INT3 model's output distribution over capture_decode.py's sequences.

Scores each captured sequence with vLLM's prompt_logprobs (one prefill, raw temperature-1 log-probabilities) and
keeps the top --top-k tokens for every position after the prompt: [positions, k] token ids and log-probabilities,
plus the probability mass they hold. dist_gap.py compares them with the BF16 target and the drafter.

  VLLM_ENABLE_V1_MULTIPROCESSING=0 python int3_logprobs.py <capture .pt> <out .pt> [--top-k 256]
"""
import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from paths import CKPT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=pathlib.Path)
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--top-k", type=int, default=256)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    records = torch.load(a.capture, weights_only=False)
    longest = max(len(r["input_ids"]) for r in records)
    llm = LLM(str(CKPT), max_model_len=longest + 64, gpu_memory_utilization=0.97, kv_cache_memory_bytes=4831838208,
              max_num_seqs=1, max_num_batched_tokens=max(8192, longest + 64), enforce_eager=True,
              enable_prefix_caching=False, engram_config={"use_thp": True}, safetensors_load_strategy="lazy",
              max_logprobs=a.top_k)
    out = []
    for i, rec in enumerate(records):
        ids = rec["input_ids"].tolist()
        res = llm.generate([TokensPrompt(prompt_token_ids=ids)],
                           SamplingParams(max_tokens=1, prompt_logprobs=a.top_k, temperature=1.0), use_tqdm=False)[0]
        # prompt_logprobs[j] is the distribution that predicted token j (None for j = 0).
        rows = res.prompt_logprobs[rec["prompt_len"] :]
        tok = torch.full((len(rows), a.top_k), -1, dtype=torch.long)
        lp = torch.full((len(rows), a.top_k), float("-inf"))
        for r, d in enumerate(rows):
            top = sorted(d.items(), key=lambda kv: kv[1].rank)[: a.top_k]
            tok[r, : len(top)] = torch.tensor([t for t, _ in top])
            lp[r, : len(top)] = torch.tensor([v.logprob for _, v in top])
        out.append({"positions": torch.arange(rec["prompt_len"], len(ids)), "token_ids": tok, "logprobs": lp})
        print(f"{i}: {len(rows)} positions, top-{a.top_k} mass mean {lp.exp().sum(-1).mean():.4f}", flush=True)
    torch.save(out, a.out)


if __name__ == "__main__":
    main()
