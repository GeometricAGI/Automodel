"""Does training on a left-truncated context give the drafter the features a full context would?

For requesty agent-session turns, render the request as the server would (tools in the system message, thinking,
effort "high"), then make truncated versions that keep the system message and the most recent messages within a
token budget. Generate greedily from each version with DSpark on, recording the drafter's target features from
decode steps (as serving computes them) for the generated tokens plus the 128 prompt positions before them.

  VLLM_ENABLE_V1_MULTIPROCESSING=0 python long_context.py <requesty parquet> <out .pt> [--n 20] [--budgets 16384 32768 65536]
"""
import argparse
import json
import pathlib
import random
import sys

import pyarrow.parquet as pq
import torch

from vllm.models.deepseek_v41.nvidia import dspark as vllm_dspark
from vllm.models.deepseek_v41.nvidia import model as vllm_model
from vllm.tokenizers.deepseek_v41 import _normalize_messages
from vllm.tokenizers.deepseek_v41_encoding import encode_messages

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from paths import CKPT  # noqa: E402

WINDOW = 128
steps: list[dict] = []
_combine = vllm_dspark.DSparkDeepseekV4Model.combine_hidden_states
_forward = vllm_model.DeepseekV4Model.forward


def _recording_forward(self, input_ids, positions, *args, **kwargs):
    # Prefill chunks are long; only their last WINDOW rows can hold features, so keep just those.
    steps.append({"ids": input_ids[-WINDOW:].detach().cpu(), "pos": positions[-WINDOW:].detach().cpu()})
    return _forward(self, input_ids, positions, *args, **kwargs)


def _recording_combine(self, aux_hidden_states):
    steps[-1]["feats"] = aux_hidden_states[-WINDOW:].detach().to("cpu", torch.bfloat16)
    return _combine(self, aux_hidden_states)


vllm_model.DeepseekV4Model.forward = _recording_forward
vllm_dspark.DSparkDeepseekV4Model.combine_hidden_states = _recording_combine


def render(messages, tools):
    conv = _normalize_messages(messages)
    system = next((m for m in conv if m["role"] == "system"), None)
    if system is None:
        system = {"role": "system", "content": ""}
        conv.insert(0, system)
    system["tools"] = tools
    return encode_messages(conv, thinking_mode="thinking", drop_thinking=True, reasoning_effort="high")


def truncate(messages, tools, budget, tok):
    """Keep the system message and the longest suffix of the rest that fits the budget, starting at a user or
    assistant message so no tool result loses its call. None if even the last message does not fit."""
    system, rest = messages[:1], messages[1:]
    starts = [i for i, m in enumerate(rest) if m["role"] in ("user", "assistant")]
    lo, hi, best = 0, len(starts) - 1, None
    while lo <= hi:  # the smallest start index whose suffix fits
        mid = (lo + hi) // 2
        n = len(tok.encode(render(system + rest[starts[mid]:], tools), add_special_tokens=False))
        if n <= budget:
            best, hi = starts[mid], mid - 1
        else:
            lo = mid + 1
    return None if best is None else system + rest[best:]


def capture(llm, prompt, max_tokens):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    steps.clear()
    gen = llm.generate([TokensPrompt(prompt_token_ids=prompt)], SamplingParams(max_tokens=max_tokens, temperature=0),
                       use_tqdm=False)[0].outputs[0].token_ids
    seq = prompt + list(gen)
    start = len(prompt) - WINDOW
    feats = torch.zeros(len(seq) - start, steps[0]["feats"].shape[1], dtype=torch.bfloat16)
    valid = torch.zeros(len(seq) - start, dtype=torch.bool)
    for s in steps:
        if "feats" not in s:
            continue
        rows = list(zip(s["pos"].tolist(), s["ids"].tolist()))
        ok = True
        for r, (p, t) in enumerate(rows):
            ok = ok and p < len(seq) and seq[p] == t
            if ok and p >= start and s["feats"][r].abs().sum() > 0:
                feats[p - start], valid[p - start] = s["feats"][r], True
    return {"input_ids": torch.tensor(seq[start:]), "prompt_len": WINDOW, "full_prompt_len": len(prompt),
            "generated": list(gen), "features": feats, "valid": valid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", type=pathlib.Path)
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--min-prompt", type=int, default=40000)
    ap.add_argument("--max-prompt", type=int, default=160000)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--budgets", type=int, nargs="+", default=[16384, 32768, 65536])
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM

    tok = AutoTokenizer.from_pretrained(CKPT)
    rng = random.Random(0)
    rows = pq.read_table(a.parquet).to_pylist()
    rng.shuffle(rows)
    samples = []
    for r in rows:
        turns = [i for i, n in enumerate(r["estimated_prompts"]) if a.min_prompt <= n <= a.max_prompt]
        if turns:
            k = rng.choice(turns)
            samples.append((r["id"], k, json.loads(r["messages"])[: r["turns"][k]], json.loads(r["tools"])))
        if len(samples) == a.n:
            break

    llm = LLM(str(CKPT), max_model_len=a.max_prompt + 16384, gpu_memory_utilization=0.97,
              kv_cache_memory_bytes=4831838208, max_num_seqs=1, max_num_batched_tokens=16384, enforce_eager=True,
              engram_config={"use_thp": True}, safetensors_load_strategy="lazy",
              speculative_config={"method": "dspark", "num_speculative_tokens": 3})
    out = []
    for sid, k, messages, tools in samples:
        arms = {"full": messages}
        for b in a.budgets:
            t = truncate(messages, tools, b, tok)
            if t is not None and len(t) < len(messages):
                arms[f"last{b // 1024}k"] = t
        record = {"session": sid, "turn": k}
        for name, msgs in arms.items():
            prompt = tok.encode(render(msgs, tools), add_special_tokens=False)
            record[name] = capture(llm, prompt, a.max_tokens)
        full = record["full"]["generated"]
        agree = {n: next((i for i, (x, y) in enumerate(zip(full, record[n]["generated"])) if x != y),
                         min(len(full), len(record[n]["generated"]))) for n in arms if n != "full"}
        print(f"{sid} turn {k}: prompt {record['full']['full_prompt_len']}, generated {len(full)}; "
              f"truncated arms agree with full for {agree} tokens", flush=True)
        out.append(record)
        torch.save(out, a.out)


if __name__ == "__main__":
    main()
