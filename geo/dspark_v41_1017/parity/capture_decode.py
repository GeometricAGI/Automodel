"""Capture the drafter's target features the way serving computes them: generate greedily with DSpark on, one
request at a time, and record each target forward's positions, input tokens and the combine_hidden_states input.

vLLM computes prefill features only for the last 128 prompt positions (the draft's attention window); decode
steps compute them for every verified token. A row is kept when every token from the step's first position up to
its own equals the final sequence (so its causal context is the real one).

  VLLM_ENABLE_V1_MULTIPROCESSING=0 python capture_decode.py <replay .jsonl> <out .pt> [--n 16] [--max-tokens 768]
"""
import argparse
import json
import pathlib
import sys

import torch

from vllm.models.deepseek_v41.nvidia import dspark as vllm_dspark
from vllm.models.deepseek_v41.nvidia import model as vllm_model

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from paths import CKPT  # noqa: E402

steps: list[dict] = []
_combine = vllm_dspark.DSparkDeepseekV4Model.combine_hidden_states
_forward = vllm_model.DeepseekV4Model.forward


def _recording_forward(self, input_ids, positions, *args, **kwargs):
    steps.append({"ids": input_ids.detach().cpu(), "pos": positions.detach().cpu()})
    return _forward(self, input_ids, positions, *args, **kwargs)


def _recording_combine(self, aux_hidden_states):
    steps[-1]["feats"] = aux_hidden_states.detach().to("cpu", torch.bfloat16)
    return _combine(self, aux_hidden_states)


vllm_model.DeepseekV4Model.forward = _recording_forward
vllm_dspark.DSparkDeepseekV4Model.combine_hidden_states = _recording_combine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("replay", type=pathlib.Path)
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=768)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.tokenizers.deepseek_v41_encoding import encode_messages

    tok = AutoTokenizer.from_pretrained(CKPT)
    prompts = []
    for line in open(a.replay):
        row = json.loads(line)
        prompts.append(tok.encode(encode_messages(row["messages"][:-1], thinking_mode="thinking", drop_thinking=True,
                                                  reasoning_effort="high"), add_special_tokens=False))
        if len(prompts) == a.n:
            break
    llm = LLM(str(CKPT), max_model_len=8192, gpu_memory_utilization=0.97, kv_cache_memory_bytes=4831838208,
              max_num_seqs=1, max_num_batched_tokens=8192, enforce_eager=True, enable_prefix_caching=False,
              engram_config={"use_thp": True}, safetensors_load_strategy="lazy",
              speculative_config={"method": "dspark", "num_speculative_tokens": 3})
    out = []
    for prompt in prompts:
        steps.clear()
        gen = llm.generate([TokensPrompt(prompt_token_ids=prompt)],
                           SamplingParams(max_tokens=a.max_tokens, temperature=0), use_tqdm=False)[0].outputs[0]
        seq = torch.tensor(prompt + list(gen.token_ids))
        feats = torch.zeros(len(seq), steps[0]["feats"].shape[1], dtype=torch.bfloat16)
        valid = torch.zeros(len(seq), dtype=torch.bool)
        for s in steps:
            if "feats" not in s:
                continue
            ok = True
            for r, (p, t) in enumerate(zip(s["pos"].tolist(), s["ids"].tolist())):
                ok = ok and p < len(seq) and seq[p].item() == t
                if ok and s["feats"][r].abs().sum() > 0:
                    feats[p], valid[p] = s["feats"][r], True
        out.append({"input_ids": seq, "prompt_len": len(prompt), "features": feats, "valid": valid})
        print(f"{len(prompt)} prompt + {len(gen.token_ids)} generated; valid feature rows {int(valid.sum())}; "
              f"generated positions valid {bool(valid[len(prompt):-1].all())}")
    torch.save(out, a.out)


if __name__ == "__main__":
    main()
