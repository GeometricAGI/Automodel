"""Diagnostic: are the drafter's target features the same when vLLM computes them in prefill and in decode?

Generate greedily from one prompt with DSpark on, recording every combine_hidden_states input; then prefill the
prompt plus that output as a single new prompt (prefix caching off) and record its features. Each decode step's
first row is a verified position, so it should equal some row of the prefill features.

  VLLM_ENABLE_V1_MULTIPROCESSING=0 python capture_paths.py <replay .jsonl> <out .pt>
"""
import json
import pathlib
import sys

import torch

from vllm.models.deepseek_v41.nvidia import dspark as vllm_dspark

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from paths import CKPT  # noqa: E402

calls: list[torch.Tensor] = []
_combine = vllm_dspark.DSparkDeepseekV4Model.combine_hidden_states


def _recording_combine(self, aux_hidden_states):
    calls.append(aux_hidden_states.detach().float().cpu())
    return _combine(self, aux_hidden_states)


vllm_dspark.DSparkDeepseekV4Model.combine_hidden_states = _recording_combine


def main():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.tokenizers.deepseek_v41_encoding import encode_messages

    tok = AutoTokenizer.from_pretrained(CKPT)
    row = json.loads(open(sys.argv[1]).readline())
    prompt = tok.encode(encode_messages(row["messages"][:-1], thinking_mode="thinking", drop_thinking=True,
                                        reasoning_effort="high"), add_special_tokens=False)
    llm = LLM(str(CKPT), max_model_len=8192, gpu_memory_utilization=0.97, kv_cache_memory_bytes=4831838208,
              max_num_seqs=4, max_num_batched_tokens=8192, enforce_eager=True, enable_prefix_caching=False,
              engram_config={"use_thp": True}, safetensors_load_strategy="lazy",
              speculative_config={"method": "dspark", "num_speculative_tokens": 3})
    calls.clear()
    out = llm.generate([TokensPrompt(prompt_token_ids=prompt)], SamplingParams(max_tokens=200, temperature=0),
                       use_tqdm=False)[0].outputs[0].token_ids
    decode_calls = list(calls)
    print("generation calls (rows):", [c.shape[0] for c in decode_calls[:12]], "...", len(decode_calls))
    calls.clear()
    full = prompt + list(out)
    llm.generate([TokensPrompt(prompt_token_ids=full)], SamplingParams(max_tokens=1, temperature=0), use_tqdm=False)
    prefill = torch.cat([c for c in calls if c.shape[0] > 64])[: len(full)]
    print("prefill rows", prefill.shape[0], "for", len(full), "tokens")
    gen_prefill = decode_calls[0][: len(prompt)]
    cos = torch.nn.functional.cosine_similarity
    torch.save({"prompt_len": len(prompt), "tokens": full, "decode_calls": decode_calls, "prefill": prefill},
               sys.argv[2])
    row_cos = cos(gen_prefill, prefill[: len(prompt)], dim=-1)
    print("prompt prefill features, 80-token vs 191-token prefill: max abs diff",
          (gen_prefill - prefill[: len(prompt)]).abs().max().item(), "| per-row cosine min/median",
          round(row_cos.min().item(), 4), round(row_cos.median().item(), 4))
    pn = torch.nn.functional.normalize(prefill, dim=-1)
    best, where = [], []
    for c in decode_calls[1:60]:
        sims = torch.nn.functional.normalize(c[:1], dim=-1) @ pn.T
        s, i = sims.max(-1)
        best.append(round(s.item(), 4)); where.append(int(i))
    print("decode first-row best cosine vs prefill rows:", best[:30])
    print("matched positions:", where[:30], "(prompt length", len(prompt), ")")
    # Per-layer: does decode row at matched position agree per 5120 slice?
    c, i = decode_calls[5][0], where[4]
    print("per-layer cosine at one matched position:",
          [round(cos(c[k * 5120:(k + 1) * 5120], prefill[i, k * 5120:(k + 1) * 5120], dim=0).item(), 4) for k in range(3)])


if __name__ == "__main__":
    main()
