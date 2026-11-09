"""Render saved conversations with NeMo AutoModel's V4.1 training template (thinking mode) and check that each
assistant turn's supervised span tokenizes to exactly the token IDs the engine generated."""
import json
import pathlib
import sys

from transformers import AutoTokenizer

from prepare import parse_tool_arguments
from replay import from_token_ids

from paths import CKPT, TRAINING_TEMPLATE

tok = AutoTokenizer.from_pretrained(CKPT)
template = TRAINING_TEMPLATE.read_text()
REBUILD = "--rebuild" in sys.argv
ok = bad = 0
for path in [a for a in sys.argv[1:] if not a.startswith("--")]:
    for line in list(open(path))[:200]:
        row = json.loads(line)
        if REBUILD:
            turns = iter(row["completion_token_ids"])
            row["messages"] = [dict(from_token_ids(next(turns)), train=True)
                               if m["role"] == "assistant" and m.get("train", True) else m for m in row["messages"]]
        if not row["messages"]:
            continue
        row["messages"] = parse_tool_arguments(row["messages"])
        out = tok.apply_chat_template(row["messages"], chat_template=template, tools=row.get("tools"),
                                      tokenize=True, return_dict=True,
                                      return_assistant_tokens_mask=True, thinking_mode="thinking",
                                      drop_thinking="--keep-thinking" not in sys.argv)
        ids, mask = out["input_ids"], out["assistant_masks"]
        spans, cur = [], []
        for i, m in zip(ids, mask):
            if m:
                cur.append(i)
            elif cur:
                spans.append(cur); cur = []
        if cur:
            spans.append(cur)
        gen = row["completion_token_ids"]
        for s, g in zip(spans, gen):
            if s == g:
                ok += 1
            else:
                bad += 1
                if bad <= 3:
                    k = next((j for j in range(min(len(s), len(g))) if s[j] != g[j]), min(len(s), len(g)))
                    print(f"{path}: mismatch at token {k}: template {tok.decode(s[max(0,k-5):k+5])!r} "
                          f"vs generated {tok.decode(g[max(0,k-5):k+5])!r} (lens {len(s)} vs {len(g)})")
        if len(spans) != len(gen):
            print(f"{path}: {len(spans)} supervised spans vs {len(gen)} generated turns")
print(f"turns identical: {ok}, different: {bad}")
