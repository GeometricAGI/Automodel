"""Turn replay.py output into NeMo AutoModel DSpark training data (train.jsonl, val.jsonl of {"messages", "tools"?}).

Assistant turns are rebuilt from the generated token IDs, so the training template reproduces them exactly (see
template_check.py); rows with a turn that hit max_tokens are dropped, since the template would close the turn with
</think> and an end-of-sentence token the model never wrote.

  python prepare.py <out dir> <replay .jsonl>... [--val 2000] [--seed 0]
"""
import argparse
import json
import pathlib
import random

from replay import TOKENIZER, from_token_ids, held_out

EOS_ID = TOKENIZER.convert_tokens_to_ids("<｜end▁of▁sentence｜>")


def parse_tool_arguments(messages):
    """Tool-call arguments as objects: the training template needs them so, and vLLM's encoder parses JSON-string
    arguments the same way before rendering them."""
    out = []
    for m in messages:
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                args = tc["function"]["arguments"]
                if isinstance(args, str):
                    try:
                        parsed = json.loads(args)
                    except json.JSONDecodeError:
                        parsed = None
                    args = parsed if isinstance(parsed, dict) else {"arguments": args}
                calls.append({**tc, "function": {**tc["function"], "arguments": args}})
            m = {**m, "tool_calls": calls}
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("inputs", type=pathlib.Path, nargs="+")
    ap.add_argument("--val", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows, dropped, heldout = [], 0, 0
    for path in a.inputs:
        for line in open(path):
            row = json.loads(line)
            if row.get("key", "").startswith("requesty:") and held_out(row["key"].split(":", 1)[1]):
                heldout += 1
                continue
            gen = row["completion_token_ids"]
            if not gen or any(not ids or ids[-1] != EOS_ID for ids in gen):
                dropped += 1
                continue
            # Supervised turns (no train flag, or train: true) are the generated ones, in order.
            turns = iter(gen)
            out = {"messages": parse_tool_arguments([dict(from_token_ids(next(turns)), **({"train": True} if "train" in m else {}))
                                if m["role"] == "assistant" and m.get("train", True) else m
                                for m in row["messages"]])}
            if row.get("tools"):
                out["tools"] = row["tools"]
            if out["messages"]:
                rows.append(out)
    random.Random(a.seed).shuffle(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    n_val = min(a.val, len(rows) // 10)
    for name, part in (("val", rows[:n_val]), ("train", rows[n_val:])):
        with open(a.out / f"{name}.jsonl", "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in part)
    print(f"{len(rows) - n_val} train, {n_val} val; dropped {dropped} rows with a truncated turn and {heldout} rows of "
          f"held-out evaluation sessions")


if __name__ == "__main__":
    main()
