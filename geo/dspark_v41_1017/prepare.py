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

from replay import TOKENIZER, from_token_ids

EOS_ID = TOKENIZER.convert_tokens_to_ids("<｜end▁of▁sentence｜>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("inputs", type=pathlib.Path, nargs="+")
    ap.add_argument("--val", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows, dropped = [], 0
    for path in a.inputs:
        for line in open(path):
            row = json.loads(line)
            gen = row["completion_token_ids"]
            if not gen or any(not ids or ids[-1] != EOS_ID for ids in gen):
                dropped += 1
                continue
            turns = iter(gen)
            out = {"messages": [from_token_ids(next(turns)) if m["role"] == "assistant" else m
                                for m in row["messages"]]}
            if row.get("tools"):
                out["tools"] = row["tools"]
            rows.append(out)
    random.Random(a.seed).shuffle(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    n_val = min(a.val, len(rows) // 10)
    for name, part in (("val", rows[:n_val]), ("train", rows[n_val:])):
        with open(a.out / f"{name}.jsonl", "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in part)
    print(f"{len(rows) - n_val} train, {n_val} val; dropped {dropped} rows with a truncated turn")


if __name__ == "__main__":
    main()
