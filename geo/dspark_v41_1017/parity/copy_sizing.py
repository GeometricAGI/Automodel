"""How much of the model's agentic output could a copy-based drafter (suffix/prompt lookup) take?

Over replay.py's held-out session rows (evaluate.sh's sessions.jsonl), each row is laid out as one token stream:
every message's text tokenized in order, with our generated turns as their exact token ids. Inside each generated
turn a suffix drafter is simulated: at every step, look up the most recent earlier occurrence of the last 8 tokens
(else the last 4) anywhere before, propose the up to --max-draft tokens that followed it, and accept the prefix that
matches what the model generated (plus the target's own token). Steps with no match decode one token, which
DSpark would take instead.

Reports the share of generated tokens accepted from copies, the accepted lengths, and a hybrid estimate: copy steps
where the 8-token pattern matched, DSpark (at --dspark tokens per step) for every other token.

  python copy_sizing.py <sessions.jsonl> <tokenizer dir> [--max-draft 32] [--dspark 2.53] [--workers 64]
"""
import argparse
import collections
import json
import multiprocessing as mp

TOK = None


def init(path):
    global TOK
    from transformers import AutoTokenizer

    TOK = AutoTokenizer.from_pretrained(path)


def text_of(m):
    text = (m.get("reasoning_content") or "") + (m.get("content") if isinstance(m.get("content"), str) else
                                                 json.dumps(m.get("content") or ""))
    if m.get("tool_calls"):
        text += json.dumps([tc["function"] for tc in m["tool_calls"]])
    return text


def simulate(args):
    line, max_draft = args
    row = json.loads(line)
    gen = iter(row["completion_token_ids"])
    stream = TOK.encode(json.dumps(row.get("tools") or []), add_special_tokens=False)
    idx = {4: {}, 8: {}}
    s = collections.Counter()
    hist = collections.Counter()

    def add(upto):  # index the n-grams ending before position upto
        for n, d in idx.items():
            for e in range(max(n, add.done + 1), upto + 1):
                d[tuple(stream[e - n : e])] = e
        add.done = max(add.done, upto)

    add.done = 0
    for m in row["messages"]:
        if m.get("train"):
            ids = next(gen)
            start = len(stream)
            stream.extend(ids)
            i = 0
            while i < len(ids):
                pos = start + i
                add(pos - 1)  # earlier occurrences only: the pattern ending at pos is the one being looked up
                hit, n_used = None, 0
                for n in (8, 4):
                    if pos >= n and (e := idx[n].get(tuple(stream[pos - n : pos]))) is not None and e < pos:
                        hit, n_used = e, n
                        break
                acc = 0
                if hit is not None:
                    while acc < max_draft and i + acc < len(ids) and hit + acc < pos and stream[hit + acc] == ids[i + acc]:
                        acc += 1
                    s[f"steps_{n_used}"] += 1
                    s[f"tokens_{n_used}"] += acc + 1
                    hist[min(acc, 32)] += 1
                else:
                    s["plain"] += 1
                i += acc + 1
            s["generated"] += len(ids)
        else:
            stream.extend(TOK.encode(text_of(m), add_special_tokens=False))
    s["context"] += len(stream)
    return s, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions")
    ap.add_argument("tokenizer")
    ap.add_argument("--max-draft", type=int, default=32)
    ap.add_argument("--dspark", type=float, default=2.53, help="DSpark tokens per step for the other tokens")
    ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args()
    lines = [l for l in open(a.sessions) if '"train": true' in l]
    total, hist = collections.Counter(), collections.Counter()
    with mp.Pool(a.workers, initializer=init, initargs=(a.tokenizer,)) as pool:
        for s, h in pool.imap_unordered(simulate, [(l, a.max_draft) for l in lines]):
            total.update(s)
            hist.update(h)
    g = total["generated"]
    copy8 = total["tokens_8"] - total["steps_8"]  # accepted from the copy, not the target's own token
    copy4 = total["tokens_4"] - total["steps_4"]
    print(f"{len(lines)} rows, {g} generated tokens, {total['context']} stream tokens")
    for n in (8, 4):
        st = total[f"steps_{n}"]
        print(f"  {n}-token pattern matched: {st} steps, {total[f'tokens_{n}'] / max(1, st):.2f} tokens per step")
    print(f"  no match: {total['plain']} steps")
    print(f"  generated tokens accepted from copies: {(copy8 + copy4) / g:.1%} ({copy8 / g:.1%} on 8-token matches)")
    steps = sum(hist.values())
    print("  accepted per matched step: " + ", ".join(f"{k}: {v / steps:.1%}" for k, v in sorted(hist.items())
                                                     if k in (0, 1, 2, 3, 4, 8, 16, 32)))
    # Hybrid: copy steps on 8-token matches with at least 4 accepted would beat DSpark; take every 8-token step
    # (a real hybrid must decide before verifying), DSpark for the rest.
    rest = g - total["tokens_8"]
    hybrid = g / (total["steps_8"] + rest / a.dspark)
    print(f"  hybrid (copy on 8-token matches, DSpark at {a.dspark} otherwise): {hybrid:.2f} tokens per step, "
          f"against {a.dspark} for DSpark alone")


if __name__ == "__main__":
    main()
