"""Replay prompts through the local engine, measuring DSpark acceptance per content type and saving the
conversations as drafter-training data (NeMo AutoModel `messages` format, reasoning kept as reasoning_content).

Tokens per streamed chunk = 1 + accepted drafts for that request's decode step, since vLLM streams each step's
tokens together; `return_token_ids` gives the exact count. Run the server with adaptive verification off so every
step drafts the same number of tokens.

  python replay.py agentic --n 16 --out data/agentic.jsonl     # latency_tests coding rollouts, turn by turn
  python replay.py prompts --n 2000 --out data/blend.jsonl      # single-turn prompts from open-perfectblend
  python replay.py sessions --n 5000 --out data/sessions.jsonl  # requesty agent sessions, tools declared

Resumable: rerunning the same command after a stop keeps the rows already written and generates only the rest.
"""
import argparse
import asyncio
import collections
import json
import os
import pathlib
import random
import re
import time

import httpx
from transformers import AutoTokenizer

# Engines to spread requests over (comma-separated base URLs); --concurrency applies to each.
URLS = [u.rstrip("/") + "/v1/chat/completions"
        for u in os.environ.get("DSPARK_ENGINES", "http://localhost:8000").split(",")]


class Engines:
    """One semaphore per engine; item i goes to engine i mod n."""

    def __init__(self, conc):
        self.sems = [asyncio.Semaphore(conc) for _ in URLS]

    def pick(self, i):
        return URLS[i % len(URLS)], self.sems[i % len(URLS)]
MODEL = "deepseek/deepseek-v4.1-flash-1017"
from vllm.tokenizers.deepseek_v41 import _normalize_messages
from vllm.tokenizers.deepseek_v41_encoding import encode_messages

from paths import CKPT, WORK

ROLLOUTS = WORK / "agentic/corpus/rollout_ctx32768.json"
BLEND_JSONL = WORK / "data/open-perfectblend.jsonl"
REQUESTY = WORK / "data/requesty-agent-sessions"
TOKENIZER = AutoTokenizer.from_pretrained(CKPT)
EOS = "<｜end▁of▁sentence｜>"
EOS_ID = TOKENIZER.convert_tokens_to_ids(EOS)
TOOL_CALL = re.compile(r"<function|\"name\"\s*:|```(?:json|bash)?\s*\{|\bexecute_bash\b|\bstr_replace_editor\b")


class Stats:
    def __init__(self):
        self.steps = collections.Counter()
        self.tokens = collections.Counter()
        self.hist = collections.defaultdict(collections.Counter)

    def add(self, kind, n):
        self.steps[kind] += 1
        self.tokens[kind] += n
        self.hist[kind][min(n, 4)] += 1

    def report(self):
        rows = []
        for kind in sorted(self.steps):
            s, t = self.steps[kind], self.tokens[kind]
            h = self.hist[kind]
            dist = " ".join(f"{k}:{100 * h[k] / s:4.1f}%" for k in sorted(h))
            rows.append(f"  {kind:10s} steps {s:7d}  tokens {t:8d}  tokens/step {t / s:.2f}  [{dist}]")
        return "\n".join(rows)


TEMPERATURE = None


async def turn(client, messages, stats, max_tokens, tools=None, url=URLS[0]):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "stream": True,
            **({"tools": tools, "tool_choice": "auto"} if tools else {}),
            **({"temperature": TEMPERATURE} if TEMPERATURE is not None else {}),
            "return_token_ids": True, "stream_options": {"include_usage": True}}
    reasoning, content, token_ids = [], [], []
    calls: dict[int, dict] = {}
    usage = None
    async with client.stream("POST", url, json=body) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            usage = ev.get("usage") or usage
            if not ev.get("choices"):
                continue
            ch = ev["choices"][0]
            ids = ch.get("token_ids") or []
            token_ids += ids
            n = len(ids)
            d = ch.get("delta", {})
            for tc in d.get("tool_calls") or []:
                call = calls.setdefault(tc.get("index", 0), {"id": tc.get("id"), "type": "function",
                                                             "function": {"name": "", "arguments": ""}})
                fn = tc.get("function") or {}
                call["function"]["name"] += fn.get("name") or ""
                call["function"]["arguments"] += fn.get("arguments") or ""
            r_txt, c_txt = d.get("reasoning") or d.get("reasoning_content") or "", d.get("content") or ""
            if r_txt:
                reasoning.append(r_txt)
            if c_txt:
                content.append(c_txt)
            if n:
                if d.get("tool_calls"):
                    kind = "tool_call"
                elif r_txt and not c_txt:
                    kind = "reasoning"
                elif c_txt and not r_txt:
                    kind = "tool_call" if TOOL_CALL.search("".join(content)[-400:]) else "answer"
                elif not r_txt and not c_txt:
                    # tokens the tool-call parser held back: the DSML call, dropped when the request declares no tools
                    kind = "tool_call"
                else:
                    kind = "mixed"
                stats.add(kind, n)
    return from_token_ids(token_ids), token_ids, usage


def from_token_ids(token_ids):
    """The assistant message exactly as generated: the parsers trim the whitespace before </think> and drop a
    DSML tool call when the request declares no tools, so the training template would not reproduce the tokens."""
    raw = TOKENIZER.decode(token_ids, skip_special_tokens=False)
    reasoning, sep, answer = raw.partition("</think>")
    if not sep:
        reasoning, answer = "", raw
    return {"role": "assistant", "content": answer.removesuffix(EOS), "reasoning_content": reasoning}


def done_keys(out):
    """Keys of the rows a previous run already wrote, so a restarted run appends only the rest. A row cut off by
    the stop is dropped from the file and generated again."""
    if not out.exists():
        return set()
    keys, good = set(), []
    for line in out.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "key" not in row:
            raise SystemExit(f"{out} was written by an older replay.py without row keys; use a new --out path")
        keys.add(row["key"])
        good.append(line)
    out.write_text("".join(g + "\n" for g in good))
    return keys


async def agentic(n, conc, out, max_tokens):
    convs = json.load(open(ROLLOUTS))["conversations"][:n]
    done = done_keys(out)
    stats, engines, tokens = Stats(), Engines(conc), [0]
    with out.open("a") as f:
      async with httpx.AsyncClient(timeout=None, limits=httpx.Limits(max_connections=None, max_keepalive_connections=None)) as client:
        async def one(i, conv):
            if f"rollout:{i}" in done:
                return
            url, sem = engines.pick(i)
            async with sem:
                history, completions = [], []
                for t in conv:
                    history += t["messages"]
                    msg, ids, u = await turn(client, history, stats, max_tokens, url=url)
                    history.append(msg)
                    completions.append(ids)
                    tokens[0] += (u or {}).get("completion_tokens", 0)
                f.write(json.dumps({"key": f"rollout:{i}", "messages": history, "completion_token_ids": completions,
                                    "source": "agentic-rollout"}) + "\n")
                f.flush()
        t0 = time.time()
        await asyncio.gather(*(one(i, c) for i, c in enumerate(convs)))
    return stats, tokens[0], time.time() - t0


async def prompts(n, conc, out, max_tokens, seed):
    # open-perfectblend.jsonl: the dataset's parquet rows in file order, one JSON object per line.
    with open(BLEND_JSONL) as fh:
        rows = fh.readlines()
    random.Random(seed).shuffle(rows)
    done = done_keys(out)
    stats, engines, tokens = Stats(), Engines(conc), [0]
    with out.open("a") as f:
      async with httpx.AsyncClient(timeout=None, limits=httpx.Limits(max_connections=None, max_keepalive_connections=None)) as client:
        async def one(i, line):
            if f"blend:{seed}:{i}" in done:
                return
            row = json.loads(line)
            conv = row.get("conversations") or row.get("messages") or []
            first = next((m for m in conv if m.get("from", m.get("role")) in ("human", "user")), None)
            if first is None:
                return
            msgs = [{"role": "user", "content": first.get("value", first.get("content"))}]
            url, sem = engines.pick(i)
            async with sem:
                msg, ids, u = await turn(client, msgs, stats, max_tokens, url=url)
            tokens[0] += (u or {}).get("completion_tokens", 0)
            f.write(json.dumps({"key": f"blend:{seed}:{i}", "messages": msgs + [msg], "completion_token_ids": [ids],
                                "source": f"open-perfectblend:{row.get('source', '')}"}) + "\n")
            f.flush()
        t0 = time.time()
        await asyncio.gather(*(one(i, r) for i, r in enumerate(rows[:n])))
    return stats, tokens[0], time.time() - t0


def render_tokens(messages, tools):
    """Prompt tokens for a request, rendered as the server does (tools in the system message)."""
    conv = _normalize_messages([{k: v for k, v in m.items() if k != "train"} for m in messages])
    system = next((m for m in conv if m["role"] == "system"), None)
    if system is None:
        system = {"role": "system", "content": ""}
        conv.insert(0, system)
    system["tools"] = tools
    prompt = encode_messages(conv, thinking_mode="thinking", drop_thinking=True, reasoning_effort="high")
    return len(TOKENIZER.encode(prompt, add_special_tokens=False))


MESSAGE_OVERHEAD = 8  # role markers and separators the template adds around a message


def message_tokens(m):
    """Tokens a history message adds to a prompt: exact for our answers (their generated ids), a close estimate
    (content, reasoning and tool calls tokenized once and cached) for the others."""
    if "_ids" in m:
        return len(m["_ids"]) + MESSAGE_OVERHEAD
    if "_n" not in m:
        text = (m.get("content") or "") if isinstance(m.get("content"), str) else json.dumps(m.get("content"))
        text += m.get("reasoning_content") or ""
        if m.get("tool_calls"):
            text += json.dumps([tc["function"] for tc in m["tool_calls"]])
        m["_n"] = len(TOKENIZER.encode(text, add_special_tokens=False)) + MESSAGE_OVERHEAD
    return m["_n"]


async def run_session(client, msgs, tools, turns, stats, max_tokens, budget, url):
    """Replay one agent session, putting the model's own answer in place of each original one.

    The context is the system message plus a window of recent messages. When a turn's context would pass the
    budget, the window's rows are closed and a new window starts at the latest message boundary that leaves half
    the budget, so each row is a context exactly as its turns were generated from. Answers carried over from the
    previous window stay as history with train: false. Returns (rows, completion tokens).
    """
    system, history = msgs[0], []
    system_tokens = render_tokens([system], tools)  # the system prompt with every tool schema, exact
    generated_in = {}  # history index -> window number of our answers
    ours = set(turns)
    window, start, rows, generated, pos = 0, 0, [], 0, 1

    def close_window(end):
        mine = [i for i in range(start, end) if generated_in.get(i) == window]
        if not mine:
            return
        rows.append({"messages": [system] + [dict(history[i], train=generated_in.get(i) == window)
                                             for i in range(start, end)],
                     "tools": tools, "completion_token_ids": [history[i]["_ids"] for i in mine]})

    for t in turns:
        if t >= len(msgs) or msgs[t]["role"] != "assistant":
            break
        for m in msgs[pos:t]:
            if m["role"] == "assistant":
                m = dict(m, train=False)  # an original answer that is not a turn point: history only
            history.append(m)
        pos = t + 1
        if system_tokens + sum(message_tokens(m) for m in history[start:]) > budget:
            close_window(len(history))
            new_start, size = None, system_tokens
            for i in range(len(history) - 1, start, -1):  # the earliest boundary whose suffix fits half the budget
                size += message_tokens(history[i])
                if size > budget // 2:
                    break
                if history[i]["role"] in ("user", "assistant"):
                    new_start = i
            if new_start is None:
                return rows, generated
            window, start = window + 1, new_start
        context = [{k: v for k, v in m.items() if k not in ("train", "_ids", "_n")} for m in [system] + history[start:]]
        msg, ids, usage = await turn(client, context, stats, max_tokens, tools, url)
        generated += (usage or {}).get("completion_tokens", 0)
        if not ids or ids[-1] != EOS_ID:
            break  # cut off at max_tokens: no history to continue from
        history.append(dict(msg, _ids=ids))
        generated_in[len(history) - 1] = window
    close_window(len(history))
    return rows, generated


async def sessions(n, conc, out, max_tokens, budget, seed):
    # sessions.jsonl holds the same sessions as the parquet shards; lines stay raw until their session starts.
    with open(REQUESTY / "sessions.jsonl") as fh:
        rows = fh.readlines()
    random.Random(seed).shuffle(rows)
    done = done_keys(out)
    stats, engines, tokens = Stats(), Engines(conc), [0]
    with out.open("a") as f:
      async with httpx.AsyncClient(timeout=None, limits=httpx.Limits(max_connections=None, max_keepalive_connections=None)) as client:
        async def one(i, line):
            key = "requesty:" + re.match(r'\{"id": "([^"]+)"', line).group(1)
            if key in done:
                return
            url, sem = engines.pick(i)
            async with sem:
                row = json.loads(line)
                session_rows, generated = await run_session(client, row["messages"], row["tools"], row["turns"],
                                                            stats, max_tokens, budget, url)
            tokens[0] += generated
            # A session's rows are written together, so a stop never leaves half a session behind.
            for r in session_rows:
                r["messages"] = [{k: v for k, v in m.items() if k not in ("_ids", "_n")} for m in r["messages"]]
                f.write(json.dumps({"key": key, **r, "source": f"requesty:{row.get('instance_id', '')}"}) + "\n")
            if not session_rows:
                f.write(json.dumps({"key": key, "messages": [], "completion_token_ids": [], "source": "empty"}) + "\n")
            f.flush()
        t0 = time.time()
        await asyncio.gather(*(one(i, r) for i, r in enumerate(rows[:n])))
    return stats, tokens[0], time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["agentic", "prompts", "sessions"])
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=32768, help="sessions: context tokens per training row")
    ap.add_argument("--temperature", type=float, default=None, help="default: the server's (1.0 in production)")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    global TEMPERATURE
    TEMPERATURE = a.temperature
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.mode == "agentic":
        stats, toks, secs = asyncio.run(agentic(a.n, a.concurrency, a.out, a.max_tokens or 1024))
    elif a.mode == "sessions":
        stats, toks, secs = asyncio.run(sessions(a.n, a.concurrency, a.out, a.max_tokens or 4096, a.budget, a.seed))
    else:
        stats, toks, secs = asyncio.run(prompts(a.n, a.concurrency, a.out, a.max_tokens or 8192, a.seed))
    print(f"[{a.mode}] {toks:,} completion tokens in {secs:.0f} s = {toks / secs:.0f} tok/s at concurrency {a.concurrency}")
    print(f"[{a.mode}] tokens per decode step by content (1 + accepted drafts):\n{stats.report()}")


if __name__ == "__main__":
    main()
