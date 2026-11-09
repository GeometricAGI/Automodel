# Fine-tuning DeepSeek-V4.1-Flash-1017's DSpark drafter

Scripts for regenerating training data with the served INT3 model, training the released drafter further with
`examples/speculative/dspark/deepseek_v41_flash_1017_dspark_finetune.yaml`, and putting the result back into the
served checkpoint. Data and captures from the first runs are in the private Hugging Face dataset
[`Geometric-AI/dsv41-flash-1017-dspark`](https://huggingface.co/datasets/Geometric-AI/dsv41-flash-1017-dspark).

Set `DSPARK_CKPT` to the served checkpoint and `DSPARK_WORK` to the folder holding inputs that are not in this repo
(the rollout corpus and Open-PerfectBlend); see `paths.py`.

## Workflow

1. **Generate** (vLLM environment, engine serving on port 8000 with adaptive verification off). Agentic data:
   `python replay.py sessions --n 5000 --concurrency 24 --out data/sessions.jsonl` replays
   [`Geometric-AI/requesty-agent-sessions`](https://huggingface.co/datasets/Geometric-AI/requesty-agent-sessions)
   (`DSPARK_WORK/data/requesty-agent-sessions`) with each session's tools declared, putting the model's own answer
   in place of each original one. The context is the system message plus a window of recent messages up to
   `--budget` tokens (32k); when it would pass that, the window becomes a training row and a new window starts with
   half the budget of recent history, whose earlier answers are marked `train: false` (context, not supervised).
   Each row is therefore exactly the context its turns were generated from, and holds many turns. Single-turn
   prompts: `python replay.py prompts --n 300000 --concurrency 24 --out data/blend.jsonl`. Both record tokens per
   decode step by content type and save each turn's generated token IDs; rerunning the same command after a stop
   appends only what is not yet written.
2. **Prepare**: `python prepare.py data/train data/*.jsonl` rebuilds each assistant turn from its token IDs (the
   reasoning and tool-call parsers change the text) and drops rows with a turn that hit `max_tokens`.
   `template_check.py` checks that NeMo's training template reproduces the generated tokens.
3. **Train**: `torchrun --standalone --nproc_per_node=8 -m nemo_automodel.recipes.llm.train_dspark -c
   examples/speculative/dspark/deepseek_v41_flash_1017_dspark_finetune.yaml`. Relaunching resumes from the latest
   checkpoint.
4. **Export**: `python export.py $DSPARK_CKPT <NeMo consolidated drafter> <out dir>` quantizes the drafter back to
   the checkpoint's MXFP4/FP8 format and writes a servable copy that links every other file. `ab_export.sh` serves
   it and the original in turn and compares acceptance.

## Checks done on one B300

| Check | Result |
|---|---|
| `roundtrip.py`: dequantize to BF16 and quantize back | MXFP4 bytes identical; FP8 values identical (some blocks choose an equivalent smaller scale) |
| `export.py` re-export of the released drafter, served | 2.62 vs 2.62 tokens per step on reasoning, 2.93 vs 2.93 on answers (240 prompts, T=0) |
| `parity/load_check.py`, `parity/expert_check.py` | NeMo loads every released tensor; values equal a direct dequantization |
| `parity/nemo_accept.py` on `parity/capture_decode.py` features | NeMo's draft accepts 3.07 tokens per step (greedy, 3 drafts); vLLM 3.0-3.3 on the same generations |
| `template_check.py` | 99.7% of turns render back to the generated tokens; the rest are retokenization or truncation |

## Things to know

- vLLM computes the drafter's prefill features only for the last 128 prompt positions (its attention window); the
  others are zero. Capture features from decode steps (`parity/capture_decode.py`), not from one long prefill.
- The prompt's reasoning effort and framing must match serving: thinking mode, effort `high`. NeMo's training
  template renders the same prompt as vLLM's encoder.
- `parity/capture_paths.py` shows the INT3 model's features for the same positions differ between prefill batch
  sizes (median cosine 0.947 with the Lattice kernels, 0.964 without); identical inputs should agree to better
  than 0.999.
- Agentic data should declare its tools in the request, as production clients do: without tools the parser
  drops DSML calls from the response and the template drops earlier turns' reasoning.
- Left-truncating agentic contexts to 16-64k changes the drafter's features (cosine 0.76-0.87 to the full
  context's) but not its predictions: acceptance on the same tokens is unchanged (`parity/long_context.py`,
  `parity/nemo_long.py`, 20 requesty turns of 46-164k tokens). The model's own output diverges after 30-60 tokens,
  so generate the data from the same truncated context the trainer sees, as `replay.py sessions` does.
