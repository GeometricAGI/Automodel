"""Locations the scripts share, overridable from the environment.

DSPARK_CKPT  the served checkpoint (default ~/models/DeepSeek-V4.1-Flash-1017-vLLM-B300)
DSPARK_WORK  inputs that are not in this repo (default ~/runs): data/requesty-agent-sessions/sessions.jsonl,
             data/open-perfectblend.jsonl and the rollout corpus
"""
import os
import pathlib

CKPT = pathlib.Path(os.environ.get("DSPARK_CKPT", "~/models/DeepSeek-V4.1-Flash-1017-vLLM-B300")).expanduser()
WORK = pathlib.Path(os.environ.get("DSPARK_WORK", "~/runs")).expanduser()
REPO = pathlib.Path(__file__).resolve().parents[2]
TRAINING_TEMPLATE = REPO / "examples/llm_finetune/deepseek_v41/deepseek_v41_flash_training_chat_template.jinja"
