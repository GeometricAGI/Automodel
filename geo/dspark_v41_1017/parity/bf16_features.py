"""Recompute capture_decode.py's sequences' drafter features with the training target (the BF16 checkpoint).

Builds the fine-tune recipe (its frozen target, EP over all ranks) and runs each captured sequence through the
recipe's target wrapper as a one-document packed row, as training does. Every rank runs every sequence; rank 0
saves a list of [sequence, features] BF16 tensors in the capture's order, and the final-norm states ([sequence,
hidden], for the target's output distribution) beside it as <out>.last.pt. feature_gap.py and tree_sizing.py read them.

  torchrun --standalone --nproc_per_node=8 bf16_features.py <capture .pt> <out .pt> -c <recipe yaml> [key=value ...]
"""
import sys

import torch
import torch.distributed as dist

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.llm.train_dspark import TrainDSparkRecipe

PAD_MULTIPLE = 128


def main():
    capture, out = sys.argv[1], sys.argv[2]
    cfg = parse_args_and_load_config(argv=sys.argv[3:])
    recipe = TrainDSparkRecipe(cfg)
    recipe.setup()
    records = torch.load(capture, weights_only=False)
    features, last = [], []
    for i, rec in enumerate(records):
        n = len(rec["input_ids"])
        width = -(-n // PAD_MULTIPLE) * PAD_MULTIPLE
        ids = torch.cat([rec["input_ids"], rec["input_ids"][-1:].repeat(width - n)])  # causal: pad changes nothing
        pos = torch.arange(width)
        batch = recipe.target_wrapper.generate_batch(
            input_ids=ids[None].to(recipe.device),
            attention_mask=torch.ones(1, width, dtype=torch.long, device=recipe.device),
            loss_mask=torch.ones(1, width, dtype=torch.long, device=recipe.device),
            position_ids=pos[None].to(recipe.device),
            seq_lens=torch.tensor([[width]], device=recipe.device),
            doc_remaining=(width - pos)[None].to(recipe.device),
        )
        features.append(batch.target_hidden_states[0, :n].to("cpu", torch.bfloat16))
        last.append(batch.target_last_hidden_states[0, :n].to("cpu", torch.bfloat16))
        if dist.get_rank() == 0:
            print(f"{i}: {n} tokens, features {tuple(features[-1].shape)}", flush=True)
    if dist.get_rank() == 0:
        torch.save(features, out)
        torch.save(last, out.removesuffix(".pt") + ".last.pt")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
