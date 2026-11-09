"""Build NeMo AutoModel's native DeepSeek V4.1 DSpark draft from the served checkpoint's config."""
import json
import pathlib

import sys

import torch

from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from paths import CKPT  # noqa: E402


class Args(dict):
    __getattr__ = dict.__getitem__


def build(ckpt: pathlib.Path, num_anchors: int = 256):
    raw = json.loads((ckpt / "config.json").read_text())
    raw.pop("quantization_config", None)
    raw["text_config"].pop("quantization_config", None)
    config = DeepseekV41Config.from_dict(raw)
    tc = config.text_config
    return config.build_dspark_draft(Args(
        num_draft_layers=tc.num_nextn_predict_layers, target_layer_ids=list(tc.dspark_target_layer_ids),
        block_size=tc.dspark_block_size, num_anchors=num_anchors, mask_token_id=tc.dspark_noise_token_id,
        markov_rank=tc.dspark_markov_rank, markov_head_type="vanilla", confidence_head_alpha=1.0,
        confidence_head_with_markov=True, confidence_head_stop_gradient=False))


if __name__ == "__main__":
    with torch.device("meta"):
        m = build(CKPT)
    n = sum(p.numel() for p in m.parameters())
    print(type(m).__name__, f"{n / 1e9:.2f}B params; trainable "
          f"{sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e9:.2f}B")
    print([k for k in m.state_dict() if not k.startswith("mtp.")][:10])
    print(sorted({k.split(".", 3)[-1] for k in m.state_dict() if k.startswith("mtp.2.")})[:60])
