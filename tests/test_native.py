from pathlib import Path

import torch

from zonos2_tts_comfyui_test.loader import inspect_checkpoint_dtype
from zonos2_tts_comfyui_test.native import (
    build_native_model,
    build_prompt,
    read_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_released_model_layout_size():
    config = read_config(ROOT / "assets" / "params.json")
    model = build_native_model(config)
    assert len(model.state_dict()) == 507
    assert sum(parameter.numel() for parameter in model.parameters()) == 7_668_118_208


def test_conditioning_rows_follow_upstream_order():
    config = read_config(ROOT / "assets" / "params.json")
    prompt, speaker_position = build_prompt(
        config,
        "hi",
        speaking_rate_bucket=3,
        quality_buckets=[None, 1, None, None, None, 4],
    )
    assert speaker_position is None
    assert prompt[0, :4, -1].tolist() == [451, 469, 512, 2]


def test_local_checkpoint_dtype_when_available():
    checkpoint = ROOT.parents[1] / "models" / "zonos2" / "zonos2-bf16.safetensors"
    if not checkpoint.is_file():
        return
    assert inspect_checkpoint_dtype(checkpoint) == torch.bfloat16
