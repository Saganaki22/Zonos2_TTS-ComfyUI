from dataclasses import replace
from pathlib import Path

import torch

from zonos2_tts_comfyui_test.loader import inspect_checkpoint_dtype
from zonos2_tts_comfyui_test.native import (
    SonicExperts,
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


def test_single_token_expert_fast_path_matches_grouped_top1():
    config = replace(
        read_config(ROOT / "assets" / "params.json"),
        dim=8,
        intermediate_size=12,
        moe_n_experts=4,
    )
    experts = SonicExperts(config)
    torch.manual_seed(7)
    experts.w13.data.normal_(mean=0.0, std=0.01)
    experts.w2.data.normal_(mean=0.0, std=0.01)
    hidden = torch.randn(1, config.dim)
    weights = torch.tensor([[0.625]])
    ids = torch.tensor([[3]])

    expected = experts._forward_grouped(hidden, weights, ids)
    actual = experts(hidden, weights, ids)

    torch.testing.assert_close(actual, expected)


def test_single_token_expert_fast_path_matches_grouped_top2():
    config = replace(
        read_config(ROOT / "assets" / "params.json"),
        dim=8,
        intermediate_size=12,
        moe_n_experts=4,
    )
    experts = SonicExperts(config)
    torch.manual_seed(11)
    experts.w13.data.normal_(mean=0.0, std=0.01)
    experts.w2.data.normal_(mean=0.0, std=0.01)
    hidden = torch.randn(1, config.dim)
    weights = torch.tensor([[0.55, 0.35]])
    ids = torch.tensor([[3, 1]])

    expected = experts._forward_grouped(hidden, weights, ids)
    actual = experts(hidden, weights, ids)

    torch.testing.assert_close(actual, expected)


def test_local_checkpoint_dtype_when_available():
    checkpoint = ROOT.parents[1] / "models" / "zonos2" / "zonos2-bf16.safetensors"
    if not checkpoint.is_file():
        return
    assert inspect_checkpoint_dtype(checkpoint) == torch.bfloat16
