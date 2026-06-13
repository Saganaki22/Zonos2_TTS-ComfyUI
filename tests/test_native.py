from dataclasses import replace
from pathlib import Path

import torch

from zonos2_tts_comfyui_test.loader import inspect_checkpoint_dtype
from zonos2_tts_comfyui_test.native import (
    SonicExperts,
    build_native_model,
    build_prompt,
    checkpoint_layout,
    read_config,
    validate_checkpoint_layout,
)


ROOT = Path(__file__).resolve().parents[1]


def test_released_model_layout_size():
    config = read_config(ROOT / "assets" / "params.json")
    model = build_native_model(config)
    assert len(checkpoint_layout(model)) == 507
    assert sum(parameter.numel() for parameter in model.parameters()) == 7_668_118_208


def test_model_builds_with_aimdo_lazy_linear_initialization(monkeypatch):
    import comfy.memory_management

    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", True)
    model = build_native_model(read_config(ROOT / "assets" / "params.json"))

    assert model.layers[0].attention.wq.weight is not None
    assert model.layers[0].attention.wkv.weight is not None
    assert model.speaker_projection.weight is not None
    assert model.multi_output.weight is not None
    assert all(
        parameter.device.type == "meta"
        for parameter in model.parameters()
    )


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
    for expert in experts.experts:
        expert.weight.data.normal_(mean=0.0, std=0.01)
        expert.bias.data.normal_(mean=0.0, std=0.01)
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
    for expert in experts.experts:
        expert.weight.data.normal_(mean=0.0, std=0.01)
        expert.bias.data.normal_(mean=0.0, std=0.01)
    hidden = torch.randn(1, config.dim)
    weights = torch.tensor([[0.55, 0.35]])
    ids = torch.tensor([[3, 1]])

    expected = experts._forward_grouped(hidden, weights, ids)
    actual = experts(hidden, weights, ids)

    torch.testing.assert_close(actual, expected)


def test_moe_experts_are_independently_pageable():
    config = replace(
        read_config(ROOT / "assets" / "params.json"),
        dim=8,
        intermediate_size=12,
        moe_n_experts=4,
    )
    experts = SonicExperts(config)

    assert len(experts.experts) == 4
    for expert in experts.experts:
        assert expert.comfy_cast_weights is True
        assert expert.w13_shape == (24, 8)
        assert expert.w2_shape == (8, 12)
        assert expert.weight.shape == (24, 8)
        assert expert.bias.shape == (8, 12)


def test_shared_token_path_stays_resident():
    config = replace(
        read_config(ROOT / "assets" / "params.json"),
        n_layers=5,
        dim=8,
        head_dim=4,
        n_heads=2,
        n_kv_heads=1,
        intermediate_size=12,
        speaker_embedding_dim=8,
        speaker_lda_dim=4,
        moe_n_experts=4,
        moe_router_dim=4,
        moe_start_from_layer=1,
        moe_end_from_layer=1,
    )
    model = build_native_model(config)
    layer = model.layers[1]

    assert not getattr(layer.attention.wq, "comfy_cast_weights", False)
    assert not getattr(layer.attention.wo, "comfy_cast_weights", False)
    assert not getattr(
        layer.feed_forward.router.down_proj,
        "comfy_cast_weights",
        False,
    )
    assert not getattr(model.multi_output, "comfy_cast_weights", False)


def test_local_checkpoint_dtype_when_available():
    checkpoint = ROOT.parents[1] / "models" / "zonos2" / "zonos2-bf16.safetensors"
    if not checkpoint.is_file():
        return
    assert inspect_checkpoint_dtype(checkpoint) == torch.bfloat16


def test_local_checkpoint_matches_pageable_runtime_layout_when_available():
    checkpoint = ROOT.parents[1] / "models" / "zonos2" / "zonos2-bf16.safetensors"
    if not checkpoint.is_file():
        return
    model = build_native_model(read_config(ROOT / "assets" / "params.json"))
    count, missing, unexpected = validate_checkpoint_layout(model, checkpoint)

    assert count == 507
    assert missing == set()
    assert unexpected == set()
