import gc
import weakref
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn

from zonos2_tts_comfyui_test import loader, nodes
from zonos2_tts_comfyui_test.native import Zonos2Model, read_config
from zonos2_tts_comfyui_test.runtime import Zonos2SpeakerEncoder


ROOT = Path(__file__).resolve().parents[1]


class _FakeSpeakerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))


def test_loader_progress_uses_reported_checkpoint_total(monkeypatch):
    events = []

    class FakeProgressBar:
        def __init__(self, total):
            events.append(("init", total))

        def update_absolute(self, current, total):
            events.append(("update", current, total))

    monkeypatch.setattr(nodes, "ProgressBar", FakeProgressBar)
    callback = nodes._progress_callback()
    callback(1, 23)

    assert events == [("init", 23), ("update", 1, 23)]


def test_resampler_cache_does_not_retain_encoder():
    encoder = Zonos2SpeakerEncoder(_FakeSpeakerModel())
    encoder._resampler(44_100)
    reference = weakref.ref(encoder)

    del encoder
    gc.collect()

    assert reference() is None


def test_cached_resampler_follows_encoder_device_moves():
    encoder = Zonos2SpeakerEncoder(_FakeSpeakerModel())
    resampler = encoder._resampler(44_100)
    assert resampler.kernel.device.type == "cpu"

    encoder.to("meta")

    assert resampler.kernel.device.type == "meta"


def test_speaker_encoder_device_can_be_managed_by_comfy():
    encoder = Zonos2SpeakerEncoder(_FakeSpeakerModel())

    encoder.device = torch.device("cuda")

    assert encoder.device == torch.device("cuda")


def test_resume_registers_with_comfy_memory_manager(monkeypatch):
    calls = []
    patcher = object()

    monkeypatch.setattr(loader, "_register_with_comfy", calls.append)

    loader.resume_runtime_module(patcher, torch.device("cuda"))

    assert calls == [patcher]


def test_dynamic_patcher_sees_each_moe_expert_as_separate_modules():
    from comfy.model_patcher import ModelPatcher

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
    model = Zonos2Model(config)
    patcher = ModelPatcher(
        model,
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
    )
    names = {
        item[3]
        for item in patcher._load_list(
            for_dynamic=True,
            default_device=None,
        )
    }

    assert "layers.1.feed_forward.experts.experts.0" in names
    assert "layers.1.feed_forward.experts.experts.3" in names


def test_high_vram_uses_static_model_path(monkeypatch):
    config = read_config(ROOT / "assets" / "params.json")
    model = Zonos2Model(config)

    monkeypatch.setattr(loader, "dynamic_vram_active", lambda device: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: type("Props", (), {"total_memory": 32 * 1024**3})(),
    )

    assert not loader.should_use_dynamic_vram(
        model,
        torch.device("cuda"),
        torch.bfloat16,
    )


def test_low_vram_keeps_dynamic_model_path(monkeypatch):
    config = read_config(ROOT / "assets" / "params.json")
    model = Zonos2Model(config)

    monkeypatch.setattr(loader, "dynamic_vram_active", lambda device: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: type("Props", (), {"total_memory": 16 * 1024**3})(),
    )

    assert loader.should_use_dynamic_vram(
        model,
        torch.device("cuda"),
        torch.bfloat16,
    )
