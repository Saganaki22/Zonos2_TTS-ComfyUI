import torch
import torch.nn as nn

from zonos2_tts_comfyui_test.loader import ATTENTION_OPTIONS, DTYPE_OPTIONS
from zonos2_tts_comfyui_test.nodes import (
    QUALITY_BUCKET_LABELS,
    Zonos2VoiceClone,
    Zonos2VoiceGeneration,
)
from zonos2_tts_comfyui_test.runtime import (
    MAX_REFERENCE_SECONDS,
    Zonos2SpeakerEncoder,
    logger,
)


def test_loader_option_order():
    assert DTYPE_OPTIONS == ["auto", "bf16", "fp16"]
    assert ATTENTION_OPTIONS == ["auto", "SDPA", "flash_attention"]


def test_clone_has_native_audio_input():
    required = Zonos2VoiceClone.INPUT_TYPES()["required"]
    assert required["reference_audio"][0] == "AUDIO"
    assert "reference_text" not in required


def test_both_generation_nodes_expose_all_quality_features():
    for node in (Zonos2VoiceGeneration, Zonos2VoiceClone):
        required = node.INPUT_TYPES()["required"]
        for control in QUALITY_BUCKET_LABELS:
            assert control in required


def test_reference_audio_over_limit_is_clipped_with_cli_warning(monkeypatch):
    encoder = Zonos2SpeakerEncoder(nn.Linear(1, 1))
    sample_rate = 24_000
    waveform = torch.zeros(1, 1, sample_rate * 61)
    warnings = []
    monkeypatch.setattr(
        logger,
        "warning",
        lambda message, *args: warnings.append(message % args),
    )

    prepared = encoder._prepare_audio(waveform, sample_rate)

    assert prepared.shape[-1] == int(sample_rate * MAX_REFERENCE_SECONDS)
    assert "Clipping to the first 60.0 seconds" in warnings[0]
