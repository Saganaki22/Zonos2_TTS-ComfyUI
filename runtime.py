"""Audio codec, speaker embedding, and end-to-end ZONOS2 inference."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from .loader import (
    Zonos2Bundle,
    add_bundle_module,
    model_dir,
    resume_bundle_to_device,
)
from .native import (
    SamplingOptions,
    build_prompt,
    generate_audio_codes,
    shear_up,
)

logger = logging.getLogger("Zonos2_TTS-ComfyUI")

ASSET_REPO_ID = "drbaph/ZONOS2-BF16"
DAC_SAMPLE_RATE = 44_100
SPEAKER_SAMPLE_RATE = 24_000
MAX_REFERENCE_SECONDS = 60.0
RECOMMENDED_REFERENCE_MIN_SECONDS = 5.0
RECOMMENDED_REFERENCE_MAX_SECONDS = 30.0


def _require_or_download(
    repo_id: str,
    local_dir: Path,
    allow_download: bool,
    repo_subdir: str,
    required_files: tuple[str, ...],
) -> Path:
    missing_files = [
        filename
        for filename in required_files
        if not (local_dir / filename).is_file()
    ]
    if not missing_files:
        return local_dir
    if not allow_download:
        raise FileNotFoundError(
            f"Required model assets are missing from {local_dir}: "
            f"{', '.join(missing_files)}. Enable download_if_missing in the "
            "ZONOS2 loader."
        )
    from huggingface_hub import snapshot_download

    logger.info(
        "Downloading %s/%s to %s.",
        repo_id,
        repo_subdir,
        local_dir,
    )
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{repo_subdir}/**"],
        local_dir=str(model_dir()),
    )
    missing_files = [
        filename
        for filename in required_files
        if not (local_dir / filename).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"{repo_id} does not currently contain all required files under "
            f"{repo_subdir}/. Missing: {', '.join(missing_files)}."
        )
    return local_dir


class Zonos2DAC(nn.Module):
    sample_rate = DAC_SAMPLE_RATE

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.eval()
        self.requires_grad_(False)

    @torch.inference_mode()
    def decode(
        self,
        delayed_codes: torch.Tensor,
        pad_id: int,
        eos_frame: int | None,
    ) -> torch.Tensor:
        device = next(self.model.parameters()).device
        codes = shear_up(delayed_codes.to(device=device, dtype=torch.long), pad_id)
        if eos_frame is not None:
            codes = codes[: max(0, int(eos_frame))]
        else:
            complete = max(0, codes.shape[0] - (codes.shape[1] - 1))
            codes = codes[:complete]
        if codes.numel() == 0:
            raise RuntimeError("ZONOS2 ended before producing decodable audio.")
        codes = codes.clamp_(0, 1023).transpose(0, 1).unsqueeze(0)
        output = self.model.decode(audio_codes=codes).audio_values
        if output.ndim == 3:
            output = output.mean(dim=1)
        return output.float().cpu()


class Zonos2SpeakerEncoder(nn.Module):
    target_sample_rate = SPEAKER_SAMPLE_RATE

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.eval()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            f_min=0.0,
            f_max=12_000.0,
            n_mels=128,
            power=1.0,
            center=False,
            norm="slaney",
            mel_scale="slaney",
        )
        self.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @lru_cache(maxsize=8)
    def _resampler(self, source_sample_rate: int):
        return torchaudio.transforms.Resample(
            int(source_sample_rate),
            self.target_sample_rate,
        ).to(self.device)

    def _prepare_audio(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        audio = waveform
        if audio.ndim == 3:
            audio = audio[0]
        if audio.ndim == 2:
            audio = audio.mean(dim=0, keepdim=True)
        elif audio.ndim == 1:
            audio = audio.unsqueeze(0)
        else:
            raise ValueError(
                f"Reference AUDIO waveform must be 1D, 2D, or 3D, got {audio.ndim}D."
            )

        source_sample_rate = int(sample_rate)
        if source_sample_rate <= 0:
            raise ValueError(
                f"Reference AUDIO sample_rate must be positive, got {sample_rate}."
            )
        duration_seconds = audio.shape[-1] / source_sample_rate
        if duration_seconds > MAX_REFERENCE_SECONDS:
            maximum_samples = round(MAX_REFERENCE_SECONDS * source_sample_rate)
            logger.warning(
                "Reference audio is %.2f seconds; the node accepts at most %.1f "
                "seconds. Clipping to the first %.1f seconds.",
                duration_seconds,
                MAX_REFERENCE_SECONDS,
                MAX_REFERENCE_SECONDS,
            )
            audio = audio[..., :maximum_samples]
        elif duration_seconds < RECOMMENDED_REFERENCE_MIN_SECONDS:
            logger.warning(
                "Reference audio is only %.2f seconds. %.0f-%.0f seconds of clean, "
                "single-speaker speech is recommended for reliable voice cloning.",
                duration_seconds,
                RECOMMENDED_REFERENCE_MIN_SECONDS,
                RECOMMENDED_REFERENCE_MAX_SECONDS,
            )
        else:
            logger.info(
                "Using %.2f seconds of reference audio for voice cloning.",
                duration_seconds,
            )

        audio = audio.to(device=self.device, dtype=torch.float32)
        if source_sample_rate != self.target_sample_rate:
            audio = self._resampler(source_sample_rate)(audio)
        if audio.shape[-1] < 1024:
            audio = F.pad(audio, (0, 1024 - audio.shape[-1]))
        return audio

    @torch.inference_mode()
    def forward(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        audio = self._prepare_audio(waveform, sample_rate)
        padding = (1024 - 256) // 2
        audio = F.pad(audio.unsqueeze(1), (padding, padding), mode="reflect")
        audio = audio.squeeze(1)
        mel = self.mel_transform.to(self.device)(audio)
        mel = torch.log(torch.clamp(mel, min=1e-5)).transpose(1, 2)
        embedding = self.model(input_values=mel).last_hidden_state
        return embedding.to(dtype=torch.float32)


def ensure_codec(bundle: Zonos2Bundle) -> Zonos2DAC:
    if bundle.codec is not None:
        return bundle.codec
    local_dir = _require_or_download(
        ASSET_REPO_ID,
        model_dir() / "dac_44khz",
        bundle.download_if_missing,
        "dac_44khz",
        ("config.json", "model.safetensors"),
    )
    from transformers import DacModel

    model = DacModel.from_pretrained(
        str(local_dir),
        local_files_only=True,
    )
    codec = Zonos2DAC(model).to(bundle.device)
    add_bundle_module(bundle, codec)
    bundle.codec = codec
    logger.info("Loaded DAC 44.1 kHz decoder from %s.", local_dir)
    return codec


def ensure_speaker_encoder(bundle: Zonos2Bundle) -> Zonos2SpeakerEncoder:
    if bundle.speaker_encoder is not None:
        return bundle.speaker_encoder
    local_dir = _require_or_download(
        ASSET_REPO_ID,
        model_dir() / "speaker_encoder",
        bundle.download_if_missing,
        "speaker_encoder",
        (
            "config.json",
            "model.safetensors",
            "configuration_ecapa_tdnn.py",
            "modeling_ecapa_tdnn.py",
        ),
    )
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        str(local_dir),
        trust_remote_code=True,
        local_files_only=True,
    )
    encoder = Zonos2SpeakerEncoder(model).to(bundle.device)
    add_bundle_module(bundle, encoder)
    bundle.speaker_encoder = encoder
    logger.info("Loaded ZONOS2 speaker encoder from %s.", local_dir)
    return encoder


def extract_speaker_embedding(
    bundle: Zonos2Bundle,
    reference_audio: dict,
) -> torch.Tensor:
    if not isinstance(reference_audio, dict):
        raise TypeError("reference_audio must be a native ComfyUI AUDIO value.")
    waveform = reference_audio.get("waveform")
    sample_rate = reference_audio.get("sample_rate")
    if not isinstance(waveform, torch.Tensor) or sample_rate is None:
        raise ValueError(
            "reference_audio must contain waveform and sample_rate."
        )
    encoder = ensure_speaker_encoder(bundle)
    resume_bundle_to_device(bundle)
    embedding = encoder(waveform, int(sample_rate))
    if embedding.shape != (1, bundle.config.speaker_embedding_dim):
        raise RuntimeError(
            "Speaker encoder returned "
            f"{tuple(embedding.shape)}, expected "
            f"(1, {bundle.config.speaker_embedding_dim})."
        )
    return embedding


def generate_zonos2_audio(
    bundle: Zonos2Bundle,
    text: str,
    options: SamplingOptions,
    speaking_rate_bucket: int,
    quality_buckets: list[int | None] | tuple[int | None, ...],
    reference_audio: dict | None = None,
    clean_speaker_background: bool = False,
    accurate_mode: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    if bundle.model is None:
        raise RuntimeError("The ZONOS2 bundle has been unloaded.")
    if not text.strip():
        raise ValueError("Text cannot be empty.")

    resume_bundle_to_device(bundle)
    codec = ensure_codec(bundle)
    speaker_embedding = None
    if reference_audio is not None:
        speaker_embedding = extract_speaker_embedding(bundle, reference_audio)

    prompt, speaker_position = build_prompt(
        bundle.config,
        text=text,
        speaking_rate_bucket=int(speaking_rate_bucket),
        quality_buckets=quality_buckets,
        speaker_embedding=speaker_embedding,
        clean_speaker_background=bool(clean_speaker_background),
        accurate_mode=bool(accurate_mode),
    )
    delayed_codes, eos_frame = generate_audio_codes(
        bundle.model,
        prompt,
        attention_backend=bundle.attention,
        options=options,
        speaker_embedding=speaker_embedding,
        speaker_position=speaker_position,
        progress_callback=progress_callback,
    )
    audio = codec.decode(
        delayed_codes,
        pad_id=bundle.config.audio_pad_id,
        eos_frame=eos_frame,
    )
    if progress_callback is not None:
        progress_callback(int(options.max_new_tokens), int(options.max_new_tokens))
    waveform = audio.unsqueeze(1) if audio.ndim == 2 else audio
    return {
        "waveform": waveform.contiguous(),
        "sample_rate": codec.sample_rate,
    }
