# Zonos2 TTS ComfyUI TODO

## Completed

- [x] Inspect the official Zyphra/ZONOS2 repository and model release.
- [x] Inspect ComfyUI node/package conventions in OmniVoice, WavTTS, and Higgs v3.
- [x] Convert and validate the official BF16 `model.pth` as `zonos2-bf16.safetensors`.
- [x] Initialize this repository on `main` for `Saganaki22/Zonos2_TTS-ComfyUI`.
- [x] Define the managed model folder as `ComfyUI/models/zonos2`.
- [x] Define the loader dtype order as `auto`, `bf16`, `fp16`.
- [x] Define the attention order as `auto`, `SDPA`, `flash_attention`.
- [x] Confirm ZONOS2 voice cloning uses audio-only 2048D speaker embeddings.

## In Progress

- [x] Implement the native PyTorch ZONOS2 model with checkpoint-compatible names.
- [x] Validate the native model layout against all 507 checkpoint tensors.
- [x] Implement safetensors loading with `auto` checkpoint dtype and explicit casts.
- [x] Implement SDPA and FlashAttention inference with a native KV cache.
- [x] Implement Sonic MoE routing and expert execution.
- [x] Implement prompt conditioning, sampling, repetition penalty, and EOS alignment.
- [x] Implement DAC 44.1 kHz decoding.
- [x] Implement the Qwen3 ECAPA-TDNN speaker encoder for audio-only voice cloning.
- [x] Expose all released ZONOS2 speaking-rate and quality buckets.
- [x] Add practical tooltips to every node input and output.
- [x] Limit voice-clone references to 60 seconds with CLI warnings and clipping.
- [x] Fix behavioral issues found by full model inference testing.

## ComfyUI Integration

- [x] Add the ZONOS2 model loader node with managed model discovery/download.
- [x] Add the ZONOS2 voice generation node.
- [x] Add the ZONOS2 voice clone node with model and native `AUDIO` inputs.
- [x] Return native ComfyUI `AUDIO`.
- [x] Add native ComfyUI progress bars.
- [x] Add in-place CLI `tqdm` progress for loading and generation.
- [x] Register real model tensors with ComfyUI model management and AIMDO.
- [x] Resume ComfyUI-offloaded modules before generation.
- [x] Hard-unload and unregister the active bundle when model, dtype, or attention changes.
- [x] Clear AIMDO state, move hard-unloaded weights to `meta`, run GC, and empty caches.
- [x] Verify AIMDO registration, residency bars, CPU offload, and GPU resume.

## Packaging And Verification

- [x] Add `pyproject.toml` and `install.py`.
- [x] Bundle the official `assets/params.json` configuration.
- [x] Add English and Chinese READMEs, license, and example workflow.
- [x] Document installation, model assets, controls, licenses, compatibility, and troubleshooting.
- [x] Run package import and node registration tests.
- [x] Run model-layout and checkpoint-key tests: 507/507 matched.
- [x] Run a full BF16 GPU load smoke test.
- [x] Run short generation and audio-noodle voice-clone smoke tests.
- [x] Verify finite 44.1 kHz native ComfyUI `AUDIO` output.
- [x] Verify SDPA and FlashAttention output paths.
- [x] Verify explicit FP16 checkpoint casting.
- [x] Verify changing model, dtype, and attention fully releases old VRAM.
- [x] Verify hard unload removes AIMDO registrations and drops CUDA allocation from 14.71 GiB to 0.009 GiB.
- [x] Run automated tests: 7 passed.
- [x] Stage the completed files on `main`.
