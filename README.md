# ZONOS2 TTS ComfyUI

[![Version](https://img.shields.io/badge/version-0.1.1-blue)](https://github.com/Saganaki22/Zonos2_TTS-ComfyUI)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom_Node-2d7dd2)](https://github.com/comfyanonymous/ComfyUI)
[![Upstream](https://img.shields.io/badge/Upstream-Zyphra%2FZONOS2-111111)](https://github.com/Zyphra/ZONOS2)
[![Official Model](https://img.shields.io/badge/Hugging_Face-Zyphra%2FZONOS2-ffd21e)](https://huggingface.co/Zyphra/ZONOS2)
[![Native BF16 Model](https://img.shields.io/badge/Hugging_Face-drbaph%2FZONOS2--BF16-ffd21e)](https://huggingface.co/drbaph/ZONOS2-BF16)
[![Model License](https://img.shields.io/badge/Model_License-Apache--2.0-green)](https://huggingface.co/Zyphra/ZONOS2)

[简体中文](README_zh.md)

<img width="1444" height="1203" alt="Screenshot 2026-06-12 214924" src="https://github.com/user-attachments/assets/a776ed13-a106-476f-b5fa-e55402365bb8" />


Native ComfyUI nodes for [Zyphra/ZONOS2](https://github.com/Zyphra/ZONOS2), with text-to-speech, audio-only voice cloning, SDPA and FlashAttention inference, native progress reporting, and ComfyUI/AIMDO memory management.

ZONOS2 is our latest text-to-speech model trained on more than 6 million hours of varied multilingual speech, delivering expressiveness and quality on par with—or even surpassing—top TTS providers at low latency with MoE. ZONOS2 excels at high-fidelity and naturalistic voice cloning.

During inference we use nemo TN normalized UTF-8 bytes and an ECAPA-TDNN embedding to generate DAC tokens with our MoE backbone. An inference overview can be seen below.

![ZONOS2 inference overview](https://huggingface.co/Zyphra/ZONOS2/resolve/main/assets/zonos2_arlooop_animated.gif)

> [!WARNING]
> Only clone voices you own or have explicit permission to use. Malicious impersonation, fraud, deception, harassment, abuse, evasion of consent, or any use intended to cause harm is strictly forbidden by this project's acceptable-use policy.

**Highlights**

- Native in-process PyTorch implementation for Windows and Linux.
- Standard ComfyUI `AUDIO` input and 44.1 kHz `AUDIO` output.
- Audio-only zero-shot voice cloning; no reference transcript is required.
- All released ZONOS2 speaking-rate and quality-conditioning buckets.
- `auto`, `bf16`, and `fp16` runtime dtypes.
- Automatic FlashAttention selection with PyTorch SDPA fallback.
- Native ComfyUI progress bars and in-place CLI `tqdm` progress.
- Real tensor registration with ComfyUI model management and AIMDO.
- Full cleanup and reload when model, dtype, or attention settings change.

<details>
<summary><strong>Installation</strong></summary>

### ComfyUI Manager

Once listed in ComfyUI Manager, search for **ZONOS2 TTS** and install it normally.

To install it immediately through Manager:

1. Open **Manager**.
2. Choose **Install via Git URL**.
3. Enter `https://github.com/Saganaki22/Zonos2_TTS-ComfyUI`.
4. Restart ComfyUI.

### Manual installation

From the `ComfyUI/custom_nodes` directory:

```powershell
git clone https://github.com/Saganaki22/Zonos2_TTS-ComfyUI.git
..\venv\Scripts\python.exe Zonos2_TTS-ComfyUI\install.py
```

Linux portable or venv installations can run:

```bash
git clone https://github.com/Saganaki22/Zonos2_TTS-ComfyUI.git
../venv/bin/python Zonos2_TTS-ComfyUI/install.py
```

Restart ComfyUI after installing or updating. `install.py` installs only missing lightweight dependencies and does not replace ComfyUI's `torch`, `torchaudio`, or `transformers`.

</details>

<details>
<summary><strong>Models and automatic downloads</strong></summary>

Models are managed under:

```text
ComfyUI/models/zonos2/
├── zonos2-bf16.safetensors
├── dac_44khz/
│   ├── .gitattributes
│   ├── config.json
│   ├── model.safetensors
│   ├── preprocessor_config.json
│   └── README.md
└── speaker_encoder/
    ├── config.json
    ├── configuration_ecapa_tdnn.py
    ├── feature_extraction_ecapa_tdnn.py
    ├── model.safetensors
    ├── modeling_ecapa_tdnn.py
    ├── preprocessor_config.json
    ├── tokenizer_config.json
    └── tokenizer_ecapa_tdnn.py
```

The preconfigured loader downloads all three assets from [drbaph/ZONOS2-BF16](https://huggingface.co/drbaph/ZONOS2-BF16):

- `zonos2-bf16.safetensors` is downloaded when the preset is selected.
- `dac_44khz/` is downloaded when the model bundle is first loaded.
- `speaker_encoder/` is downloaded lazily when voice cloning is first used.

Enable `download_if_missing` to allow these downloads. Disable it for fully offline operation after placing every required asset in the paths above.

The custom node ships the official architecture configuration at `assets/params.json` and always validates the checkpoint names and shapes against it. Locally added `.safetensors` files also appear in the model dropdown.

Upload the complete `dac_44khz/` and `speaker_encoder/` directories to the model repository rather than reconstructing them from the abbreviated tree. At runtime, the DAC directly requires `config.json` and `model.safetensors`. The speaker encoder directly requires `config.json`, `model.safetensors`, `configuration_ecapa_tdnn.py`, and `modeling_ecapa_tdnn.py` because it is loaded through Transformers remote-code support. The preprocessor, feature-extractor, tokenizer, README, and Git metadata files are not directly used by this node's current inference path, but retaining the complete upstream folders preserves a valid, reusable Hugging Face package.

</details>

<details>
<summary><strong>Nodes and settings</strong></summary>

### ZONOS2 Model Loader

| Input | Default | Options | Description |
|---|---|---|---|
| `model` | ZONOS2 BF16 preset | Presets and local safetensors | Loads from `ComfyUI/models/zonos2`. |
| `dtype` | `auto` | `auto`, `bf16`, `fp16` | `auto` preserves the checkpoint dtype. Explicit values cast floating tensors while loading. |
| `attention` | `auto` | `auto`, `SDPA`, `flash_attention` | `auto` uses FlashAttention when installed and compatible, otherwise SDPA. |
| `download_if_missing` | `true` | boolean | Downloads missing checkpoint, DAC, and speaker encoder assets. |

Changing the model, dtype, or attention backend intentionally hard-unloads the previous bundle before loading the replacement.

### ZONOS2 Voice Generation

Accepts `zonos2_model` and UTF-8 text, then returns mono ComfyUI `AUDIO` at 44.1 kHz.

### ZONOS2 Voice Clone

Accepts `zonos2_model`, UTF-8 text, and a required native ComfyUI `reference_audio` noodle. ZONOS2 extracts its official 2048-dimensional ECAPA-TDNN speaker embedding directly from the audio; reference text is not required.

| Clone setting | Default | Description |
|---|---:|---|
| `reference_audio` | required | Use 5–30 seconds of clean, single-speaker speech. The node accepts at most 60 seconds and clips longer input with a CLI warning. |
| `clean_speaker_background` | `true` | Enable for clean speech; disable when the recording has noticeable room noise or ambience. |
| `accurate_mode` | `true` | Favors closer speaker matching. Disable for looser, potentially more expressive conditioning. |

### Sampling controls

| Control | Default | Range | Description |
|---|---:|---:|---|
| `max_new_tokens` | 1024 | 32–6000, step 8 | Maximum generated DAC-code frames. Higher values allow longer speech and use more time and KV-cache memory. |
| `temperature` | 1.15 | 0–2 | Sampling randomness; `0` is greedy. |
| `top_k` | 106 | 0–1026 | Keeps the K most likely tokens; `0` disables it. |
| `top_p` | 0.0 | 0–1 | Nucleus sampling threshold; `0` disables it. |
| `min_p` | 0.18 | 0–1 | Removes tokens below a fraction of the most likely token; `0` disables it. |
| `repetition_window` | 50 | 0–512 | Recent frames checked for repetition; `0` disables tracking. |
| `repetition_penalty` | 1.2 | 1–2 | Discourages repeated audio tokens; `1` disables the penalty. |
| `repetition_codebooks` | 8 | -1–9 | Number of codebooks receiving the penalty; `-1` means all and `0` means none. |
| `seed` | 0 | 0–9223372036854775807 | Positive values make identical runs repeatable; `0` uses the current random state. |

### ZONOS2 conditioning

Every conditioning dropdown includes `default`, which leaves that feature unconditioned.

| Control | Released buckets | Default |
|---|---|---|
| `speaking_rate` | 8 UTF-8 bytes-per-second ranges from `0–8` through `40+` | `default` |
| `loudness_lufs` | 12 ranges from below `-50` through `-5+` LUFS | `default` |
| `estimated_snr` | 12 ranges from below `0` through `60+` dB | `default` |
| `maximum_pause` | 12 ranges from `0–0.5` through `5.5–6` seconds | `default` |
| `estimated_bandlimit_hz` | 8 ranges from `495.3–3433` through `21062–24000` Hz | `default` |
| `leading_silence` | 8 ranges from `0–0.05` through `4+` seconds | `default` |
| `trailing_silence` | 8 ranges from `0–0.05` through `4+` seconds | `0.25–0.5` |

</details>

<details>
<summary><strong>Reference audio guidance</strong></summary>

- Recommended: 5–30 seconds of clean, uninterrupted speech from one speaker.
- Maximum accepted by this nodepack: 60 seconds.
- Longer input is clipped to the first 60 seconds before resampling and embedding.
- The ComfyUI CLI reports the supplied duration and clipping action.
- Very short references below 5 seconds also produce a recommendation warning.
- Avoid music, overlapping speakers, heavy denoising artifacts, strong reverb, and long silence.

The 60-second ceiling is a practical memory and latency guardrail in this integration, not a documented architectural limit of the upstream ZONOS2 model.

</details>

<details>
<summary><strong>Transformers compatibility</strong></summary>

The nodepack was tested end to end with **Transformers 5.3.0**. DAC decoding and speaker-encoder forward passes were also tested with:

`5.0.0`, `5.2.0`, `5.3.0`, `5.5.4`, and `5.12.0`

Supported and tested range: **`transformers>=5.0.0,<=5.12.0`**.

Transformers 4.x is not supported by this nodepack because common current ComfyUI environments conflict with the older NumPy and `huggingface-hub` constraints required by those releases. `install.py` reports whether the installed Transformers version is inside the tested range but never replaces it automatically.

</details>

<details>
<summary><strong>Memory management and progress</strong></summary>

The ZONOS2 model, DAC decoder, and lazy speaker encoder are registered as real PyTorch modules with ComfyUI/AIMDO.

- ComfyUI can offload the bundle to CPU and restore it for the next run.
- AIMDO bars reflect actual GPU tensor residency.
- Reusing identical loader settings resumes the existing bundle.
- Changing model, dtype, or attention unregisters the old bundle, moves its tensors to `meta`, clears references, runs garbage collection, and empties accelerator caches before loading the replacement.
- Loading and generation use native ComfyUI progress bars plus CLI `tqdm`.

Measured BF16 CUDA allocation with the main model and DAC loaded is approximately 14.7 GiB. Leave additional VRAM available for the KV cache, speaker encoder, ComfyUI, and other nodes.

</details>

<details>
<summary><strong>Troubleshooting</strong></summary>

**The model dropdown download fails or returns 404**

Confirm [drbaph/ZONOS2-BF16](https://huggingface.co/drbaph/ZONOS2-BF16) contains `zonos2-bf16.safetensors`, `dac_44khz/`, and `speaker_encoder/`. Partial uploads can fail until every required file is available.

**FlashAttention is unavailable**

Use `attention: auto` or `SDPA`. Auto falls back to SDPA when FlashAttention is not installed, the device is not CUDA, or the selected dtype is incompatible.

**CUDA out of memory**

Unload other large ComfyUI models, reduce `max_new_tokens`, use ComfyUI offloading, or restart ComfyUI after a failed allocation. ZONOS2 BF16 plus DAC uses roughly 14.7 GiB before generation cache growth.

**Voice cloning sounds weak or inaccurate**

Use 5–30 seconds of clean, single-speaker speech. Set `clean_speaker_background` to match the recording and leave `accurate_mode` enabled for closer identity matching.

**The reference is longer than 60 seconds**

The node clips it to the first 60 seconds and prints a warning such as: `Reference audio is 75.00 seconds; the node accepts at most 60.0 seconds. Clipping to the first 60.0 seconds.`

**Output stops too early**

Increase `max_new_tokens`. The model may still stop earlier when it emits end-of-audio.

**Transformers import or model-loading errors**

Check the startup output from `install.py` and use a release in the tested `5.0.0–5.12.0` range. Version `5.3.0` is the recommended baseline.

**A model, dtype, or attention change reloads everything**

This is expected. The old tensors are hard-unloaded to prevent stale AIMDO registrations and retained VRAM.

</details>

<details>
<summary><strong>Example workflow</strong></summary>

Load `example_workflows/zonos2_tts_example_workflow.json` in ComfyUI. It contains normal generation and voice-cloning branches. Select your own audio file in the `LoadAudio` node before running the clone branch.

</details>

<details>
<summary><strong>Licenses and responsible use</strong></summary>

- The original ZONOS2 model weights are released under the [Apache License 2.0](https://huggingface.co/Zyphra/ZONOS2).
- This ComfyUI integration code is released under the MIT License.
- DAC, speaker encoder, and other dependencies remain governed by their respective upstream licenses.

The Apache-2.0 model license and this project's acceptable-use policy are separate. Regardless of license permissions, this project must not be used for malicious impersonation, fraud, deception, harassment, non-consensual voice cloning, or causing harm.

</details>

<details>
<summary><strong>Citation</strong></summary>

If you find this model useful in an academic context, please cite:

```bibtex
@misc{zyphra2025zonos,
  title     = {Zonos V2 Technical Report},
  author    = {Gabriel Clark, Sofian Mejjoute, Mohamed Osman, George Close, Beren Millidge},
  year      = {2026},
}
```

</details>

## Credits

- [Zyphra/ZONOS2](https://github.com/Zyphra/ZONOS2)
- [Official Zyphra/ZONOS2 model release](https://huggingface.co/Zyphra/ZONOS2)
- [Native BF16 ComfyUI model package](https://huggingface.co/drbaph/ZONOS2-BF16)
- [Descript DAC 44.1 kHz](https://huggingface.co/descript/dac_44khz)
