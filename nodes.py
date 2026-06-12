"""ComfyUI node definitions for native ZONOS2 TTS."""

from __future__ import annotations

from .loader import (
    ATTENTION_OPTIONS,
    DTYPE_OPTIONS,
    get_model_choices,
    load_zonos2_bundle,
)
from .native import SamplingOptions
from .runtime import (
    MAX_REFERENCE_SECONDS,
    RECOMMENDED_REFERENCE_MAX_SECONDS,
    RECOMMENDED_REFERENCE_MIN_SECONDS,
    generate_zonos2_audio,
)

try:
    from comfy.utils import ProgressBar
except Exception:
    ProgressBar = None


MODEL_TYPE = "ZONOS2_MODEL"
SPEAKING_RATE_OPTIONS = [
    "default",
    "0: 0-8",
    "1: 8-11",
    "2: 11-14",
    "3: 14-17",
    "4: 17-21",
    "5: 21-28",
    "6: 28-40",
    "7: 40+",
]
QUALITY_BUCKET_LABELS = {
    "loudness_lufs": [
        "-1000--50",
        "-50--45.5",
        "-45.5--41",
        "-41--36.5",
        "-36.5--32",
        "-32--27.5",
        "-27.5--23",
        "-23--18.5",
        "-18.5--14",
        "-14--9.5",
        "-9.5--5",
        "-5+",
    ],
    "estimated_snr": [
        "-1000-0",
        "0-6",
        "6-12",
        "12-18",
        "18-24",
        "24-30",
        "30-36",
        "36-42",
        "42-48",
        "48-54",
        "54-60",
        "60+",
    ],
    "maximum_pause": [
        "0-0.5",
        "0.5-1",
        "1-1.5",
        "1.5-2",
        "2-2.5",
        "2.5-3",
        "3-3.5",
        "3.5-4",
        "4-4.5",
        "4.5-5",
        "5-5.5",
        "5.5-6",
    ],
    "estimated_bandlimit_hz": [
        "495.3-3433",
        "3433-6371",
        "6371-9310",
        "9310-12248",
        "12248-15186",
        "15186-18124",
        "18124-21062",
        "21062-24000",
    ],
    "leading_silence": [
        "0-0.05",
        "0.05-0.1",
        "0.1-0.25",
        "0.25-0.5",
        "0.5-1",
        "1-2",
        "2-4",
        "4+",
    ],
    "trailing_silence": [
        "0-0.05",
        "0.05-0.1",
        "0.1-0.25",
        "0.25-0.5",
        "0.5-1",
        "1-2",
        "2-4",
        "4+",
    ],
}


def _bucket_options(control: str) -> list[str]:
    return [
        "default",
        *[
            f"{index}: {label}"
            for index, label in enumerate(QUALITY_BUCKET_LABELS[control])
        ],
    ]


def _bucket(value: str) -> int:
    if value == "default":
        return -1
    return int(value.split(":", 1)[0])


def _quality_buckets(
    loudness_lufs: str,
    estimated_snr: str,
    maximum_pause: str,
    estimated_bandlimit_hz: str,
    leading_silence: str,
    trailing_silence: str,
) -> list[int | None]:
    values = [
        loudness_lufs,
        estimated_snr,
        maximum_pause,
        estimated_bandlimit_hz,
        leading_silence,
        trailing_silence,
    ]
    return [None if value == "default" else _bucket(value) for value in values]


def _text_input() -> tuple:
    return (
        "STRING",
        {
            "multiline": True,
            "default": "Hello! This is ZONOS2 running natively inside ComfyUI.",
            "tooltip": "UTF-8 text to synthesize.",
        },
    )


def _generation_controls() -> dict:
    return {
        "max_new_tokens": (
            "INT",
            {
                "default": 1024,
                "min": 32,
                "max": 6000,
                "step": 8,
                "tooltip": "Maximum DAC-code frames the model may generate. More frames allow longer speech but increase generation time and KV-cache memory. Generation can stop earlier when ZONOS2 emits end-of-audio.",
            },
        ),
        "temperature": (
            "FLOAT",
            {
                "default": 1.15,
                "min": 0.0,
                "max": 2.0,
                "step": 0.01,
                "tooltip": "Sampling randomness. Lower values are steadier; higher values add variation but can reduce clarity. 0 uses greedy sampling. The official ZONOS2 default is 1.15.",
            },
        ),
        "top_k": (
            "INT",
            {
                "default": 106,
                "min": 0,
                "max": 1026,
                "step": 1,
                "tooltip": "Keep only the K most likely tokens in each audio codebook before sampling. 0 disables Top-K filtering. The official default is 106.",
            },
        ),
        "top_p": (
            "FLOAT",
            {
                "default": 0.0,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "tooltip": "Keep the smallest token set whose combined probability reaches this value. 0 disables Top-P filtering. ZONOS2 normally uses Min-P instead.",
            },
        ),
        "min_p": (
            "FLOAT",
            {
                "default": 0.18,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "tooltip": "Remove tokens whose probability is below this fraction of the most likely token. 0 disables Min-P. The official default is 0.18.",
            },
        ),
        "repetition_window": (
            "INT",
            {
                "default": 50,
                "min": 0,
                "max": 512,
                "step": 1,
                "tooltip": "Number of recent generated frames checked for repeated audio tokens. 0 disables repetition tracking.",
            },
        ),
        "repetition_penalty": (
            "FLOAT",
            {
                "default": 1.2,
                "min": 1.0,
                "max": 2.0,
                "step": 0.01,
                "tooltip": "Reduces the probability of recently generated tokens to discourage loops. 1.0 disables the penalty. The official default is 1.2.",
            },
        ),
        "repetition_codebooks": (
            "INT",
            {
                "default": 8,
                "min": -1,
                "max": 9,
                "step": 1,
                "tooltip": "Apply repetition penalty to this many codebooks starting from codebook 0. -1 applies it to all 9; 0 disables it for every codebook. The official default is 8.",
            },
        ),
        "speaking_rate": (
            SPEAKING_RATE_OPTIONS,
            {
                "default": "default",
                "tooltip": "Optional ZONOS2 speaking-rate conditioning in cleaned UTF-8 bytes per second. Lower ranges generally produce slower speech. Default leaves speaking rate unconditioned.",
            },
        ),
        "loudness_lufs": (
            _bucket_options("loudness_lufs"),
            {
                "default": "default",
                "tooltip": "Optional target integrated loudness in LUFS. More-negative ranges are quieter; less-negative ranges are louder. Default leaves loudness unconditioned.",
            },
        ),
        "estimated_snr": (
            _bucket_options("estimated_snr"),
            {
                "default": "default",
                "tooltip": "Optional estimated signal-to-noise ratio in dB. Higher ranges bias toward cleaner audio; lower ranges can reproduce noisier recording characteristics. Default leaves SNR unconditioned.",
            },
        ),
        "maximum_pause": (
            _bucket_options("maximum_pause"),
            {
                "default": "default",
                "tooltip": "Optional maximum internal pause duration in seconds. Lower ranges favor tighter delivery; higher ranges permit longer pauses. Default leaves maximum pause unconditioned.",
            },
        ),
        "estimated_bandlimit_hz": (
            _bucket_options("estimated_bandlimit_hz"),
            {
                "default": "default",
                "tooltip": "Optional estimated recording bandlimit in Hz. Higher ranges bias toward wider-band, brighter audio; lower ranges can sound more bandwidth-limited. Default leaves bandlimit unconditioned.",
            },
        ),
        "leading_silence": (
            _bucket_options("leading_silence"),
            {
                "default": "default",
                "tooltip": "Optional amount of silence before speech begins. Default leaves leading silence unconditioned.",
            },
        ),
        "trailing_silence": (
            _bucket_options("trailing_silence"),
            {
                "default": "3: 0.25-0.5",
                "tooltip": "Requested silence after speech ends. Bucket 3 (0.25-0.5 seconds) is the official ZONOS2 default. Select default to leave it unconditioned.",
            },
        ),
        "seed": (
            "INT",
            {
                "default": 0,
                "min": 0,
                "max": 2**63 - 1,
                "tooltip": "Sampling seed. A positive value makes generation repeatable for identical inputs and settings. 0 uses the current random state.",
            },
        ),
    }


def _sampling_options(
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    min_p: float,
    repetition_window: int,
    repetition_penalty: float,
    repetition_codebooks: int,
    seed: int,
) -> SamplingOptions:
    return SamplingOptions(
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
        min_p=float(min_p),
        repetition_window=int(repetition_window),
        repetition_penalty=float(repetition_penalty),
        repetition_codebooks=int(repetition_codebooks),
        seed=int(seed),
    )


def _progress_callback(total: int):
    pbar = ProgressBar(total) if ProgressBar is not None else None

    def update(current: int, reported_total: int) -> None:
        if pbar is not None:
            pbar.update_absolute(current, reported_total)

    return update


class Zonos2ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    get_model_choices(),
                    {
                        "tooltip": "Checkpoint in ComfyUI/models/zonos2.",
                    },
                ),
                "dtype": (
                    DTYPE_OPTIONS,
                    {
                        "default": "auto",
                        "tooltip": "Runtime weight dtype. Auto preserves the checkpoint dtype; bf16 or fp16 casts floating tensors while loading. Changing this setting hard-unloads the previous bundle.",
                    },
                ),
                "attention": (
                    ATTENTION_OPTIONS,
                    {
                        "default": "auto",
                        "tooltip": "Attention backend. Auto selects FlashAttention on compatible CUDA BF16/FP16 systems and otherwise uses PyTorch SDPA. Changing this setting hard-unloads the previous bundle.",
                    },
                ),
                "download_if_missing": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When enabled, download a missing selected checkpoint, the DAC decoder, and the voice-clone speaker encoder from Hugging Face. When disabled, every required asset must already exist under ComfyUI/models/zonos2.",
                    },
                ),
            }
        }

    RETURN_TYPES = (MODEL_TYPE,)
    RETURN_NAMES = ("zonos2_model",)
    OUTPUT_TOOLTIPS = (
        "Loaded ZONOS2 model bundle containing the native language model and DAC decoder, managed by ComfyUI/AIMDO.",
    )
    FUNCTION = "load_model"
    CATEGORY = "ZONOS2 TTS"
    DESCRIPTION = (
        "Load native ZONOS2 weights with ComfyUI/AIMDO memory tracking."
    )

    def load_model(
        self,
        model: str,
        dtype: str,
        attention: str,
        download_if_missing: bool,
    ):
        update = _progress_callback(507)
        bundle = load_zonos2_bundle(
            model_choice=model,
            dtype_name=dtype,
            attention=attention,
            download_if_missing=bool(download_if_missing),
            progress_callback=update,
        )
        return (bundle,)


class Zonos2VoiceGeneration:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "zonos2_model": (
                MODEL_TYPE,
                {
                    "tooltip": "Connect the zonos2_model output from ZONOS2 Model Loader.",
                },
            ),
            "text": _text_input(),
        }
        required.update(_generation_controls())
        return {"required": required}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    OUTPUT_TOOLTIPS = (
        "Generated mono speech as native ComfyUI AUDIO at 44.1 kHz.",
    )
    FUNCTION = "generate"
    CATEGORY = "ZONOS2 TTS"
    DESCRIPTION = "Generate speech with ZONOS2."

    def generate(
        self,
        zonos2_model,
        text: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float,
        repetition_window: int,
        repetition_penalty: float,
        repetition_codebooks: int,
        speaking_rate: str,
        loudness_lufs: str,
        estimated_snr: str,
        maximum_pause: str,
        estimated_bandlimit_hz: str,
        leading_silence: str,
        trailing_silence: str,
        seed: int,
    ):
        options = _sampling_options(
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            min_p,
            repetition_window,
            repetition_penalty,
            repetition_codebooks,
            seed,
        )
        audio = generate_zonos2_audio(
            zonos2_model,
            text=text,
            options=options,
            speaking_rate_bucket=_bucket(speaking_rate),
            quality_buckets=_quality_buckets(
                loudness_lufs,
                estimated_snr,
                maximum_pause,
                estimated_bandlimit_hz,
                leading_silence,
                trailing_silence,
            ),
            progress_callback=_progress_callback(options.max_new_tokens),
        )
        return (audio,)


class Zonos2VoiceClone:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "zonos2_model": (
                MODEL_TYPE,
                {
                    "tooltip": "Connect the zonos2_model output from ZONOS2 Model Loader.",
                },
            ),
            "text": _text_input(),
            "reference_audio": (
                "AUDIO",
                {
                    "tooltip": (
                        "Reference voice AUDIO noodle. "
                        f"{RECOMMENDED_REFERENCE_MIN_SECONDS:.0f}-"
                        f"{RECOMMENDED_REFERENCE_MAX_SECONDS:.0f} seconds of clean, "
                        "single-speaker speech is recommended. Audio longer than "
                        f"{MAX_REFERENCE_SECONDS:.0f} seconds is clipped to the first "
                        f"{MAX_REFERENCE_SECONDS:.0f} seconds with a CLI warning."
                    ),
                },
            ),
            "clean_speaker_background": (
                "BOOLEAN",
                {
                    "default": True,
                    "tooltip": "Tell ZONOS2 whether the reference voice was recorded with little background noise. Enable for clean studio-like speech; disable when the source contains noticeable room noise or ambience.",
                },
            ),
            "accurate_mode": (
                "BOOLEAN",
                {
                    "default": True,
                    "tooltip": "Enable ZONOS2 accurate mode for closer speaker matching. Disable for a more expressive result that may follow the reference identity less strictly.",
                },
            ),
        }
        required.update(_generation_controls())
        return {"required": required}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    OUTPUT_TOOLTIPS = (
        "Voice-cloned mono speech as native ComfyUI AUDIO at 44.1 kHz.",
    )
    FUNCTION = "clone"
    CATEGORY = "ZONOS2 TTS"
    DESCRIPTION = (
        "Clone a voice from a native ComfyUI AUDIO input without reference text."
    )

    def clone(
        self,
        zonos2_model,
        text: str,
        reference_audio: dict,
        clean_speaker_background: bool,
        accurate_mode: bool,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float,
        repetition_window: int,
        repetition_penalty: float,
        repetition_codebooks: int,
        speaking_rate: str,
        loudness_lufs: str,
        estimated_snr: str,
        maximum_pause: str,
        estimated_bandlimit_hz: str,
        leading_silence: str,
        trailing_silence: str,
        seed: int,
    ):
        options = _sampling_options(
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            min_p,
            repetition_window,
            repetition_penalty,
            repetition_codebooks,
            seed,
        )
        audio = generate_zonos2_audio(
            zonos2_model,
            text=text,
            options=options,
            speaking_rate_bucket=_bucket(speaking_rate),
            quality_buckets=_quality_buckets(
                loudness_lufs,
                estimated_snr,
                maximum_pause,
                estimated_bandlimit_hz,
                leading_silence,
                trailing_silence,
            ),
            reference_audio=reference_audio,
            clean_speaker_background=bool(clean_speaker_background),
            accurate_mode=bool(accurate_mode),
            progress_callback=_progress_callback(options.max_new_tokens),
        )
        return (audio,)


NODE_CLASS_MAPPINGS = {
    "Zonos2ModelLoader": Zonos2ModelLoader,
    "Zonos2VoiceGeneration": Zonos2VoiceGeneration,
    "Zonos2VoiceClone": Zonos2VoiceClone,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Zonos2ModelLoader": "ZONOS2 Model Loader",
    "Zonos2VoiceGeneration": "ZONOS2 Voice Generation",
    "Zonos2VoiceClone": "ZONOS2 Voice Clone",
}
