"""ZONOS2 model discovery, loading, and ComfyUI/AIMDO registration."""

from __future__ import annotations

import gc
import importlib.util
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from safetensors import safe_open

from .native import (
    Zonos2Config,
    Zonos2Model,
    build_native_model,
    load_native_weights,
    read_config,
    set_runtime_dtype,
    validate_checkpoint_layout,
)

logger = logging.getLogger("Zonos2_TTS-ComfyUI")

MODEL_FOLDER_NAME = "zonos2"
MODEL_REPO_ID = "drbaph/ZONOS2-BF16"
PRESET_MODELS = {
    "ZONOS2 BF16 - drbaph/ZONOS2-BF16": "zonos2-bf16.safetensors",
}
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
PARAMS_PATH = ASSETS_DIR / "params.json"

DTYPE_OPTIONS = ["auto", "bf16", "fp16"]
ATTENTION_OPTIONS = ["auto", "SDPA", "flash_attention"]

_ACTIVE_BUNDLE: "Zonos2Bundle | None" = None
_ACTIVE_LOAD_KEY: tuple[Any, ...] | None = None


@dataclass
class Zonos2Bundle:
    model: Zonos2Model | None
    config: Zonos2Config
    model_path: Path
    device: torch.device
    torch_dtype: torch.dtype
    dtype_name: str
    attention: str
    download_if_missing: bool
    patchers: list[Any] = field(default_factory=list)
    codec: Any = None
    speaker_encoder: Any = None


def bundled_params_path() -> Path:
    if not PARAMS_PATH.is_file():
        raise FileNotFoundError(
            f"Bundled ZONOS2 configuration is missing: {PARAMS_PATH}"
        )
    return PARAMS_PATH


def read_bundled_config() -> Zonos2Config:
    return read_config(bundled_params_path())


def model_dir() -> Path:
    try:
        import folder_paths

        base = Path(folder_paths.models_dir) / MODEL_FOLDER_NAME
    except Exception:
        base = Path(__file__).resolve().parents[2] / "models" / MODEL_FOLDER_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


def register_model_folder() -> None:
    try:
        import folder_paths

        paths = getattr(folder_paths, "folder_names_and_paths", {})
        extensions = {".safetensors", ".sft", ".pth", ".pt"}
        if MODEL_FOLDER_NAME in paths:
            current_paths, current_extensions = paths[MODEL_FOLDER_NAME]
            normalized = list(current_paths)
            target = str(model_dir())
            if target not in normalized:
                normalized.append(target)
            paths[MODEL_FOLDER_NAME] = (
                normalized,
                set(current_extensions) | extensions,
            )
        else:
            paths[MODEL_FOLDER_NAME] = ([str(model_dir())], extensions)
    except Exception as exc:
        logger.debug("Could not register the ZONOS2 model folder: %s", exc)


def get_model_choices() -> list[str]:
    register_model_folder()
    local = sorted(
        path.name
        for path in model_dir().iterdir()
        if path.is_file() and path.suffix.lower() in {".safetensors", ".sft"}
    )
    choices = list(PRESET_MODELS)
    preset_files = set(PRESET_MODELS.values())
    choices.extend(name for name in local if name not in preset_files)
    return choices


def _download_model(filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    logger.info("Downloading %s from %s.", filename, MODEL_REPO_ID)
    downloaded = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=filename,
        local_dir=str(model_dir()),
    )
    return Path(downloaded)


def resolve_model_path(model_choice: str, download_if_missing: bool) -> Path:
    filename = PRESET_MODELS.get(model_choice, Path(model_choice).name)
    path = model_dir() / filename
    if path.is_file():
        return path
    if download_if_missing:
        return _download_model(filename)
    raise FileNotFoundError(
        f"ZONOS2 model not found at {path}. Enable download_if_missing or "
        f"place {filename} in {model_dir()}."
    )


def inspect_checkpoint_dtype(checkpoint_path: Path) -> torch.dtype:
    floating_dtypes: set[torch.dtype] = set()
    dtype_map = {
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "F32": torch.float32,
        "F64": torch.float64,
    }
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            raw_dtype = str(handle.get_slice(name).get_dtype()).upper()
            dtype = dtype_map.get(raw_dtype)
            if dtype is not None:
                floating_dtypes.add(dtype)
    if not floating_dtypes:
        raise ValueError(f"No floating-point tensors found in {checkpoint_path}.")
    if len(floating_dtypes) != 1:
        values = ", ".join(sorted(str(value) for value in floating_dtypes))
        raise ValueError(
            f"Mixed floating-point checkpoint dtypes are not supported: {values}"
        )
    return next(iter(floating_dtypes))


def resolve_dtype(
    dtype_name: str,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.dtype:
    if dtype_name == "auto":
        dtype = inspect_checkpoint_dtype(checkpoint_path)
    elif dtype_name == "bf16":
        dtype = torch.bfloat16
    elif dtype_name == "fp16":
        dtype = torch.float16
    else:
        raise ValueError(f"Unsupported dtype: {dtype_name}")

    if device.type == "cpu" and dtype == torch.float16:
        logger.warning("FP16 on CPU is poorly supported; using FP32 instead.")
        return torch.float32
    return dtype


def resolve_device() -> torch.device:
    try:
        import comfy.model_management as mm

        return torch.device(mm.get_torch_device())
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dynamic_vram_active(device: torch.device) -> bool:
    if torch.device(device).type == "cpu":
        return False
    try:
        import comfy.memory_management

        return bool(comfy.memory_management.aimdo_enabled)
    except Exception:
        return False


def estimate_runtime_model_bytes(
    model: torch.nn.Module,
    dtype: torch.dtype,
) -> int:
    total = 0
    for value in list(model.parameters()) + list(model.buffers()):
        item_size = dtype.itemsize if value.is_floating_point() else value.element_size()
        total += value.numel() * item_size
    return total


def should_use_dynamic_vram(
    model: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> bool:
    if not dynamic_vram_active(device):
        return False
    if torch.device(device).type != "cuda":
        return True
    total_vram = torch.cuda.get_device_properties(device).total_memory
    model_bytes = estimate_runtime_model_bytes(model, dtype)
    reserve = 3 * 1024**3
    return model_bytes + reserve > total_vram


def _flash_attention_available(
    device: torch.device,
    dtype: torch.dtype,
) -> bool:
    return (
        device.type == "cuda"
        and dtype in {torch.float16, torch.bfloat16}
        and importlib.util.find_spec("flash_attn") is not None
    )


def resolve_attention(
    attention: str,
    device: torch.device,
    dtype: torch.dtype,
) -> str:
    if attention == "auto":
        return (
            "flash_attention"
            if _flash_attention_available(device, dtype)
            else "sdpa"
        )
    if attention == "SDPA":
        return "sdpa"
    if attention == "flash_attention":
        if not _flash_attention_available(device, dtype):
            raise RuntimeError(
                "flash_attention requires CUDA, BF16/FP16, and flash-attn."
            )
        return "flash_attention"
    raise ValueError(f"Unsupported attention mode: {attention}")


try:
    import comfy.model_patcher as _model_patcher
    Zonos2Patcher = _model_patcher.CoreModelPatcher
except Exception:
    Zonos2Patcher = None


def _empty_accelerator_cache() -> None:
    try:
        import comfy.model_management as mm

        mm.soft_empty_cache()
        return
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()


def _register_with_comfy(patcher: Any) -> None:
    if patcher is None:
        return
    try:
        import comfy.model_management as mm

        if patcher.load_device.type == "cpu":
            return
        mm.load_models_gpu([patcher])
        logger.info(
            "Loaded %s through ComfyUI%s memory management.",
            patcher.model.__class__.__name__,
            "/AIMDO" if patcher.is_dynamic() else "",
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not load ZONOS2 through ComfyUI memory management."
        ) from exc


def _unregister_from_comfy(patcher: Any) -> None:
    try:
        import comfy.model_management as mm

        survivors = []
        for loaded in mm.current_loaded_models:
            if loaded.model is patcher:
                try:
                    if loaded.model_finalizer is not None:
                        loaded.model_finalizer.detach()
                    loaded.model_finalizer = None
                    loaded.real_model = None
                except Exception:
                    pass
                try:
                    finalizer = getattr(loaded, "_patcher_finalizer", None)
                    if finalizer is not None:
                        finalizer.detach()
                    loaded._patcher_finalizer = None
                except Exception:
                    pass
                continue
            survivors.append(loaded)
        mm.current_loaded_models[:] = survivors
    except Exception:
        pass


def register_runtime_module(
    module: torch.nn.Module,
    device: torch.device,
    dynamic: bool | None = None,
) -> Any:
    if Zonos2Patcher is None or torch.device(device).type == "cpu":
        module.to(device)
        return None
    import comfy.model_patcher as model_patcher

    use_dynamic = dynamic_vram_active(device) and dynamic is not False
    patcher_class = (
        model_patcher.CoreModelPatcher
        if use_dynamic
        else model_patcher.ModelPatcher
    )
    patcher = patcher_class(
        module,
        load_device=torch.device(device),
        offload_device=torch.device("cpu"),
    )
    module.model_loaded_weight_memory = 0
    if not patcher.is_dynamic() and hasattr(module, "device"):
        module.device = torch.device(device)
    _register_with_comfy(patcher)
    return patcher


def resume_runtime_module(patcher: Any, device: torch.device) -> None:
    if patcher is None:
        return
    _register_with_comfy(patcher)


def unload_runtime_module(patcher: Any, hard: bool = True) -> None:
    if patcher is None:
        return
    _unregister_from_comfy(patcher)
    try:
        patcher.detach()
    except Exception:
        pass


def resume_bundle_to_device(bundle: Zonos2Bundle) -> None:
    for patcher in bundle.patchers:
        resume_runtime_module(patcher, bundle.device)


def add_bundle_module(
    bundle: Zonos2Bundle,
    module: torch.nn.Module,
    dynamic: bool | None = None,
) -> Any:
    patcher = register_runtime_module(
        module,
        bundle.device,
        dynamic=dynamic,
    )
    if patcher is not None:
        bundle.patchers.append(patcher)
    return patcher


def unload_zonos2_bundle(
    bundle: Zonos2Bundle | None,
    reason: str = "manual unload",
    hard: bool = True,
) -> None:
    global _ACTIVE_BUNDLE, _ACTIVE_LOAD_KEY

    if bundle is None:
        return
    logger.info("Unloading ZONOS2 bundle (%s).", reason)
    for patcher in list(bundle.patchers):
        unload_runtime_module(patcher, hard=hard)
    bundle.patchers.clear()

    modules = [
        bundle.model,
        getattr(bundle.codec, "model", bundle.codec),
        getattr(bundle.speaker_encoder, "model", bundle.speaker_encoder),
    ]
    for module in modules:
        if not isinstance(module, torch.nn.Module):
            continue
        try:
            module.model_loaded_weight_memory = 0
            if hasattr(module, "dynamic_vbars"):
                module.dynamic_vbars.clear()
            if hasattr(module, "dynamic_pins"):
                module.dynamic_pins.clear()
            if hard and hasattr(module, "to_empty"):
                module.to_empty(device=torch.device("meta"))
            elif not hard:
                module.to("cpu")
        except Exception:
            pass

    if hard:
        bundle.model = None
        bundle.codec = None
        bundle.speaker_encoder = None
    gc.collect()
    _empty_accelerator_cache()
    if _ACTIVE_BUNDLE is bundle:
        _ACTIVE_BUNDLE = None
        _ACTIVE_LOAD_KEY = None


def load_zonos2_bundle(
    model_choice: str,
    dtype_name: str,
    attention: str,
    download_if_missing: bool,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Zonos2Bundle:
    global _ACTIVE_BUNDLE, _ACTIVE_LOAD_KEY

    register_model_folder()
    checkpoint_path = resolve_model_path(model_choice, download_if_missing)
    device = resolve_device()
    torch_dtype = resolve_dtype(dtype_name, checkpoint_path, device)
    runtime_attention = resolve_attention(attention, device, torch_dtype)
    stat = checkpoint_path.stat()
    load_key = (
        str(checkpoint_path.resolve()),
        stat.st_size,
        stat.st_mtime_ns,
        str(device),
        str(torch_dtype),
        runtime_attention,
    )

    if _ACTIVE_BUNDLE is not None and _ACTIVE_LOAD_KEY == load_key:
        resume_bundle_to_device(_ACTIVE_BUNDLE)
        return _ACTIVE_BUNDLE
    if _ACTIVE_BUNDLE is not None:
        unload_zonos2_bundle(
            _ACTIVE_BUNDLE,
            reason="model, dtype, or attention changed",
            hard=True,
        )

    config = read_bundled_config()
    model = build_native_model(config)
    count, missing, unexpected = validate_checkpoint_layout(
        model,
        checkpoint_path,
    )
    if missing or unexpected:
        raise RuntimeError(
            f"ZONOS2 checkpoint does not match bundled params.json. "
            f"Missing={sorted(missing)[:10]}, "
            f"unexpected={sorted(unexpected)[:10]}"
        )
    logger.info(
        "Loading %d ZONOS2 tensors from %s as %s, then staging for %s with %s.",
        count,
        checkpoint_path,
        torch_dtype,
        device,
        runtime_attention,
    )
    use_dynamic_vram = should_use_dynamic_vram(model, device, torch_dtype)
    source_dtype = inspect_checkpoint_dtype(checkpoint_path)
    weight_device = torch.device("cpu") if device.type != "cpu" else device
    load_native_weights(
        model,
        checkpoint_path,
        weight_device,
        source_dtype,
        progress_callback=progress_callback,
    )
    set_runtime_dtype(model, torch_dtype)
    if use_dynamic_vram:
        logger.info(
            "ZONOS2 is using file-backed %s weights with on-demand %s "
            "AIMDO residency.",
            source_dtype,
            torch_dtype,
        )
    elif dynamic_vram_active(device):
        logger.info(
            "ZONOS2 fits in total VRAM with a 3 GiB runtime reserve; using "
            "ComfyUI's static GPU path to avoid retained CPU model backing."
        )
    patchers: list[Any] = []
    model_patcher = register_runtime_module(
        model,
        device,
        dynamic=use_dynamic_vram,
    )
    if model_patcher is not None:
        patchers.append(model_patcher)

    bundle = Zonos2Bundle(
        model=model,
        config=config,
        model_path=checkpoint_path,
        device=device,
        torch_dtype=torch_dtype,
        dtype_name=dtype_name,
        attention=runtime_attention,
        download_if_missing=bool(download_if_missing),
        patchers=patchers,
    )
    try:
        from .runtime import ensure_codec

        ensure_codec(bundle)
    except Exception:
        unload_zonos2_bundle(
            bundle,
            reason="codec initialization failed",
            hard=True,
        )
        raise
    _ACTIVE_BUNDLE = bundle
    _ACTIVE_LOAD_KEY = load_key
    _empty_accelerator_cache()
    return bundle


def unload_active_bundle() -> None:
    unload_zonos2_bundle(_ACTIVE_BUNDLE, reason="active unload", hard=True)
