"""Dependency check/install helper for Zonos2_TTS-ComfyUI."""

from __future__ import annotations

import importlib.util
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import shutil
import subprocess
import sys


CRITICAL_IMPORTS = ["torch", "torchaudio", "transformers"]
LIGHTWEIGHT_IMPORTS = {
    "accelerate": "accelerate",
    "huggingface_hub": "huggingface-hub",
    "safetensors": "safetensors",
    "tqdm": "tqdm",
}
MIN_TRANSFORMERS = (5, 0, 0)
MAX_TRANSFORMERS = (5, 12, 0)
RECOMMENDED_TRANSFORMERS = "5.3.0"


def _release_tuple(raw_version: str) -> tuple[int, int, int]:
    release = raw_version.split("+", 1)[0].split("-", 1)[0].split(".")
    values = []
    for part in release[:3]:
        digits = "".join(character for character in part if character.isdigit())
        values.append(int(digits or 0))
    return tuple((values + [0, 0, 0])[:3])


def _print_transformers_status() -> None:
    try:
        installed = version("transformers")
    except PackageNotFoundError:
        return
    parsed = _release_tuple(installed)
    if MIN_TRANSFORMERS <= parsed <= MAX_TRANSFORMERS:
        print(
            f"Transformers {installed} is within the tested range "
            "5.0.0 through 5.12.0."
        )
        return
    print(
        f"WARNING: Transformers {installed} is outside the tested range "
        "5.0.0 through 5.12.0. The recommended version is "
        f"{RECOMMENDED_TRANSFORMERS}. This helper will not replace it."
    )


def _install_command(packages: list[str]) -> list[str]:
    uv = shutil.which("uv")
    if uv:
        return [
            uv,
            "pip",
            "install",
            "--python",
            sys.executable,
            *packages,
        ]
    return [sys.executable, "-m", "pip", "install", *packages]


def main() -> int:
    missing_critical = [
        name for name in CRITICAL_IMPORTS if importlib.util.find_spec(name) is None
    ]
    if missing_critical:
        print("Missing ComfyUI/runtime dependency:", ", ".join(missing_critical))
        print(
            "Install this nodepack inside a working ComfyUI environment. "
            "This helper will not modify torch, torchaudio, or transformers."
        )
        return 1

    _print_transformers_status()

    missing = [
        package
        for module, package in LIGHTWEIGHT_IMPORTS.items()
        if importlib.util.find_spec(module) is None
    ]
    if not missing:
        print("Zonos2_TTS-ComfyUI dependencies are already present.")
        return 0

    print("Installing missing lightweight dependencies:", ", ".join(missing))
    print("Torch, torchaudio, and transformers are not modified.")
    command = _install_command(missing)
    installer = "uv" if Path(command[0]).stem.lower() == "uv" else "pip"
    print(f"Using {installer} with the active ComfyUI Python: {sys.executable}")
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
