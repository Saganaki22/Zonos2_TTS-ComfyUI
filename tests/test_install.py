import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("zonos2_install_test", ROOT / "install.py")
INSTALL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSTALL)


def test_install_command_uses_uv_for_active_python(monkeypatch):
    monkeypatch.setattr(INSTALL.shutil, "which", lambda name: "C:/tools/uv.exe")

    command = INSTALL._install_command(["tqdm"])

    assert command == [
        "C:/tools/uv.exe",
        "pip",
        "install",
        "--python",
        sys.executable,
        "tqdm",
    ]


def test_install_command_falls_back_to_pip(monkeypatch):
    monkeypatch.setattr(INSTALL.shutil, "which", lambda name: None)

    command = INSTALL._install_command(["tqdm"])

    assert command == [sys.executable, "-m", "pip", "install", "tqdm"]
