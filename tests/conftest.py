import importlib.util
import sys
from pathlib import Path


PACKAGE_NAME = "zonos2_tts_comfyui_test"
ROOT = Path(__file__).resolve().parents[1]

if PACKAGE_NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = package
    spec.loader.exec_module(package)
