"""ComfyUI custom nodes for Zyphra ZONOS2."""

from __future__ import annotations

import logging
from typing import Any

__version__ = "0.1.4"
__citation__ = """@misc{zyphra2025zonos,
  title     = {Zonos V2 Technical Report},
  author    = {Gabriel Clark, Sofian Mejjoute, Mohamed Osman, George Close, Beren Millidge},
  year      = {2026},
}"""

logger = logging.getLogger("Zonos2_TTS-ComfyUI")
logger.propagate = False
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[Zonos2_TTS-ComfyUI] %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

NODE_CLASS_MAPPINGS: dict[str, Any] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}

if __package__:
    try:
        from .loader import register_model_folder
        from .nodes import NODE_CLASS_MAPPINGS as _NODE_CLASS_MAPPINGS
        from .nodes import (
            NODE_DISPLAY_NAME_MAPPINGS as _NODE_DISPLAY_NAME_MAPPINGS,
        )

        register_model_folder()
        NODE_CLASS_MAPPINGS.update(_NODE_CLASS_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(_NODE_DISPLAY_NAME_MAPPINGS)
        logger.info("Registered %d node(s).", len(NODE_CLASS_MAPPINGS))
    except Exception as exc:
        logger.error("Failed to register ZONOS2 nodes: %s", exc, exc_info=True)

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "__citation__",
    "__version__",
]
