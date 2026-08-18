from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from .config import AppConfig


def add_handoff_code_to_path(handoff_root: Path) -> None:
    code_dir = str((handoff_root / "code").resolve())
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)


@lru_cache(maxsize=2)
def load_pipeline(device: str | None = None):
    config = AppConfig.from_env()
    config.validate_handoff()
    add_handoff_code_to_path(config.handoff_root)

    from inference_clip_lr import load_pipeline as handoff_load_pipeline

    return handoff_load_pipeline(config.handoff_root, device=device)
