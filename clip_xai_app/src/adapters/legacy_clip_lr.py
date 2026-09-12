from __future__ import annotations

from ..contracts import AdapterHealth
from ..model_registry import get_model_registry


class LegacyClipLRAdapter:
    model_id = "legacy_clip_lr"

    def health(self) -> AdapterHealth:
        return next(item for item in get_model_registry() if item.model_id == self.model_id)

    def load_pipeline(self, device: str | None = None):
        from ..model_loader import load_pipeline

        return load_pipeline(device=device)
