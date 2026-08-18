from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_HANDOFF_NAME = "clip_lr_grad_eclip_handoff_v1_20260812_105027"


@dataclass(frozen=True)
class AppConfig:
    app_root: Path
    handoff_root: Path
    artifacts_root: Path

    @classmethod
    def from_env(cls) -> "AppConfig":
        app_root = Path(__file__).resolve().parents[1]
        repo_root = app_root.parent
        handoff_env = os.getenv("CLIP_XAI_HANDOFF_ROOT")
        handoff_root = Path(handoff_env).expanduser() if handoff_env else repo_root / DEFAULT_HANDOFF_NAME
        return cls(
            app_root=app_root,
            handoff_root=handoff_root.resolve(),
            artifacts_root=(app_root / "artifacts").resolve(),
        )

    def validate_handoff(self) -> None:
        required = [
            self.handoff_root / "code" / "inference_clip_lr.py",
            self.handoff_root / "code" / "grad_eclip.py",
            self.handoff_root / "models" / "clip_model",
            self.handoff_root / "models" / "clip_processor",
            self.handoff_root / "models" / "clip_lr_classifier_new_run.joblib",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing handoff files: " + ", ".join(missing))
