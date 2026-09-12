"""Read-only reuse of the existing CLIPtoXAI CLIP+LR pipeline.

This module NEVER imports or modifies anything under ``clip_xai_app/``. It
loads the underlying handoff package directly by file path so this extension
has no dependency on (and cannot collide with) clip_xai_app's own ``src``
package name on ``sys.path``.

Reused originals (paths are read-only; nothing here writes to them):

- ``clip_lr_grad_eclip_handoff_v1_20260812_105027/code/inference_clip_lr.py``
  - ``load_pipeline(root, device)`` -> ``PipelineBundle``
  - ``predict_slices(images, bundle)`` -> ``SlicePredictionBatch`` (has ``.probs``)
  - ``generate_xai(image, target_class_idx, bundle)`` -> ``XAIResult`` (has ``.heatmap_224``)
- ``clip_lr_grad_eclip_handoff_v1_20260812_105027/code/grad_eclip.py``
  - ``GradECLIPExplainer`` (used indirectly via ``PipelineBundle.get_explainer()``)

Class order is never hardcoded here either: it comes from
``PipelineBundle.class_names`` / ``predictor.classes_idx``, which the original
``load_pipeline`` already validates against ``clf.classes_`` and
``metadata/model_config.json``.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image as PILImage


DEFAULT_HANDOFF_DIR_NAME = "clip_lr_grad_eclip_handoff_v1_20260812_105027"
REQUIRED_RELATIVE_PATHS = (
    "code/inference_clip_lr.py",
    "code/grad_eclip.py",
    "models/clip_model",
    "models/clip_processor",
)


class ModelUnavailableError(RuntimeError):
    """Raised when the real classifier/CLIP pipeline cannot be located or loaded."""


def _repo_root() -> Path:
    # graph_xai_extension/graph_xai/legacy_adapter.py -> repo root is 2 levels up
    return Path(__file__).resolve().parents[2]


def resolve_handoff_root(classifier_path: str | Path | None) -> Path:
    """Resolve the handoff package root directory.

    ``classifier_path`` (GRAPH_XAI_CLASSIFIER_PATH) may point either at the
    joblib file itself (``<root>/models/xxx.joblib``) or directly at the
    handoff root directory. If unset, falls back to the repo's default
    handoff directory name.
    """
    if classifier_path:
        given = Path(classifier_path).expanduser()
        if given.is_file():
            return given.parent.parent.resolve()
        if given.is_dir():
            return given.resolve()
        raise ModelUnavailableError(f"GRAPH_XAI_CLASSIFIER_PATH not found: {given}")
    return (_repo_root() / DEFAULT_HANDOFF_DIR_NAME).resolve()


def validate_handoff_root(root: Path) -> list[str]:
    """Return a list of missing required paths (empty list = ready to load)."""
    missing = [str(root / rel) for rel in REQUIRED_RELATIVE_PATHS if not (root / rel).exists()]
    models_dir = root / "models"
    if not models_dir.exists() or not any(models_dir.glob("*.joblib")):
        missing.append(str(models_dir / "*.joblib"))
    return missing


def _import_inference_clip_lr(root: Path):
    code_dir = root / "code"
    code_dir_str = str(code_dir)
    if code_dir_str not in sys.path:
        # Needed only because inference_clip_lr.PipelineBundle.get_explainer()
        # lazily does `from grad_eclip import GradECLIPExplainer` by bare name.
        sys.path.insert(0, code_dir_str)

    module_name = "graph_xai_extension._legacy_inference_clip_lr"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, code_dir / "inference_clip_lr.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _to_pil_image(image):
    """The reused ``inference_clip_lr``/``grad_eclip`` code only accepts a PIL
    Image or a file path. Masked region previews produced by this extension's
    own ``masking.apply_region_mask`` are plain numpy arrays, so bridge that
    gap here rather than in the pure numpy-based masking/perturbation code."""
    if isinstance(image, PILImage.Image):
        return image
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return PILImage.fromarray(array)


class LegacyPipelineHandle:
    """Thin wrapper exposing the reused pipeline through a small stable API."""

    def __init__(self, module, bundle, root: Path):
        self._module = module
        self._bundle = bundle
        self.root = root

    @property
    def class_names(self) -> list[str]:
        return list(self._bundle.class_names)

    @property
    def device(self):
        return self._bundle.device

    def predict_proba(self, images):
        pil_images = [_to_pil_image(image) for image in images]
        batch = self._module.predict_slices(pil_images, self._bundle)
        return batch.probs

    def generate_cam(self, image, target_class_idx: int):
        image = _to_pil_image(image)
        result = self._module.generate_xai(image, target_class_idx=target_class_idx, bundle=self._bundle)
        return result.heatmap_224


_PIPELINE_CACHE: dict[tuple, LegacyPipelineHandle] = {}


def load_real_pipeline(classifier_path: str | Path | None = None, device: str | None = None) -> LegacyPipelineHandle:
    root = resolve_handoff_root(classifier_path)
    missing = validate_handoff_root(root)
    if missing:
        raise ModelUnavailableError(
            "분류 모델 파일을 찾을 수 없어 실제 Graph XAI 분석을 실행할 수 없습니다. "
            "학습된 분류기 경로를 지정해 주세요. 테스트용 mock 모델 결과는 실제 의료영상 분석 결과가 아닙니다. "
            "누락된 경로: " + ", ".join(missing)
        )

    cache_key = (str(root), device)
    if cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key]

    module = _import_inference_clip_lr(root)
    bundle = module.load_pipeline(root, device=device)
    handle = LegacyPipelineHandle(module=module, bundle=bundle, root=root)
    _PIPELINE_CACHE[cache_key] = handle
    return handle
