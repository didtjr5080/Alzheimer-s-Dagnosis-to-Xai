"""
inference_clip_lr.py

새 런타임에서 전역 변수 없이 CLIP+LR 모델을 로드하고 추론하는 독립 스크립트.
패키지 루트 기준 상대경로로 모델을 로드한다.

사용 예:
    from inference_clip_lr import ClipLRPredictor
    predictor = ClipLRPredictor(package_root=".")
    result = predictor.predict_subject(["path/to/slice1.png", "path/to/slice2.png"])
"""

import os
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import joblib


class ClipLRPredictor:
    def __init__(self, package_root, device=None):
        self.package_root = package_root
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        clip_model_dir = os.path.join(package_root, "models", "clip_model")
        clip_processor_dir = os.path.join(package_root, "models", "clip_processor")
        clf_path = os.path.join(package_root, "models", "clip_lr_classifier_new_run.joblib")

        assert os.path.exists(clip_model_dir), f"CLIP 모델 폴더 없음: {clip_model_dir}"
        assert os.path.exists(clip_processor_dir), f"CLIP processor 폴더 없음: {clip_processor_dir}"
        assert os.path.exists(clf_path), f"LR classifier 파일 없음: {clf_path}"

        self.clip_model = CLIPModel.from_pretrained(clip_model_dir).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_processor_dir)
        self.clip_model.eval()

        self.clf = joblib.load(clf_path)

        # 클래스 순서는 반드시 저장된 classes_에서 읽는다 (하드코딩 금지)
        self.classes_idx = self.clf.classes_
        class_name_map_path = os.path.join(package_root, "metadata", "model_config.json")
        with open(class_name_map_path, "r", encoding="utf-8") as f:
            model_config = json.load(f)
        self.idx_to_name = model_config["class_name_mapping"]  # {"0": "CN", "1": "MCI", "2": "AD"}
        self.class_names_ordered = [self.idx_to_name[str(c)] for c in self.classes_idx]

    def _image_to_pixel_values(self, image):
        img = image.convert("RGB") if isinstance(image, Image.Image) else Image.open(image).convert("RGB")
        pixel_values = self.clip_processor.image_processor(
            images=img, return_tensors="pt"
        )["pixel_values"]
        return pixel_values

    def _load_pixel_values(self, image_path):
        return self._image_to_pixel_values(image_path)

    @torch.no_grad()
    def get_image_embedding_from_image(self, image):
        """단일 이미지의 CLIP 512차원 임베딩 (L2 정규화 미적용, 원본 파이프라인과 동일)."""
        pixel_values = self._image_to_pixel_values(image).to(self.device)
        outputs = self.clip_model.get_image_features(pixel_values=pixel_values)
        embeds = outputs.pooler_output if not isinstance(outputs, torch.Tensor) else outputs
        return embeds.cpu().numpy()[0]

    def get_image_embedding(self, image_path):
        return self.get_image_embedding_from_image(image_path)

    def predict_slice_from_image(self, image):
        """단일 슬라이스의 LR logit/probability."""
        embed = self.get_image_embedding_from_image(image)
        logits = embed @ self.clf.coef_.T + self.clf.intercept_
        probs = self.clf.predict_proba(embed.reshape(1, -1))[0]
        return {
            "embedding": embed,
            "logits": dict(zip(self.class_names_ordered, logits.tolist())),
            "probs": dict(zip(self.class_names_ordered, probs.tolist())),
        }

    def predict_slice(self, image_path):
        """단일 슬라이스의 LR logit/probability."""
        result = self.predict_slice_from_image(image_path)
        result.pop("embedding", None)
        return result

    def predict_subject(self, image_paths):
        """여러 슬라이스(같은 MR_ID)의 확률을 단순 평균하여 subject 확률 산출."""
        assert len(image_paths) > 0, "이미지 경로가 비어있습니다."
        embeds = np.stack([self.get_image_embedding(p) for p in image_paths])
        slice_probs = self.clf.predict_proba(embeds)  # (n_slices, 3), 열 순서는 self.classes_idx 기준
        subject_probs = slice_probs.mean(axis=0)
        pred_idx = int(self.classes_idx[subject_probs.argmax()])
        return {
            "subject_probs": dict(zip(self.class_names_ordered, subject_probs.tolist())),
            "predicted_class": self.idx_to_name[str(pred_idx)],
            "n_slices_used": len(image_paths),
        }


@dataclass
class PipelineBundle:
    root: Path
    device: torch.device
    predictor: ClipLRPredictor
    _explainer: object = None

    @property
    def class_names(self):
        return list(self.predictor.class_names_ordered)

    @property
    def classes_idx(self):
        return np.asarray(self.predictor.classes_idx)

    def get_explainer(self):
        if self._explainer is None:
            from grad_eclip import GradECLIPExplainer
            self._explainer = GradECLIPExplainer(str(self.root), device=self.device)
        return self._explainer


@dataclass
class SlicePredictionBatch:
    logits: np.ndarray
    probs: np.ndarray
    embeddings: np.ndarray
    predicted_class_indices: np.ndarray
    predicted_classes: list
    confidences: np.ndarray
    class_names: list
    classes_idx: np.ndarray


@dataclass
class SubjectPrediction:
    probs: dict
    predicted_class: str
    predicted_class_idx: int
    confidence: float
    n_slices_used: int
    single_slice_warning: str | None = None


@dataclass
class XAIResult:
    heatmap_224: np.ndarray
    predicted_class: str
    target_class: str
    logits: dict
    probs: dict


def _resolve_device(device):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pipeline(root: Path, device: str | None = None) -> PipelineBundle:
    root = Path(root)
    torch_device = _resolve_device(device)
    predictor = ClipLRPredictor(package_root=str(root), device=torch_device)

    expected_names = ["CN", "MCI", "AD"]
    if predictor.class_names_ordered != expected_names:
        raise ValueError(
            "Unexpected class order from clf.classes_: "
            f"{predictor.classes_idx.tolist()} -> {predictor.class_names_ordered}"
        )
    return PipelineBundle(root=root, device=torch_device, predictor=predictor)


def predict_slices(images: list[Image.Image], bundle: PipelineBundle) -> SlicePredictionBatch:
    if not images:
        raise ValueError("At least one slice image is required.")

    embeddings = np.stack([
        bundle.predictor.get_image_embedding_from_image(image) for image in images
    ])
    logits = embeddings @ bundle.predictor.clf.coef_.T + bundle.predictor.clf.intercept_
    probs = bundle.predictor.clf.predict_proba(embeddings)
    pred_local = probs.argmax(axis=1)
    pred_class_indices = bundle.classes_idx[pred_local].astype(int)
    pred_classes = [bundle.predictor.idx_to_name[str(idx)] for idx in pred_class_indices]
    confidences = probs.max(axis=1)

    return SlicePredictionBatch(
        logits=logits,
        probs=probs,
        embeddings=embeddings,
        predicted_class_indices=pred_class_indices,
        predicted_classes=pred_classes,
        confidences=confidences,
        class_names=bundle.class_names,
        classes_idx=bundle.classes_idx,
    )


def aggregate_subject(slice_predictions: SlicePredictionBatch) -> SubjectPrediction:
    if slice_predictions.probs.shape[0] == 0:
        raise ValueError("At least one slice prediction is required.")

    subject_probs = slice_predictions.probs.mean(axis=0)
    pred_local = int(subject_probs.argmax())
    pred_idx = int(slice_predictions.classes_idx[pred_local])
    warning = None
    if slice_predictions.probs.shape[0] == 1:
        warning = (
            "Single-slice result: this is not equivalent to the validated "
            "subject-level multi-slice probability averaging protocol."
        )

    return SubjectPrediction(
        probs=dict(zip(slice_predictions.class_names, subject_probs.tolist())),
        predicted_class=slice_predictions.class_names[pred_local],
        predicted_class_idx=pred_idx,
        confidence=float(subject_probs[pred_local]),
        n_slices_used=int(slice_predictions.probs.shape[0]),
        single_slice_warning=warning,
    )


def generate_xai(image: Image.Image, target_class_idx: int, bundle: PipelineBundle) -> XAIResult:
    classes_idx = bundle.classes_idx.astype(int).tolist()
    if int(target_class_idx) not in classes_idx:
        raise ValueError(f"Unknown target_class_idx: {target_class_idx}")

    target_local_idx = classes_idx.index(int(target_class_idx))
    target_class_name = bundle.class_names[target_local_idx]
    result = bundle.get_explainer().explain_image(image, target_class_name=target_class_name)
    return XAIResult(
        heatmap_224=result["heatmap_224"],
        predicted_class=result["predicted_class"],
        target_class=result["target_class"],
        logits=result["logits"],
        probs=result["probs"],
    )


def select_representative_slices(slice_predictions, max_count: int = 3) -> list[int]:
    if max_count <= 0:
        return []
    n_slices = int(slice_predictions.probs.shape[0])
    if n_slices == 0:
        return []

    subject = aggregate_subject(slice_predictions)
    final_local_idx = slice_predictions.class_names.index(subject.predicted_class)
    confidence = slice_predictions.probs.max(axis=1)
    uncertainty = 1.0 - confidence

    candidates = [
        n_slices // 2,
        int(slice_predictions.logits[:, final_local_idx].argmax()),
        int(uncertainty.argmax()),
    ]

    selected = []
    for idx in candidates:
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= max_count:
            break
    return selected


if __name__ == "__main__":
    import sys
    package_root = sys.argv[1] if len(sys.argv) > 1 else "."
    predictor = ClipLRPredictor(package_root=package_root)
    print("클래스 순서:", predictor.class_names_ordered)
    print("모델 로드 및 초기화 성공.")
