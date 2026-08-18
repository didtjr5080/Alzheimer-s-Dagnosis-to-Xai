# CLIP + Logistic Regression — Alzheimer's Staging Model Handoff Package

## ⚠️ 연구용 제한 및 경고

- 본 모델은 **연구용**이며, 독립된 임상 검증을 거치지 않았습니다. 현재 성능 수치(subject Accuracy 72.2%, Macro-F1 0.435)는 임상적 유효성을 의미하지 않습니다.
- 이후 생성될 MRI 히트맵(Grad-ECLIP)은 **모델이 점수에 기여한 영역**을 보여줄 뿐이며, **병변·진단·인과관계의 증거가 아닙니다.**
- 본 패키지는 DICOM 헤더, 이름, 생년월일 등 식별정보를 포함하지 않습니다. OASIS-3 파생 PNG와 가명화된 MR_ID만 다루며, OASIS-3 데이터 이용조건을 준수해야 합니다.

## 모델 목적

OASIS-3 파생 coronal MRI 슬라이스를 입력으로 받아 CN(정상)/MCI(경도인지장애)/AD(치매) 3-class 확률을 subject 단위로 산출합니다.

## 전체 파이프라인

1. Coronal MRI 파생 PNG를 RGB로 로드
2. `CLIPProcessor.image_processor`로 전처리 (resize→224, center crop, normalize)
3. `openai/clip-vit-base-patch16`의 `get_image_features()`로 512차원 이미지 임베딩 추출 (**L2 정규화 미적용**)
4. `LogisticRegression(max_iter=2000, classes_=[0,1,2]=[CN,MCI,AD])`로 슬라이스별 확률 계산
5. 같은 MR_ID(subject)의 모든 슬라이스 확률을 **단순 평균**하여 subject 확률 산출

## 클래스 순서

`clf.classes_` = `[0, 1, 2]` = `[CN, MCI, AD]` — 반드시 이 순서를 사용하고, 하드코딩하지 말고 저장된 `classes_` 배열을 참조할 것.

## CLIP 모델 정보

- Model ID: `openai/clip-vit-base-patch16`
- Commit hash: `57c216476eefef5ab752ec549e440a49ae4ae5f3`
- image_size=224, patch_size=16, grid=14×14=196 patches + CLS token, hidden_size=768, num_attention_heads=12, projection_dim=512

## LR 설정

`{"C": 1.0, "max_iter": 2000, "penalty": "l2", "solver": "lbfgs", "class_weight": None}` — 전체 `get_params()`는 `metadata/model_config.json` 참조.

## 저장 파일 역할

| 경로 | 역할 |
|---|---|
| `models/clip_model/` | CLIP 모델 (safetensors) |
| `models/clip_processor/` | CLIP image processor 설정 |
| `models/clip_lr_classifier_new_run.joblib` | 학습된 LR (joblib) |
| `models/clip_lr_weights.npz` | LR coef/intercept/classes 순수 배열 |
| `data/manifests/slice_manifest_large_used_original.csv` | 실제 사용된 원본 manifest (정제 전, 무수정 복사) |
| `data/manifests/slice_manifest_portable.csv` | 상대경로 버전 manifest |
| `data/manifests/embedding_index.csv` | 임베딩 배열 행 ↔ MR_ID/slice_path 매핑 |
| `data/embeddings/train_embeds_checkpoint.npz`, `test_embeds_checkpoint.npz` | 학습/평가에 실제 사용된 임베딩 |
| `data/reference/test_probs_clip_new_run.csv` | new_run 기준 subject 확률 |
| `data/xai_samples/` | XAI 검증용 샘플 6 subject(152 slice) 이미지 및 logit/확률 |
| `code/` | 추론·검증·Grad-ECLIP 준비 코드 (다음 단계에서 생성) |

## 새 환경에서 로드·추론·검증

```python
from transformers import CLIPModel, CLIPProcessor
import joblib

clip_model = CLIPModel.from_pretrained("models/clip_model")
clip_processor = CLIPProcessor.from_pretrained("models/clip_processor")
clf = joblib.load("models/clip_lr_classifier_new_run.joblib")
```

자세한 추론 코드는 `code/inference_clip_lr.py` 참조 (다음 단계 생성 예정).

## Grad-ECLIP 적용 시 주의사항

- **목표 점수는 원 논문형 이미지-텍스트 유사도가 아니라 LR class logit**: `z_c = W_c f_CLIP(x) + b_c`
- **`torch.no_grad()`를 절대 사용하지 말 것** — gradient가 차단되어 Grad-ECLIP이 작동하지 않습니다.
- 최종 attention 위치: `clip_model.vision_model.encoder.layers[-1].self_attn` (q_proj/k_proj/v_proj/out_proj 모두 확인됨, 상세는 `metadata/xai_target_spec.json`)
- image_size=224, patch_size=16 → 14×14 grid + CLS token 1개, 총 197 토큰

## 알려진 누락·경고·재현성 제한

- **`OAS31378_MR_d0196`**: 원본 195명 분할 중 이 subject가 최종 `test_probs_clip.csv`(원본 및 new_run 모두)에서 누락되어 194명만 존재. 원인 미상, 임의 복원하지 않음.
- **정제 전 manifest 사용**: `slice_manifest_large.csv`(정제 전)가 실제 사용됨. `slice_manifest_large_clean.csv`(정제 후, 거의 빈 슬라이스 제거됨)는 사용되지 않음.
- **new_run 재현성 제한**: 이 패키지의 `clf`는 원본 `test_probs_clip.csv`를 만든 최초 실행의 `clf`가 아닙니다(해당 실행의 임베딩/객체는 캐시되지 않아 소실됨). 동일 코드로 새로 실행한 `new_run`이며:
  - 예측 클래스: 원본과 **100% 일치**
  - 확률값: 원본 대비 **최대 절대오차 0.0088** (환경/하드웨어 드리프트로 추정)
  - new_run 자체는 재실행 시 **완전히 결정론적으로 재현됨** (rerun 간 오차 0.0)
- 현재 성능 수치는 독립된 임상 검증을 의미하지 않습니다.

## 검증 이력

- clf shape 검증: `coef_=(3,512)`, `intercept_=(3,)`, `classes_=[0,1,2]` — 통과
- joblib 재로드 검증: 원본과 오차 0.0 — 통과
- processor 재로드 검증: pixel_values 오차 0.0 — 통과
- embedding_index 행 수 일치(assert) — 통과
- manifest 복사 SHA-256 일치 — 통과


## Grad-ECLIP 히트맵 구현 결과 (사후 업데이트)

`code/grad_eclip.py`로 실제 히트맵을 구현하고 검증했습니다. **핵심: attention 가중치는 사용하지 않습니다.**
CLIP ViT의 12개 vision attention head 중 9개가 이미지 내용과 무관하게 배경에 쏠리는 것으로 진단되어,
gradient(Grad-CAM 스타일)만 사용하는 방식으로 변경했습니다. 6개 검증 샘플 중 5개가 "전경 히트맵 평균 >
배경 히트맵 평균" 정량 기준을 통과했습니다. 상세 진단 과정과 실패 사례 분석은
`reports/grad_eclip_validation.md`를 참조하세요.

**주의**: 이 히트맵은 해부학적으로 검증되지 않았습니다. 임상적 해석 전 방사선의학 전문가의 정성 평가가 필요합니다.
