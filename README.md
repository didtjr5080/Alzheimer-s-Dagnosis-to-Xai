# CLIPtoXAI

Frozen CLIP image encoder + Logistic Regression 기반 Alzheimer MRI XAI 연구용 프로토타입입니다.

이 저장소는 기존 handoff 패키지의 모델/가중치/전처리 기준을 재학습 없이 사용하고, 다중 coronal MRI PNG 슬라이스를 입력받아 slice/subject 확률, 대표 슬라이스 XAI heatmap, PDF 보고서를 생성하는 Gradio UI를 제공합니다.

## 중요 주의사항

- 연구용 프로토타입입니다. 의료기기가 아니며 진단, 치료 결정, 개별 환자에 대한 임상 결론에 사용할 수 없습니다.
- 모델은 `openai/clip-vit-base-patch16` CLIP image encoder와 sklearn Logistic Regression 조합입니다.
- CLIP 임베딩은 512차원이며 L2 정규화를 추가로 적용하지 않습니다.
- 클래스 순서는 저장된 `clf.classes_` 기준으로 `[0, 1, 2] = [CN, MCI, AD]`를 검증해 사용합니다.
- subject 예측은 동일 subject의 슬라이스별 `predict_proba`를 단순 평균합니다.
- XAI target은 선택 클래스의 LR logit `z_c = W_c f_CLIP(x) + b_c`입니다.
- XAI forward에서는 `torch.no_grad()`를 사용하지 않습니다.

## 폴더 구조

```text
CLIPtoXAI/
├─ README.md
├─ .gitignore
├─ clip_lr_grad_eclip_handoff_v1_20260812_105027/
│  ├─ checksums.sha256
│  ├─ code/
│  │  ├─ verify_handoff.py
│  │  ├─ inference_clip_lr.py
│  │  ├─ grad_eclip.py
│  │  └─ gradient_smoke_test.py
│  ├─ data/
│  ├─ metadata/
│  ├─ models/
│  └─ reports/
├─ clip_xai_app/
│  ├─ app.py
│  ├─ README.md
│  ├─ requirements.txt
│  ├─ configs/
│  │  └─ explanation.yaml
│  ├─ src/
│  │  ├─ config.py
│  │  ├─ model_loader.py
│  │  ├─ inference.py
│  │  ├─ xai.py
│  │  ├─ visualization.py
│  │  ├─ report.py
│  │  ├─ schemas.py
│  │  └─ warnings.py
│  ├─ tests/
│  │  ├─ test_model_contract.py
│  │  ├─ test_inference_reference.py
│  │  ├─ test_subject_aggregation.py
│  │  ├─ test_xai_smoke.py
│  │  └─ test_report_generation.py
│  ├─ artifacts/
│  │  ├─ verification/
│  │  ├─ heatmaps/
│  │  └─ reports/
│  └─ handoff/
└─ graph_xai_extension/
   ├─ run_graph_xai.py         # standalone Gradio app (does not import/modify clip_xai_app)
   ├─ README.md
   ├─ graph_xai/                # region grid, masking, perturbation, ranking, Graph XAI, PDF report
   ├─ tests/
   └─ docs/                     # implementation report, integrity baselines, integration guide
```

## Graph XAI 확장 (`graph_xai_extension/`)

`clip_xai_app`의 CLIP+LR 분류기와 Grad-ECLIP CAM을 **읽기 전용**으로 재사용해서 만든 독립 확장입니다. 원본 `clip_xai_app` 코드/모델/가중치는 전혀 수정하지 않으며, 자체 폴더 안에서만 동작합니다.

주요 기능:

- 3×3 공간 구역 분할 + zero/mean/blur 마스킹 기반 perturbation 분석 (구역이 모델 출력에 얼마나 민감한지 측정 — 해부학적 의미 아님)
- 지지 근거(probability_drop > 0) / 억제 근거(< 0) / 절대 민감도(방향 무관) 3종 분리 순위
- 마스킹 방식 간 안정성 지표(Spearman/Kendall, 부호 일치율)와 CAM-perturbation 탐색적 일치도
- NetworkX/Plotly 기반 Graph XAI(구역 간 공간 인접성 그래프, 실제 신경 연결 아님)
- CN/MCI/AD 클래스 순서를 분류기의 `classes_`에서 직접 검증(하드코딩 금지) + 다중 샘플 배치 평가(피험자 단위 중복 제거)
- PDF 보고서 3종 모드: `clinical_summary`(의료진용 2~3쪽 요약), `technical_full`(연구자용 상세), `combined`(기본값, 요약 + 기술 부록), 연구 검토자 확인란 포함
- CSV/JSON/HTML 내보내기, 독립 Gradio UI

실행 명령:

```powershell
cd E:\Develop\CLIPtoXAI
python -m pip install -r graph_xai_extension\requirements_graph_xai.txt
python graph_xai_extension\run_graph_xai.py
```

자세한 구현 내역, 테스트 결과, 원본 파일 무결성 검증 기록은
[graph_xai_extension/README.md](graph_xai_extension/README.md)와
[graph_xai_extension/docs/IMPLEMENTATION_REPORT.md](graph_xai_extension/docs/IMPLEMENTATION_REPORT.md)를 참고하세요.

## 지금까지 구현한 것

### 1. Handoff 검증

초기 검증에서 handoff 패키지의 필수 파일, CLIP model/processor, LR classifier, NPZ weight, 기준 embedding/probability, subject 집계, gradient smoke test, checksum 검증을 통과했습니다.

실행 명령:

```powershell
cd E:\Develop\CLIPtoXAI
$env:PYTHONIOENCODING='utf-8'
python clip_lr_grad_eclip_handoff_v1_20260812_105027\code\verify_handoff.py clip_lr_grad_eclip_handoff_v1_20260812_105027
```

초기 결과 요약:

```text
EXPORT_OK: CLIP+LR handoff package is reproducible and Grad-ECLIP-ready
checksum_entries=202
missing=0
mismatch=0
SHA256_OK
```

참고: 이후 `code/inference_clip_lr.py`, `code/grad_eclip.py`에 앱 연동용 API를 추가했기 때문에 원본 `checksums.sha256` 기준으로는 해당 코드 파일 checksum이 달라질 수 있습니다. 모델, processor, LR classifier, embedding/data 파일은 변경하지 않았습니다.

### 2. Handoff 공개 API 추가

[inference_clip_lr.py](clip_lr_grad_eclip_handoff_v1_20260812_105027/code/inference_clip_lr.py)에 UI와 테스트에서 직접 사용할 수 있는 함수형 API를 추가했습니다.

```python
load_pipeline(root, device=None)
predict_slices(images, bundle)
aggregate_subject(slice_predictions)
generate_xai(image, target_class_idx, bundle)
select_representative_slices(slice_predictions, max_count=3)
```

추가 동작:

- PIL Image 입력 지원
- slice별 logits/probabilities/embedding 반환
- subject 평균 확률 계산
- 단일 슬라이스 입력 경고
- 대표 슬라이스 선택: 중앙 slice, 최종 클래스 logit 최대 slice, 불확실성 최대 slice
- XAI 결과 `heatmap_224`, logits, probabilities 반환

### 3. Grad-ECLIP 응용형 XAI 호환 수정

[grad_eclip.py](clip_lr_grad_eclip_handoff_v1_20260812_105027/code/grad_eclip.py)에 PIL Image 기반 `explain_image()`와 `explain_pixel_values()`를 추가했습니다.

현재 XAI 명칭:

```text
LR class logit 기반 CLIP ViT gradient heatmap
Grad-ECLIP 응용형 - attention 결합 제거
```

현재 설치된 `transformers 4.56.2`와 호환되도록 `CLIPVisionTransformer.forward()`에 전달하던 `return_dict=True`를 제거했습니다.

### 4. Gradio UI

[clip_xai_app/app.py](clip_xai_app/app.py)에 Gradio UI를 구현했습니다.

UI 기능:

- coronal MRI PNG 다중 업로드
- MR_ID 입력, 비어 있으면 session ID 자동 생성
- 분석 버튼
- subject 결과 표시
- 클래스별 subject 확률 표
- slice별 확률 표
- 대표 슬라이스 원본/heatmap/overlay 갤러리
- PDF 보고서 다운로드
- traceback은 UI에 노출하지 않고 `clip_xai_app/artifacts/verification/ui_errors.log`에 기록

실행 명령:

```powershell
cd E:\Develop\CLIPtoXAI
python clip_xai_app\app.py
```

접속 URL:

```text
http://127.0.0.1:7860/
```

### 5. PDF 보고서

[clip_xai_app/src/report.py](clip_xai_app/src/report.py)에 한글 PDF 보고서 생성을 구현했습니다.

보고서 기능:

- A4 용지
- 맑은 고딕 폰트 등록
- footer: `Research Use Only | 보고서 ID: ... | Page N`
- KST ISO 8601 생성 시각
- PDF metadata title/author/subject/CreationDate/ModDate 설정
- 단일 슬라이스와 다중 슬라이스 subject 평균 분기
- 실제 파일명에서 MR_ID 추출
- 서로 다른 MR_ID가 섞이면 PDF 생성 실패
- 원본/히트맵/오버레이 이미지 포함
- 모델 정보 표 포함
- 모델 예측 근거 섹션 포함
- Top-1, Top-2, 확률 차이 `%p` 계산
- 불확실성 설명 규칙 적용
- 검증 성능과 알려진 제한 포함

단일 슬라이스 기준 검증 샘플:

```text
OAS30009_MR_d2457_cor094.png

CN  = 0.9180
MCI = 0.0619
AD  = 0.0200
Top-1 = CN
Top-2 = MCI
Margin = 85.61%p
```

다중 슬라이스 subject 평균 검증 샘플:

```text
OAS30009_MR_d2457

CN  = 0.8962
MCI = 0.0861
AD  = 0.0176
Predicted class = CN
```

### 6. 예측 근거 규칙

[clip_xai_app/configs/explanation.yaml](clip_xai_app/configs/explanation.yaml)에 설명 규칙 버전을 추가했습니다.

```yaml
version: "explanation-rule-v1"
top1_low_threshold: 0.50
margin_boundary_threshold: 0.10
margin_moderate_threshold: 0.25
clinical_validation: false
```

현재 규칙:

- `top1 < 0.50`: 높은 불확실성
- `margin < 0.10`: 경계 영역
- `margin < 0.25`: 중간 수준 분리
- 그 외: 상대적으로 큰 클래스 분리

이 규칙은 모델 출력값의 상대적 차이를 설명하기 위한 연구용 규칙이며 임상적 확신을 의미하지 않습니다.

## 주요 실행/검증 명령

문법 검사:

```powershell
python -m py_compile clip_xai_app\app.py clip_xai_app\src\report.py
```

단일 샘플 PDF 생성 스모크:

```powershell
$env:PYTHONIOENCODING='utf-8'
python -c "import sys; sys.path.insert(0, 'clip_xai_app'); import app; p='clip_lr_grad_eclip_handoff_v1_20260812_105027/data/xai_samples/images/OAS30009_MR_d2457_cor094.png'; r=app.analyze([p], 'OAS30009_MR_d2457'); print(r[4]); print(r[5])"
```

다중 슬라이스 PDF 생성 스모크:

```powershell
$env:PYTHONIOENCODING='utf-8'
python -c "import sys; from pathlib import Path; sys.path.insert(0, 'clip_xai_app'); import app; root=Path('clip_lr_grad_eclip_handoff_v1_20260812_105027/data/xai_samples/images'); files=[str(p) for p in sorted(root.glob('OAS30009_MR_d2457_*.png'))]; r=app.analyze(files, 'OAS30009_MR_d2457'); print(r[4]); print(r[5])"
```

PDF 텍스트 검증에는 `pypdf`, 페이지 렌더링 검증에는 `PyMuPDF(fitz)`를 사용했습니다.

## 의존성

주요 설치 패키지:

```text
gradio 6.4.0
fastapi 0.141.1
starlette 0.52.1
python-multipart 0.0.32
torch 2.5.1+cu121
transformers 4.56.2
scikit-learn 1.5.2
reportlab 4.2.2
pypdf
PyMuPDF
```

설치:

```powershell
python -m pip install -r clip_xai_app\requirements.txt
```

## Git/데이터 관리

[.gitignore](.gitignore)는 생성 artifact와 대용량/민감 가능성이 있는 로컬 payload를 제외하도록 설정했습니다.

무시 대상:

- `clip_xai_app/artifacts/*` 생성물
- `clip_xai_app/handoff/*`
- handoff 패키지의 `models/`
- handoff 패키지의 `data/embeddings/`
- handoff 패키지의 `data/xai_samples/images/`

## 남은 작업

- 테스트 파일에 실제 pytest 케이스 작성
- PDF 보고서 생성 로직의 단위 테스트 추가
- UI end-to-end 테스트 자동화
- 전체 `pytest -q` 통과 기준 정리
- XAI 전경/배경 QC 수치 연동 가능 여부 검토
- 전체 슬라이스 XAI 일관성 검증은 별도 연구 과제로 분리
