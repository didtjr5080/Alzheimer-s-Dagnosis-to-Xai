# Codex 작업지시서: XAI 다음 단계 — 기준선 고정 및 다중 모델 통합

아래 작업을 `E:\Develop\CLIPtoXAI` 저장소에서 순서대로 수행하라.

참고 저장소:

- GitHub: <https://github.com/didtjr5080/Alzheimer-s-Dagnosis-to-Xai>
- 점검 기준 브랜치: `main`
- 점검 당시 커밋: `ad706dc` (`Initial commit: CLIP to XAI project`)

## 1. 이번 단계의 목적

기존 `CLIP image embedding + LogisticRegression + gradient heatmap` 앱을 회귀 가능한 기준선으로 고정한 후, 로컬 `merged_project`에 전달된 CLIP·3D CNN·GBM·최종 앙상블 산출물을 검증 가능한 어댑터 구조로 연결한다.

이번 단계의 완료 기준은 다음과 같다.

1. 기존 CLIP 기준선의 실제 자동 테스트가 작성되어 통과한다.
2. `merged_project` 산출물의 입출력 계약, 클래스 순서, 전처리, 데이터 분할 정보를 검증한다.
3. CLIP·3D CNN·GBM·앙상블을 동일한 내부 결과 스키마로 호출할 수 있다.
4. 사용 가능한 모델만 선택해 실행할 수 있고, 누락된 모델은 앱 전체를 중단시키지 않는다.
5. 이미지 분기 XAI와 3D CNN XAI가 모델의 실제 예측 점수를 대상으로 생성된다.
6. GBM/앙상블 설명은 영상 heatmap으로 위장하지 않고 특징 기여도 또는 모델별 기여도로 분리한다.
7. UI와 PDF에서 slice, subject, 모델별 결과와 최종 앙상블 결과를 혼동하지 않는다.
8. 모든 결과는 연구용 모델 출력으로 표현하며 임상 진단으로 단정하지 않는다.

## 2. 현재 저장소에서 확인된 사실

실제 파일을 다시 확인한 뒤 아래 내용을 검증 기준으로 사용하라.

### 기존 구현

- `clip_xai_app/app.py`: Gradio UI
- `clip_xai_app/src/model_loader.py`: 기존 handoff 로더
- `clip_xai_app/src/inference.py`: slice 예측 및 subject 평균
- `clip_xai_app/src/xai.py`: 대표 슬라이스 XAI
- `clip_xai_app/src/report.py`: PDF 보고서
- `clip_lr_grad_eclip_handoff_v1_20260812_105027/`: 기존 CLIP+LR handoff
- 기존 CLIP 모델 계약:
  - `openai/clip-vit-base-patch16`
  - 입력 `224×224`
  - patch `16×16`, grid `14×14`
  - 이미지 임베딩 512차원
  - 추가 L2 정규화 없음
  - 클래스 순서는 `clf.classes_`에서 읽으며 `[CN, MCI, AD]`
  - subject 확률은 slice별 `predict_proba` 단순 평균
  - XAI target은 선택 클래스 LR logit
  - XAI forward에서 `torch.no_grad()` 사용 금지

### 반드시 먼저 수정해야 하는 문제

- `clip_xai_app/tests/`의 다음 파일들은 실제 테스트가 아니라 한 줄짜리 자리표시자다.
  - `test_model_contract.py`
  - `test_inference_reference.py`
  - `test_subject_aggregation.py`
  - `test_xai_smoke.py`
  - `test_report_generation.py`
- `clip_xai_app/app.py`에는 아직 `Confidence` 표현이 남아 있다.
- 저장소 README의 “검증 통과” 설명과 실제 자동 테스트 상태가 일치하지 않는다.
- handoff의 `models/clip_model`, `models/clip_processor`, LR joblib 등 대형 로컬 산출물은 Git 트리에 없을 수 있으므로 로컬 의존성을 명확히 검사해야 한다.
- 기존 XAI는 원 논문의 완전한 Grad-ECLIP이 아니라 attention 결합을 제거한 LR-logit 기반 CLIP ViT gradient heatmap이다.

### 새로 병합된 산출물

예상 경로:

```text
E:\Develop\CLIPtoXAI\merged_project
```

예상 항목:

```text
merged_project/
├─ models/
│  ├─ gbm_stage1.txt
│  ├─ gbm_stage2.txt
│  └─ gbm_baseline.txt
├─ checkpoints/
│  ├─ clip_best.pt
│  ├─ clip_latest.pt
│  ├─ 3dcnn_best.pt
│  └─ 3dcnn_latest.pt
├─ gradcam_results/
│  ├─ gradcam_summary.csv
│  └─ gradcam_*.png
├─ slices_multi/
├─ mri_3d_cache/
├─ clip_test_probs.csv
├─ 3dcnn_test_probs.csv
├─ gbm_hierarchical_test_probs.csv
├─ final_ensemble_results.csv
├─ unified_manifest.csv
├─ unified_slice_manifest.csv
├─ unified_final_features_gbm.csv
└─ task_A_column_summary.txt
```

이 구조는 예상일 뿐이다. 실제 로컬 파일과 체크포인트 내부 구조를 근거로 계약을 확정하라.

## 3. 용어와 범위 제한

- CLIP과 3D CNN이 모두 MRI에서 파생된 입력만 사용한다면 이를 곧바로 “임상 멀티모달 XAI”라고 부르지 말 것.
- 임상 변수, 인지검사, PET 등 독립 모달리티의 실제 입력·매칭·분기 모델이 확인되기 전에는 다음 표현을 사용한다.

```text
다중 모델 MRI 앙상블 XAI 연구용 프로토타입
```

- GBM 입력 열에 임상 변수가 실제 포함되어 있고 시점 매칭·subject 분할 누수 검증까지 완료된 경우에만 “MRI+임상 멀티모달” 표현을 별도 검토한다.
- heatmap은 병변 지도, 진단 부위 또는 의학적 인과 근거가 아니다.
- UI/PDF 용어는 다음처럼 제한한다.
  - `진단 결과` → `모델 예측 결과`
  - `신뢰도` 또는 `Confidence` → `모델 출력 확률`
  - `병변 위치` → `모델 점수 기여 영역`
  - `진단 근거` → `모델 예측 근거`

## 4. 절대 준수 사항

- 기존 기준선 동작과 기준 출력값을 테스트로 고정하기 전에 구조를 대규모로 변경하지 말 것.
- 기존 handoff와 `merged_project` 원본 산출물을 수정·이동·삭제하지 말 것.
- 체크포인트 형식을 추측해 모델 클래스를 임의 구현하지 말 것.
- 클래스 순서를 파일명이나 관습으로 하드코딩하지 말고, checkpoint·manifest·CSV에서 교차검증할 것.
- ADNI/OASIS 데이터를 같은 subject로 잘못 결합하지 말 것.
- slice 단위 데이터를 무작위로 나눠 동일 subject가 train/test에 중복되는 누수를 허용하지 말 것.
- 모델 간 확률 열의 클래스 순서를 검증하지 않은 상태로 평균하지 말 것.
- `clip_best.pt`와 기존 CLIP+LR handoff가 같은 모델이라고 가정하지 말 것.
- 3D CNN에 2D PNG를 임의 적층하여 입력하지 말고, 학습 때 사용한 `.npy` shape와 전처리를 확인할 것.
- GBM feature 열 순서를 재정렬하거나 누락값을 임의로 0 처리하지 말 것.
- 기존 Grad-CAM PNG를 새 입력에 대한 동적 XAI인 것처럼 표시하지 말 것.
- 모델 로딩 실패를 숨긴 채 더미 확률을 반환하지 말 것.
- 오류 시 traceback 전체를 UI에 노출하지 말고, 사용자 메시지와 로컬 상세 로그를 분리할 것.
- 실제 테스트 데이터의 기대값을 현재 출력에 맞춰 임의 변경하지 말 것.

## 5. 구현 전략

다음 구조를 목표로 하되, 기존 파일과 겹치면 최소 변경으로 적용한다.

```text
clip_xai_app/
├─ app.py
├─ configs/
│  ├─ explanation.yaml
│  └─ model_registry.yaml
├─ src/
│  ├─ contracts.py
│  ├─ model_registry.py
│  ├─ adapters/
│  │  ├─ base.py
│  │  ├─ legacy_clip_lr.py
│  │  ├─ merged_clip.py
│  │  ├─ cnn3d.py
│  │  ├─ gbm.py
│  │  └─ ensemble.py
│  ├─ xai/
│  │  ├─ clip_gradient.py
│  │  ├─ cnn3d_gradcam.py
│  │  ├─ gbm_contribution.py
│  │  └─ ensemble_evidence.py
│  ├─ inference.py
│  ├─ report.py
│  └─ warnings.py
├─ tests/
│  ├─ fixtures/
│  ├─ test_model_contract.py
│  ├─ test_inference_reference.py
│  ├─ test_subject_aggregation.py
│  ├─ test_xai_smoke.py
│  ├─ test_report_generation.py
│  ├─ test_merged_artifact_contract.py
│  ├─ test_adapter_parity.py
│  └─ test_ensemble_contract.py
└─ artifacts/
   └─ verification/
```

기존 `src/xai.py`처럼 파일과 패키지 이름이 충돌할 수 있으므로, 실제 변경 전에 import 영향도를 검사하고 안전한 이름을 선택하라.

## 6. 단계별 작업

### 단계 A. 저장소와 로컬 산출물 감사

1. 프로젝트 루트, 현재 브랜치, `git status --short`, 최근 커밋을 기록한다.
2. 사용자 변경사항을 보존하고 관련 없는 파일을 수정하지 않는다.
3. GitHub `main`과 현재 로컬 브랜치 차이를 확인한다.
4. 기존 handoff 필수 파일 존재 여부를 검사한다.
5. `merged_project`의 파일 목록, 크기, SHA-256, 확장자별 개수를 기록한다.
6. 다음 파일의 헤더와 스키마를 읽기 전용으로 조사한다.
   - `unified_manifest.csv`
   - `unified_slice_manifest.csv`
   - `unified_final_features_gbm.csv`
   - `clip_test_probs.csv`
   - `3dcnn_test_probs.csv`
   - `gbm_hierarchical_test_probs.csv`
   - `final_ensemble_results.csv`
   - `task_A_column_summary.txt`
7. 모든 결과 CSV에서 다음을 교차검증한다.
   - subject/MR/scan 식별자
   - 실제 라벨
   - 데이터 출처 ADNI/OASIS
   - split 열
   - 클래스별 확률 열
   - 예측 클래스
   - 동일 샘플 수와 정렬 기준
8. `artifacts/verification/merged_artifact_audit.json`과 사람이 읽을 수 있는 Markdown 요약을 생성한다.

### 단계 B. 실패하는 회귀 테스트부터 작성

기존 자리표시자 테스트를 실제 테스트로 교체한다. 구현 수정 전에 테스트를 먼저 실행해 예상한 실패를 확인하고, 실패 원인을 기록한 뒤 구현을 수정한다.

필수 테스트:

1. 모델 계약
   - 모델 ID와 commit
   - 입력 크기, patch/grid
   - 임베딩 차원
   - L2 정규화 여부
   - `clf.classes_`와 클래스 이름 매핑
2. 기준 추론
   - 기준 샘플 `OAS30009_MR_d2457_cor094.png`
   - 단일 slice 기대값: CN `0.9180`, MCI `0.0619`, AD `0.0200`
   - 허용오차를 근거 있게 지정하고 과도하게 넓히지 않는다.
3. subject 집계
   - 다중 slice 단순 평균
   - 기대값: CN `0.8962`, MCI `0.0861`, AD `0.0176`
   - slice 입력 순서가 결과를 바꾸지 않는지
4. XAI smoke
   - `requires_grad` 유지
   - 출력 shape `224×224`
   - finite 값
   - 정규화 범위
   - 상수 heatmap이 아닌지
5. PDF 보고서
   - 단일/다중 slice 분기
   - slice 확률과 subject 확률 혼용 금지
   - `신뢰도`, `Confidence`, `진단 결과`, `병변 위치` 금지
   - 연구용 경고 포함
   - 실제 확률과 PDF 표 값 일치
6. 혼합 MR_ID 입력 거부
7. 누락 handoff에 대한 명확한 오류

테스트 실패가 환경·대형 파일 누락 때문이면 skip으로 숨기지 말고, `integration` marker 또는 명시적 사전조건으로 분리한다.

### 단계 C. `merged_project` 모델 계약 추출

다음 체크포인트를 CPU로 안전하게 검사한다.

- `clip_best.pt`
- `clip_latest.pt`
- `3dcnn_best.pt`
- `3dcnn_latest.pt`

각 파일에 대해 다음을 기록한다.

- 파일 크기와 SHA-256
- 최상위 객체 타입
- key 목록
- `state_dict` 존재 여부
- tensor key와 shape 요약
- epoch, metric, class mapping, preprocessing metadata
- best/latest 관계
- 저장된 optimizer 포함 여부
- 모델 아키텍처 재구성에 필요한 정보의 충족 여부

주의:

- `clip_best.pt`가 수 KB 수준이면 전체 CLIP 가중치가 아니라 head 또는 metadata일 가능성이 높다.
- 모델 정의 코드나 pretrained identifier가 없으면 로더를 추측 구현하지 말고 `BLOCKED_MODEL_CONTRACT`로 보고한다.
- `3dcnn_best.pt`와 `latest`의 파일 크기가 크게 다르면 전체 checkpoint와 state dict 저장 방식 차이를 확인한다.

GBM 모델은 LightGBM 텍스트 모델로 예상하되 실제 헤더를 확인한다. 다음을 검증한다.

- LightGBM/다른 GBM 형식
- feature name과 개수
- 클래스 개수
- objective
- stage1/stage2 계층 규칙
- `unified_final_features_gbm.csv` 열 순서와 모델 feature 순서 일치
- 결측치 처리 규칙

검증 결과를 `configs/model_registry.yaml`에 기록하되, 확인되지 않은 값은 추측하지 말고 `null`과 차단 사유를 사용한다.

### 단계 D. 공통 결과 계약 정의

최소한 다음 데이터 구조를 정의한다.

```python
PredictionResult
- model_id
- model_version
- level: slice | subject
- sample_id
- class_names
- probabilities
- predicted_class
- warnings
- provenance

XAIResult
- model_id
- target_class
- target_score_type
- method_name
- source_level
- heatmap_2d 또는 heatmap_3d
- feature_contributions
- model_contributions
- representative_inputs
- validation_metrics
- warnings
```

검증 규칙:

- 확률은 finite이며 각 행의 합이 허용오차 내에서 1이어야 한다.
- 클래스 순서와 확률 배열 길이가 일치해야 한다.
- slice 결과와 subject 결과를 타입 수준에서 구분한다.
- XAI method와 target score를 반드시 함께 저장한다.
- heatmap이 없는 모델에 빈 이미지나 가짜 heatmap을 생성하지 않는다.

### 단계 E. 모델 어댑터 구현

다음 순서로 구현한다.

1. `LegacyClipLRAdapter`
   - 기존 handoff 함수를 감싼다.
   - 기존 출력과 수치적으로 동일해야 한다.
   - 기존 코드를 복제하지 않는다.
2. `MergedClipAdapter`
   - 계약이 충분한 경우에만 구현한다.
   - 기존 handoff CLIP과 동일 모델인지 체크포인트 key와 전처리로 판정한다.
3. `CNN3DAdapter`
   - `.npy`의 실제 shape/dtype과 학습 전처리를 사용한다.
   - subject 수준 출력만 반환한다.
4. `GBMAdapter`
   - 정확한 feature 순서 검증을 통과해야 실행한다.
   - stage1/stage2 계층형 로직을 원본 결과 CSV와 대조한다.
5. `EnsembleAdapter`
   - `final_ensemble_results.csv`에서 실제 결합 규칙을 복원할 수 있을 때만 구현한다.
   - 평균, 가중 평균, stacking 중 하나를 임의 선택하지 않는다.
   - 입력 모델 중 하나가 없을 때 정책을 명시한다.

각 어댑터는 다음 상태 중 하나를 반환할 수 있어야 한다.

- `AVAILABLE`
- `UNAVAILABLE_MISSING_FILES`
- `BLOCKED_MODEL_CONTRACT`
- `BLOCKED_PREPROCESSING_CONTRACT`
- `FAILED_VALIDATION`

### 단계 F. 모델별 XAI 구현

#### 기존 CLIP+LR

- 기존 LR class logit 기반 CLIP ViT gradient heatmap을 유지한다.
- 메서드 명칭과 attention 제거 사실을 UI/PDF에 표시한다.
- 기존 5/6 전경>배경 검증 결과를 새로운 임상적 유효성으로 확대 해석하지 않는다.

#### 새 CLIP 체크포인트

- 모델 구조와 예측 target이 확인된 경우에만 동적 XAI를 연결한다.
- `clip_best.pt`가 head-only라면 backbone과 head 결합 경로를 명시한다.
- target은 최종 예측에 실제 사용되는 logit이어야 한다.

#### 3D CNN

- 마지막 합성곱 feature map에 대한 3D Grad-CAM을 구현한다.
- 모델 forward hook과 backward gradient를 사용한다.
- 출력 CAM을 입력 volume shape로 보간한다.
- axial/coronal/sagittal 중 실제 UI에서 표시할 방향과 voxel index를 기록한다.
- 대표 coronal slice 선택 기준을 명시한다.
- 미리 생성된 `gradcam_results`와 동일 샘플에 대해 정성·shape·finite 검증을 수행한다.

#### GBM

- 영상 heatmap을 만들지 않는다.
- LightGBM `pred_contrib` 또는 검증된 SHAP TreeExplainer 중 환경에서 재현 가능한 방식을 사용한다.
- base value와 feature contribution의 합이 raw model output을 재구성하는지 확인한다.
- 상위 양/음 기여 feature와 실제 feature value를 제공한다.
- feature가 CLIP/3D CNN 확률이라면 이를 해부학적 특징으로 표현하지 않는다.

#### 최종 앙상블

- 모델별 최종 클래스 확률과 결합 가중치를 표로 제공한다.
- final logit/probability에 대한 모델별 기여도를 계산 가능한 경우에만 표시한다.
- 서로 다른 XAI map을 근거 없이 단순 평균하지 않는다.
- 영상 XAI와 tabular/model contribution을 별도 패널로 유지한다.

### 단계 G. UI 통합

기존 Gradio 앱에 다음을 추가한다.

- 실행 가능한 모델 목록과 상태 표시
- 분석 모드:
  - 기존 CLIP 기준선
  - 새 CLIP
  - 3D CNN
  - GBM
  - 최종 앙상블
- 입력 요구사항을 모델별로 표시
- slice 확률과 subject 확률을 별도 표로 표시
- 모델별 확률과 최종 앙상블 확률을 별도 표로 표시
- XAI 패널:
  - 2D CLIP heatmap
  - 3D CNN 대표 slice Grad-CAM
  - GBM feature contribution
  - ensemble model contribution
- 누락된 모델은 비활성화하고 원인을 표시
- 단일 slice 입력은 3D/subject 분석으로 처리하지 않는다.
- UI의 모든 `Confidence`/`신뢰도` 표현을 `모델 출력 확률`로 변경한다.

초기 실행 시 모든 대형 모델을 동시에 로드하지 말고 lazy loading과 캐시를 사용한다. CPU 환경에서도 기준선 모드가 실행되어야 한다.

### 단계 H. PDF 보고서 통합

보고서에는 다음 섹션을 구분한다.

1. 입력 및 분석 범위
2. 모델별 출력 확률
3. 최종 앙상블 출력
4. 영상 모델 XAI
5. GBM 특징 기여도
6. 모델 간 일치/불일치
7. 제한사항과 연구용 경고
8. 모델·데이터·설명 방법 provenance

필수 규칙:

- 단일 slice이면 subject 평균이라고 쓰지 않는다.
- 특정 모델이 실패하면 결과에서 제외된 사실과 이유를 표시한다.
- 확률을 임상적 확률 또는 질병 보유 확률로 표현하지 않는다.
- AD/CN/MCI 클래스가 모델마다 다르면 앙상블하지 않는다.
- 해마·내측 측두엽 등 해부학적 명칭은 자동 XAI map만으로 단정하지 않는다.
- 텍스트 설명은 계산된 값과 검증된 메타데이터에서만 생성한다.

### 단계 I. 검증과 회귀 테스트

다음 테스트를 추가하고 실행한다.

- 기존 CLIP 출력 parity
- 각 체크포인트 로딩 smoke
- CSV 기준 확률 재현
- class order 불일치 감지
- subject 누수 검사
- GBM feature order 불일치 감지
- 3D volume shape 불일치 감지
- 3D Grad-CAM finite/shape/nonconstant
- GBM contribution 합 검증
- ensemble 기준 CSV 재현
- 한 모델 누락 시 degraded mode
- 동일 MR_ID 검증
- PDF 금지 표현 검사
- UI import 및 최소 분석 smoke

테스트 명령은 현재 환경에 맞게 문서화하고, 실행하지 못한 테스트는 이유와 필요한 환경을 구분해 보고한다.

## 7. 단계 중단 기준

다음 중 하나라도 발생하면 해당 모델 통합을 중단하고, 기존 기준선까지는 정상 유지한다.

- 모델 아키텍처 정의 부재
- checkpoint와 코드 key 불일치
- 클래스 순서 확인 불가
- 전처리 shape/정규화 확인 불가
- 데이터 split 누수 의심
- GBM feature 순서 확인 불가
- ensemble 결합 규칙 확인 불가
- 기준 결과 CSV를 허용오차 내에서 재현하지 못함

중단 시 필요한 자료를 구체적으로 요청한다.

- 학습 코드 또는 모델 클래스
- checkpoint 저장 코드
- 전처리 설정
- 클래스 매핑
- split manifest
- ensemble weight/stacker
- feature schema
- 버전이 고정된 requirements/environment

## 8. 생성할 문서

다음을 `docs/` 또는 `clip_xai_app/artifacts/verification/`에 생성한다.

- `merged_artifact_audit.md`
- `model_contracts.md`
- `xai_method_registry.md`
- `integration_validation.md`
- `missing_handoff_requirements.md`

README에는 실제 검증된 상태만 반영한다. 아직 자리표시자인 기능을 완료된 것으로 쓰지 않는다.

## 9. 완료 보고 형식

### 구현 결과

- 완료된 모델 어댑터
- 차단된 모델 어댑터와 이유
- 기존 CLIP 기준선 parity 결과
- 최종 앙상블 재현 여부

### XAI 결과

| 모델 | 예측 수준 | XAI 방법 | target score | 검증 상태 | 제한사항 |
|---|---|---|---|---|---|
| 기존 CLIP+LR | | | | | |
| 새 CLIP | | | | | |
| 3D CNN | | | | | |
| GBM | | | | | |
| Ensemble | | | | | |

### 테스트 결과

- 전체 테스트 수
- 통과/실패/스킵 수
- 실패 원인
- 실행하지 못한 통합 테스트

### 데이터 무결성

- 클래스 순서 검증 결과
- subject split 누수 검사 결과
- 모델별 기준 CSV 재현 오차
- 누락·중복·손상 파일

### 변경 파일

- 생성 파일
- 수정 파일
- 수정 이유
- 사용자 기존 변경사항 보존 여부

### 남은 TODO

- 즉시 필요한 학습팀 전달물
- UI/PDF 후속 개선
- 실제 멀티모달 확장을 위한 데이터 요건

## 10. 체크리스트

- [ ] Git 상태와 사용자 변경사항 확인
- [ ] 기존 handoff와 `merged_project` 감사
- [ ] 자리표시자 테스트를 실제 회귀 테스트로 교체
- [ ] 실패하는 테스트를 먼저 실행하고 실패 이유 기록
- [ ] 기존 CLIP 기준 출력 고정
- [ ] UI/PDF의 `Confidence`·`신뢰도` 표현 제거
- [ ] 새 체크포인트 내부 구조 검사
- [ ] 클래스·전처리·feature·ensemble 계약 확인
- [ ] 공통 Prediction/XAI 스키마 구현
- [ ] 기존 CLIP 어댑터 parity 검증
- [ ] 새 CLIP 어댑터 구현 또는 차단 사유 보고
- [ ] 3D CNN 어댑터 및 3D Grad-CAM 구현 또는 차단 사유 보고
- [ ] GBM 어댑터 및 feature contribution 구현 또는 차단 사유 보고
- [ ] 앙상블 규칙 재현 또는 차단 사유 보고
- [ ] 모델별 UI 패널 통합
- [ ] PDF 모델별 근거 섹션 통합
- [ ] 회귀·계약·XAI·보고서 테스트 실행
- [ ] README와 검증 문서를 실제 상태에 맞게 수정

## 11. TODO 우선순위

1. **P0:** 실제 회귀 테스트 작성 및 기존 CLIP 기준선 고정
2. **P0:** `merged_project` 모델·데이터·클래스·전처리 계약 감사
3. **P0:** UI/PDF 잘못된 `Confidence`·`신뢰도` 표현 수정
4. **P1:** 공통 어댑터와 결과 스키마 구현
5. **P1:** 3D CNN·GBM·앙상블 재현 검증
6. **P1:** 모델별 XAI 구현 및 검증
7. **P2:** Gradio UI와 PDF 통합
8. **P2:** 실제 MRI+임상 멀티모달 확장 가능성 평가

작업을 한 번에 완료했다고 가정하지 말고, P0 감사 결과에서 계약이 부족한 모델은 안전하게 차단하라. 기존 기준선이 계속 실행 가능한 상태를 유지하면서, 검증된 모델부터 순차적으로 통합하라.
