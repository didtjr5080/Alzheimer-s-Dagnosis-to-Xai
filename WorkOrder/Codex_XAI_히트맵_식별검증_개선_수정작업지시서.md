# Codex 수정 작업지시서: XAI 히트맵 식별·검증·보고 개선

아래 수정 작업을 `E:\Develop\CLIPtoXAI` 저장소에서 수행하라.

참고 저장소:

- <https://github.com/didtjr5080/Alzheimer-s-Dagnosis-to-Xai>
- 대상 앱: `clip_xai_app`
- 기존 handoff: `clip_lr_grad_eclip_handoff_v1_20260812_105027`

## 1. 수정 목적

현재 XAI 결과는 동일 subject의 서로 다른 slice가 subject 이름만으로 저장되어, 원본·heatmap·overlay·확률·target 클래스가 잘못 대응될 위험이 있다. 이 문제는 확인된 `cor084`와 `cor088`에 한정되지 않는다. 전체 ADNI/OASIS 데이터, 모든 subject·scan·slice·모델·실행·target에서 동일한 혼동이 발생하지 않도록 일반화된 식별 및 무결성 체계를 구현한다.

1. 정확한 원본 slice
2. 실제 라벨
3. 예측 클래스
4. XAI target 클래스와 target score 종류
5. slice 확률과 subject 확률
6. 원시 heatmap과 overlay
7. brain-mask 기반 전경·배경 활성도
8. 정답/오답 여부
9. XAI 생성 방법과 모델 버전

이번 수정은 의료 진단 기능을 추가하는 작업이 아니다. 모델이 어떤 입력 영역에 반응했는지를 연구용으로 더 정확하게 기록하고 검증하는 작업이다.

## 2. 확인된 문제 사례와 적용 범위

다음 사례를 필수 회귀 fixture로 사용한다.

```text
subject_id: OAS30100_MR_d0158
slice_file: OAS30100_MR_d0158_cor088.png
true_label: AD
predicted_class: CN
CN_prob: 0.5537639592790997
MCI_prob: 0.35217214265480157
AD_prob: 0.09406389806609877
prediction_correct: false
XAI target: CN
```

기존 검증 이미지 `OAS30100_MR_d0158_incorrect_gradeclip.png`에 표시된 확률은 다음과 같다.

```text
CN ≈ 0.4191
MCI ≈ 0.4529
AD ≈ 0.1280
predicted_class: MCI
```

이 확률은 `cor088`이 아니라 `cor084`의 값과 일치한다. 즉, subject 이름만 있는 검증 이미지가 어떤 slice를 사용했는지 알기 어렵고, 현재 UI의 `cor088` 결과와 혼동될 수 있다.

`cor084`와 `cor088`은 문제를 재현하기 위한 최소 회귀 사례일 뿐이다. 수정 범위는 다음 전체 조합이다.

```text
dataset/source × subject × session/visit × scan/MR_ID × slice/volume
× model/checkpoint × preprocessing version × XAI method × target class × run
```

하나라도 다르면 별개의 분석 결과로 취급해야 한다.

### 현재 `cor088` overlay 참고 분석

원본 밝기 기반 임시 brain mask와 색상 overlay에서 계산한 참고값은 다음과 같다.

```text
brain mask 비율: 21.72%
background 비율: 78.28%
전경/배경 고활성 평균 비율: 약 5.33
상위 5% 고활성 영역 중 배경 비율: 약 22.16%
최대 활성 좌표: (101, 152)
최대 활성 brain mask 내부 여부: true
```

이 값은 원시 heatmap이 아니라 overlay 색상에서 근사한 값이므로 정답값으로 사용하지 말고, 수정 후 원시 heatmap을 기준으로 다시 계산하라.

## 3. 절대 준수 사항

- 기존 모델 가중치, processor, classifier, manifest를 수정하지 말 것.
- 기존 추론 확률을 XAI 품질 개선을 이유로 변경하지 말 것.
- 원본 검증 이미지를 삭제하거나 덮어쓰지 말 것.
- 실제 라벨과 예측 클래스가 다르면 `incorrect`를 명시할 것.
- 오분류 heatmap을 질병의 근거로 설명하지 말 것.
- heatmap의 붉은 영역을 병변 위치 또는 해부학적 이상이라고 단정하지 말 것.
- raw heatmap이 없는데 overlay에서 역산한 값을 공식 검증값으로 저장하지 말 것.
- subject ID만으로 slice-level XAI 파일명을 만들지 말 것.
- `subject_id + slice_index`만으로도 전역 고유성을 가정하지 말 것.
- ADNI와 OASIS처럼 데이터셋이 다르면 유사한 ID나 파일명이 있어도 같은 샘플로 합치지 말 것.
- 동일 subject의 서로 다른 방문·촬영·시리즈를 같은 scan으로 취급하지 말 것.
- 동일 입력이라도 모델 checkpoint, 전처리 버전, target 클래스 또는 XAI 방법이 다르면 기존 결과를 재사용하지 말 것.
- 화면 정렬 순서나 DataFrame 행 번호로 원본과 XAI를 연결하지 말 것.
- 파일명 부분일치 또는 subject 단위 확률 유사성만으로 slice를 자동 매핑하지 말 것.
- 클래스 순서를 하드코딩하지 말고 모델의 검증된 class mapping을 사용할 것.
- XAI target은 실제로 gradient를 계산한 logit과 일치해야 한다.
- XAI 계산 중 `torch.no_grad()`를 사용하지 말 것.
- brain mask 계산 실패를 정상 통과로 처리하지 말 것.
- 사용자 작업과 관련 없는 파일을 수정하지 말 것.

## 4. 수정 대상 조사

작업 시작 전에 다음 파일을 읽고 실제 호출 관계를 기록한다.

```text
clip_xai_app/app.py
clip_xai_app/src/inference.py
clip_xai_app/src/xai.py
clip_xai_app/src/visualization.py
clip_xai_app/src/report.py
clip_xai_app/src/schemas.py
clip_xai_app/src/config.py
clip_xai_app/src/warnings.py
clip_xai_app/tests/
clip_lr_grad_eclip_handoff_v1_20260812_105027/code/inference_clip_lr.py
clip_lr_grad_eclip_handoff_v1_20260812_105027/code/grad_eclip.py
clip_lr_grad_eclip_handoff_v1_20260812_105027/data/xai_samples/xai_sample_manifest.csv
clip_lr_grad_eclip_handoff_v1_20260812_105027/reports/grad_eclip_validation.md
```

다음을 먼저 보고한다.

- 실제 heatmap 반환 타입과 범위
- heatmap resize·정규화 위치
- overlay alpha와 colormap
- 대표 slice 선정 기준
- 현재 파일명 생성 위치
- PDF에 전달되는 slice/subject 확률 경로
- 실제 라벨을 현재 앱에서 확보할 수 있는지

## 5. 테스트 우선 수정 방식

현재 `clip_xai_app/tests/`의 자리표시자 테스트를 실제 테스트로 교체한다. 구현 코드를 수정하기 전에 실패하는 테스트를 먼저 작성하고 실행 결과를 기록한다.

최소한 다음 실패를 먼저 재현한다.

1. subject 이름만으로 저장되는 XAI 파일명
2. `cor084`와 `cor088` 결과가 같은 이름으로 충돌하는 문제
3. XAI target 클래스가 결과 metadata에 없는 문제
4. raw heatmap이 저장되지 않는 문제
5. 실제 라벨 AD와 예측 CN이 보고서에서 명확히 구분되지 않는 문제
6. slice 확률과 subject 평균 확률이 혼동될 수 있는 문제
7. 배경 활성 검증값이 제공되지 않는 문제

## 6. 전역 샘플 식별 계약

모든 입력과 산출물에 다음 필드를 포함하는 `SampleIdentity`를 먼저 정의한다.

```python
@dataclass(frozen=True)
class SampleIdentity:
    dataset_id: str
    subject_id: str
    session_or_visit_id: str | None
    scan_id: str
    series_id: str | None
    source_relative_path: str
    source_sha256: str
    input_level: str  # slice | volume
    slice_axis: str | None
    slice_index: int | None
```

다음 조건을 검증한다.

- `source_relative_path`는 프로젝트 또는 데이터 루트 기준으로 정규화한다.
- Windows 대소문자 및 경로 구분자 차이를 정규화한 뒤 충돌을 검사한다.
- `source_sha256`으로 같은 이름의 다른 파일과 이름이 다른 동일 파일을 구분한다.
- slice 입력은 `slice_axis`와 `slice_index`가 필수다.
- volume 입력은 slice index를 갖지 않으며 volume-level XAI로 분리한다.
- manifest의 subject/scan/slice와 파일명에서 파싱한 값이 다르면 자동 수정하지 않고 오류로 처리한다.
- 중복 key가 발견되면 각 원본 경로와 SHA-256을 보고하고 해당 항목의 XAI 생성을 차단한다.

전역 분석 식별자는 다음 정보를 canonical JSON으로 직렬화한 후 SHA-256 digest를 계산하여 생성한다.

```text
SampleIdentity
+ model_id
+ checkpoint_sha256
+ preprocessing_version
+ class_mapping_version
+ xai_method
+ target_class
+ target_score_type
+ xai_config_hash
+ run_id
```

사용자에게 보이는 파일명에는 읽을 수 있는 정보를 넣고, metadata에는 전체 `analysis_id` digest를 저장한다. 데이터베이스나 manifest의 기본키는 긴 파일명이 아니라 `analysis_id`를 사용한다.

## 7. XAI 데이터 계약 추가

`clip_xai_app/src/schemas.py` 또는 별도 `contracts.py`에 다음과 동등한 구조를 추가한다.

```python
@dataclass(frozen=True)
class XAIProvenance:
    analysis_id: str
    dataset_id: str
    subject_id: str
    session_or_visit_id: str | None
    scan_id: str
    series_id: str | None
    slice_index: int
    slice_filename: str
    source_relative_path: str
    source_sha256: str
    true_label: str | None
    predicted_class: str
    prediction_correct: bool | None
    target_class: str
    target_class_index: int
    target_score_type: str
    method_name: str
    model_id: str
    model_version: str | None
    checkpoint_sha256: str
    preprocessing_version: str
    xai_config_hash: str
    run_id: str
    class_names: tuple[str, ...]
    slice_probabilities: dict[str, float]
    subject_probabilities: dict[str, float] | None

@dataclass(frozen=True)
class HeatmapValidation:
    mask_method: str
    mask_threshold: float | None
    brain_fraction: float
    background_fraction: float
    foreground_mean: float
    background_mean: float
    foreground_background_ratio: float
    background_activation_fraction: float
    top_percent: float
    top_activation_background_fraction: float
    max_activation_xy: tuple[int, int]
    max_activation_inside_brain: bool
    finite: bool
    nonconstant: bool
    warnings: tuple[str, ...]

@dataclass(frozen=True)
class XAIArtifact:
    provenance: XAIProvenance
    validation: HeatmapValidation
    original_path: Path
    raw_heatmap_npy_path: Path
    normalized_heatmap_png_path: Path
    overlay_path: Path
    metadata_json_path: Path
```

기존 구조와 충돌하면 이름을 조정하되 필드 의미를 유지한다.

## 8. 고유 식별자와 파일명 규칙

모든 XAI 산출물의 공통 stem은 다음 정보를 포함해야 한다.

```text
{dataset_id}_{scan_id}_{axis}{slice_index:03d}_true-{true_label-or-NA}_pred-{predicted_class}_target-{target_class}_{correctness}_{method}_{model_short}_{analysis_id_short}
```

예시:

```text
OASIS3_OAS30100_MR_d0158_cor088_true-AD_pred-CN_target-CN_incorrect_clip-gradient_clip-lr_a1b2c3d4e5f6
```

동일 stem으로 다음 파일을 저장한다.

```text
*_original.png
*_heatmap.npy
*_heatmap.png
*_overlay.png
*_metadata.json
```

규칙:

- 기존 파일이 있으면 묵시적으로 덮어쓰지 않는다.
- `analysis_id_short`는 전체 digest의 표시용 접두부이고 metadata에는 전체 digest를 기록한다.
- `run_id`는 실행 시각만 사용하지 말고 모델·체크포인트·설정과 결합한다.
- 파일명과 JSON 내부 provenance가 일치하는지 검증한다.
- 실제 라벨이 없으면 `true-NA`, 정오 여부는 `unknown`으로 기록한다.
- subject 평균 결과는 별도 `subject_summary` 파일로 저장하고 slice XAI 파일명과 혼합하지 않는다.
- 동일 stem이 이미 존재하면 metadata의 전체 `analysis_id`와 모든 provenance를 비교한다.
- provenance가 완전히 같으면 재사용 가능하지만, 하나라도 다르면 충돌로 처리하고 새로운 고유 ID를 생성한다.

## 9. 전체 Artifact Index 구현

모든 XAI 생성 결과를 다음 파일에 한 행씩 기록한다.

```text
clip_xai_app/artifacts/xai_artifact_index.csv
```

필수 열:

```text
analysis_id,dataset_id,subject_id,session_or_visit_id,scan_id,series_id,
input_level,slice_axis,slice_index,source_relative_path,source_sha256,
true_label,predicted_class,prediction_status,target_class,target_score_type,
model_id,checkpoint_sha256,preprocessing_version,xai_method,xai_config_hash,run_id,
original_path,raw_heatmap_path,heatmap_png_path,overlay_path,metadata_path,
xai_qc_status,created_at
```

Index 기록 전 다음 제약을 검사한다.

- `analysis_id`는 unique여야 한다.
- 동일 산출물 경로가 서로 다른 `analysis_id`에 연결되면 실패한다.
- 동일 `analysis_id`가 서로 다른 source SHA-256 또는 checkpoint SHA-256을 가지면 실패한다.
- index 행의 각 파일이 실제 존재하고 metadata JSON과 일치해야 한다.
- 병렬 실행 시 임시 파일에 쓴 뒤 atomic rename 또는 file lock으로 index 손상을 방지한다.
- 부분 실패한 실행은 완성된 artifact로 index에 등록하지 않는다.

## 10. 원본·heatmap·overlay 개별 저장

XAI 생성 함수가 최소한 다음을 반환하도록 수정한다.

```text
raw_heatmap
normalized_heatmap
target_logit
target_probability
target_class
method_name
```

저장 규칙:

- `raw_heatmap`: 정규화 전 `float32` NPY
- `normalized_heatmap`: `[0, 1]` 범위 `float32` NPY 또는 검증 가능한 배열
- 표시용 heatmap: PNG
- overlay: 원본과 동일한 크기의 PNG
- original: 입력에 사용한 정확한 slice의 복사본 또는 추적 가능한 참조
- interpolation 방식과 `align_corners` 설정을 metadata에 기록
- overlay alpha와 colormap을 metadata에 기록

PNG만 저장하고 원시 배열을 버리지 말 것.

## 11. Brain mask 생성

이번 단계에서는 진단용 brain segmentation을 구현하지 않는다. XAI 배경 검출을 위한 보수적인 QC mask를 만든다.

권장 순서:

1. 원본 grayscale을 `[0,1]`로 변환
2. 비정상 NaN/Inf 확인
3. 낮은 intensity threshold로 초기 foreground 생성
4. 가장 큰 연결 성분 유지
5. 작은 구멍 채움과 최소한의 morphology 적용
6. mask가 지나치게 작거나 큰지 검사

mask 방법과 threshold는 설정 파일에서 관리한다.

```yaml
heatmap_validation:
  mask_method: largest_component_intensity
  intensity_threshold: 0.08
  top_activation_percent: 5.0
  minimum_brain_fraction: 0.10
  maximum_brain_fraction: 0.70
```

위 threshold는 현재 예시에서 출발한 연구용 QC 기본값이며 임상 기준이 아니다. 데이터 전체 분포를 확인한 뒤 조정해야 한다.

mask가 실패하거나 brain fraction이 범위를 벗어나면 정량 지표를 신뢰하지 말고 경고를 반환한다.

## 12. Heatmap 정량 검증

원시 또는 정규화된 heatmap `H`와 brain mask `M`으로 다음을 계산한다.

```text
foreground_mean = mean(H[M])
background_mean = mean(H[~M])
foreground_background_ratio = foreground_mean / (background_mean + epsilon)
background_activation_fraction = sum(H[~M]) / (sum(H) + epsilon)
top_activation_background_fraction = count(top-p% pixels outside M) / count(top-p% pixels)
max_activation_inside_brain = M[argmax(H)]
```

검증 항목:

- shape가 원본과 동일한지
- finite인지
- 최소값·최대값·평균·표준편차
- 상수 heatmap이 아닌지
- 최대 활성 위치가 brain mask 내부인지
- 전경 평균이 배경 평균보다 큰지
- 상위 5% 활성의 배경 과대표 여부

단일 임계값으로 임상적 합격을 선언하지 않는다. 결과는 다음 QC 상태로 분류한다.

```text
PASS_BASIC_SPATIAL_QC
WARN_BACKGROUND_ACTIVATION
FAIL_MAX_OUTSIDE_BRAIN
FAIL_CONSTANT_HEATMAP
FAIL_NONFINITE
FAIL_MASK
```

임계값은 설정 파일에 두고 validation 데이터 분포와 함께 문서화한다.

## 13. 예측 정확성과 XAI 품질 분리

다음 두 판정을 절대 합치지 않는다.

```text
prediction_status: correct | incorrect | unknown
xai_qc_status: PASS_* | WARN_* | FAIL_*
```

예시 `cor088`은 다음처럼 표시해야 한다.

```text
실제 라벨: AD
모델 예측: CN
모델 출력 확률: CN 0.5538 / MCI 0.3522 / AD 0.0941
예측 상태: 오분류
XAI target: CN class logit
설명 의미: 모델이 잘못된 CN 예측을 만들 때 점수에 기여한 영역
```

XAI spatial QC를 통과하더라도 예측이 맞았다는 의미가 아니다.

## 14. UI 수정

`clip_xai_app/app.py`를 다음과 같이 수정한다.

- 결과 상단에 `subject_id`, `slice_filename`, `slice_index` 표시
- dataset, visit/session, scan ID, model/checkpoint, run ID 표시
- 실제 라벨이 존재하면 실제 라벨 표시
- 예측 클래스와 정답/오답 상태 표시
- `Confidence`를 `모델 출력 확률`로 변경
- XAI target 클래스와 target score 종류 표시
- 갤러리 항목에 다음 형식의 caption 사용

```text
cor088 | true=AD | pred=CN | target=CN | incorrect | overlay
```

- 원본, heatmap, overlay를 서로 다른 갤러리 항목으로 표시
- QC 결과 표 추가
- `WARN` 또는 `FAIL`이면 눈에 띄는 연구용 경고 표시
- 사용자가 target 클래스를 변경할 수 있다면 예측 클래스와 다를 수 있음을 명시
- target을 변경하면 heatmap을 다시 계산하고 metadata도 새로 저장
- raw heatmap NPY와 metadata JSON 다운로드 경로 제공 여부를 검토

UI 오류 메시지에는 traceback을 노출하지 않고 error ID만 표시한다.

## 15. PDF 보고서 수정

`clip_xai_app/src/report.py`에 다음 내용을 추가한다.

### 입력 식별 정보

- subject ID
- dataset/source
- session/visit 및 scan ID
- 정확한 slice 파일명과 slice index
- 단일 slice 또는 subject 다중 slice 분석 여부

### 모델 출력

- 실제 라벨이 있으면 별도 표시
- 예측 클래스
- 클래스별 모델 출력 확률
- 정답/오답/미확인 상태
- slice 확률과 subject 확률을 다른 표로 분리

### XAI provenance

- target 클래스
- target score: LR class logit
- method name
- 모델 ID/버전
- checkpoint SHA-256 또는 짧은 digest
- preprocessing version
- analysis ID
- 입력 파일명
- 생성 시각과 run ID

### XAI QC

- brain fraction
- foreground/background mean
- foreground/background ratio
- background activation fraction
- top 5% background fraction
- 최대 활성 brain 내부 여부
- QC 상태와 경고

### 문구 규칙

오분류이면 다음 취지로 표시한다.

> 이 히트맵은 실제 AD 라벨에 대한 근거가 아니라, 모델이 CN으로 오분류할 때 CN class logit에 기여한 입력 영역을 표시합니다.

금지 표현:

- 진단 결과
- 병변 위치
- 알츠하이머가 발견된 영역
- 임상적 신뢰도
- 질병 확률

## 16. 기존 검증 산출물 전체 마이그레이션 검사

기존 `reports/grad_eclip_samples/*_gradeclip.png`를 삭제하지 않는다.

대신 다음을 수행한다.

1. manifest와 이미지에 표시된 확률을 대조한다.
2. 실제 사용된 slice 번호를 역추적한다.
3. 확정 가능한 경우 새 metadata JSON을 생성한다.
4. 새 규칙의 파일명으로 복사본을 생성한다.
5. 확정할 수 없으면 `slice-UNKNOWN`으로 표시하고 검증 대상에서 제외한다.
6. 기존 `grad_eclip_validation.md`에 subject 단위 이름만 사용했던 한계를 명시한다.

`OAS30100_MR_d0158_incorrect_gradeclip.png`는 확률상 `cor084`와 일치하는지 코드로 검증하고, 허용오차 내 일치할 때만 `cor084`로 매핑한다. 이 방식은 해당 예시만 처리하기 위한 예외 코드로 구현하지 말고, 모든 기존 검증 이미지에 대해 후보 slice 확률을 비교하는 일반 migration 도구로 구현한다.

마이그레이션 도구는 다음 규칙을 따른다.

- exact filename/slice metadata가 있으면 그것을 우선한다.
- metadata가 없으면 subject 내 모든 후보 slice와 클래스별 확률 벡터 전체를 비교한다.
- 한 후보만 엄격한 허용오차 내에서 일치할 때만 자동 매핑한다.
- 후보가 0개 또는 2개 이상이면 `AMBIGUOUS`로 분류한다.
- Top-1 클래스 하나만 같다는 이유로 매핑하지 않는다.
- 이미지 제목 OCR이나 표시 문자열만으로 확정하지 않는다.
- 매핑 결과, 근거, 오차, 후보 수를 migration report에 기록한다.
- 모든 기존 XAI 이미지의 처리 결과가 `MAPPED`, `AMBIGUOUS`, `UNRESOLVED`, `CORRUPT` 중 하나여야 한다.

## 17. 필수 테스트

### 식별 테스트

- `cor084`와 `cor088`가 서로 다른 artifact stem을 생성한다.
- 모든 manifest 행을 대상으로 canonical `SampleIdentity`를 생성했을 때 예상하지 않은 중복이 0개다.
- 서로 다른 dataset에서 같은 subject/scan/slice 문자열을 사용해도 다른 `analysis_id`가 생성된다.
- 동일 slice라도 checkpoint가 다르면 다른 `analysis_id`가 생성된다.
- 동일 slice라도 target 클래스가 다르면 다른 `analysis_id`가 생성된다.
- 동일 slice라도 preprocessing 또는 XAI 설정이 다르면 다른 `analysis_id`가 생성된다.
- 동일 내용의 파일이 다른 경로에 있으면 SHA-256 기반 중복 후보로 보고된다.
- 같은 이름이지만 내용이 다른 파일은 절대 같은 샘플로 합쳐지지 않는다.
- DataFrame 정렬 순서를 바꿔도 원본–확률–XAI 대응이 유지된다.
- 일부 행을 제거해도 나머지 행의 대응이 밀리지 않는다.
- slice 파일명에서 index를 정확히 추출한다.
- 파일명과 metadata의 subject/slice/target이 일치한다.
- 서로 다른 MR_ID 혼합 입력을 거부한다.

### 확률 테스트

- `cor088` 확률이 기준값과 허용오차 내 일치한다.
- 확률 열이 `[CN, MCI, AD]` 매핑과 일치한다.
- slice 확률이 subject 평균 확률로 대체되지 않는다.

### XAI 테스트

- XAI target이 CN이면 CN logit에서 gradient를 계산한다.
- heatmap shape가 `224×224`이다.
- raw heatmap이 finite이고 nonconstant이다.
- normalization 후 범위가 `[0,1]`이다.
- original, heatmap, overlay가 같은 크기다.
- 원시 NPY를 저장하고 다시 읽었을 때 값이 유지된다.

### QC 테스트

- 합성 heatmap의 최대값이 배경에 있으면 `FAIL_MAX_OUTSIDE_BRAIN`
- 상수 heatmap이면 `FAIL_CONSTANT_HEATMAP`
- NaN/Inf가 있으면 `FAIL_NONFINITE`
- 정상 foreground 집중 heatmap은 기본 spatial QC 통과
- mask 실패를 정상 통과시키지 않는다.

### 보고서 테스트

- `cor088`, 실제 AD, 예측 CN, target CN, 오분류가 PDF에 존재한다.
- 금지 표현이 PDF에 존재하지 않는다.
- exact slice filename이 포함된다.
- QC 수치가 metadata JSON과 일치한다.

### 전수 무결성 테스트

- 모든 artifact index 행의 `analysis_id`가 unique다.
- 모든 index 경로가 존재한다.
- 모든 original SHA-256이 metadata와 일치한다.
- 모든 checkpoint SHA-256이 실제 사용 파일과 일치한다.
- 모든 확률 벡터가 올바른 sample key에 연결된다.
- 원본·raw heatmap·PNG·overlay·JSON이 정확히 1:1:1:1:1로 대응한다.
- orphan artifact와 unindexed artifact가 0개다.
- 중복 산출물 경로가 0개다.
- 전체 migration 결과에서 `AMBIGUOUS`와 `UNRESOLVED` 항목을 별도 보고한다.

## 18. 실행 및 검증 순서

1. `git status --short`와 현재 브랜치 기록
2. 기존 테스트 실행 및 baseline 기록
3. 실패하는 회귀 테스트 작성
4. 테스트 실패가 의도한 문제 때문인지 확인
5. schema와 artifact naming 구현
6. raw heatmap 저장 구현
7. brain mask와 QC 지표 구현
8. UI 수정
9. PDF 수정
10. 기존 검증 산출물 migration 검사
11. 전체 테스트 실행
12. `cor084`와 `cor088`를 동시에 분석해 최소 회귀 사례의 교차 혼동이 없는지 확인
13. 전체 manifest를 대상으로 전역 식별키와 충돌 검사 실행
14. dataset·subject·scan·slice·model·target별 표본을 층화 추출하여 대응 관계 검사
15. 전체 artifact index 무결성 검사
16. 생성된 PNG·NPY·JSON·PDF를 실제로 열어 검증
17. `git diff --check`와 `git status --short` 확인

## 19. 완료 기준

다음을 모두 만족해야 완료로 판정한다.

- [ ] 모든 XAI 결과가 exact slice filename을 포함한다.
- [ ] target 클래스와 target score가 metadata에 기록된다.
- [ ] 실제 라벨과 예측 클래스가 분리된다.
- [ ] 오분류 여부가 UI와 PDF에 표시된다.
- [ ] raw heatmap NPY가 저장된다.
- [ ] 표시용 heatmap과 overlay가 개별 저장된다.
- [ ] brain-mask 기반 QC 수치가 계산된다.
- [ ] QC 상태가 UI와 PDF에 표시된다.
- [ ] `cor084`와 `cor088` 결과가 충돌하지 않는다.
- [ ] 전체 manifest에서 예상하지 않은 sample identity 충돌이 0개다.
- [ ] 모든 artifact의 `analysis_id`가 고유하다.
- [ ] 모든 원본·확률·XAI·metadata·PDF가 ID 기반으로 연결된다.
- [ ] 서로 다른 dataset·scan·model·target·run 결과가 섞이지 않는다.
- [ ] artifact index에 orphan·중복·누락 경로가 없다.
- [ ] 기존 CLIP 예측 확률이 변경되지 않는다.
- [ ] 모든 실제 테스트가 통과한다.
- [ ] 자리표시자 테스트가 남아 있지 않다.
- [ ] README와 검증 문서가 실제 구현 상태와 일치한다.

## 20. 완료 보고 형식

### 수정 결과

- 수정한 파일
- 생성한 파일
- 각 변경 목적

### 회귀 사례

| 항목 | cor084 | cor088 |
|---|---|---|
| 실제 라벨 | | |
| 예측 클래스 | | |
| XAI target | | |
| 클래스별 확률 | | |
| artifact stem | | |
| QC 상태 | | |

### XAI QC

- mask 방법
- 설정값
- 원시 heatmap 기준 전경·배경 지표
- 최대 활성 위치
- 경고 및 실패 사례

### 전체 식별 무결성

- 전체 입력 sample 수
- 고유 SampleIdentity 수
- 생성된 analysis ID 수
- 중복 ID 수
- 경로 충돌 수
- orphan/unindexed artifact 수
- migration `MAPPED/AMBIGUOUS/UNRESOLVED/CORRUPT` 수
- dataset·subject·scan·slice·model·target별 충돌 검사 결과

### 테스트

- 전체 테스트 수
- 통과/실패/스킵
- 실행하지 못한 테스트와 이유

### 남은 제한사항

- 해부학적 검증 여부
- 모델 무작위화 검사 여부
- deletion/insertion 검사 여부
- 인접 slice 일관성 검사 여부
- 전문가 검토 여부

## 21. 체크리스트

- [ ] 실제 코드 호출 관계 조사
- [ ] `cor084`/`cor088` 혼동 회귀 테스트 작성
- [ ] 전체 dataset용 `SampleIdentity` 구현
- [ ] source/checkpoint/config SHA-256 기반 `analysis_id` 구현
- [ ] 전체 artifact index 구현
- [ ] 전수 중복·누락·orphan 검사 구현
- [ ] XAI provenance schema 구현
- [ ] 고유 artifact stem 구현
- [ ] raw/normalized heatmap 분리
- [ ] original/heatmap/overlay/JSON 저장
- [ ] brain mask 구현
- [ ] 전경·배경 QC 지표 구현
- [ ] prediction status와 XAI QC status 분리
- [ ] UI 식별·경고 정보 수정
- [ ] PDF 식별·경고·QC 정보 수정
- [ ] 기존 검증 이미지 slice 역추적
- [ ] 모든 기존 검증 이미지 migration 수행
- [ ] 금지 표현 회귀 테스트
- [ ] 전체 테스트와 시각 검증
- [ ] README 및 검증 보고서 수정

## 22. TODO 우선순위

1. **P0:** slice 식별자와 XAI target 추적 문제 수정
2. **P0:** 전역 `SampleIdentity`와 `analysis_id` 구현
3. **P0:** 모든 dataset·subject·scan·slice·model·target 결과 충돌 방지
4. **P0:** artifact index와 전수 무결성 검사 구현
5. **P0:** raw heatmap 저장과 brain-mask QC 구현
6. **P0:** 오분류 설명 문구와 실제 라벨 표시
7. **P1:** UI·PDF QC 표 통합
8. **P1:** 기존 검증 산출물 전체 slice mapping 복구
9. **P1:** 전체 validation set의 QC 분포 계산
10. **P2:** 모델 무작위화 및 label-randomization sanity check
11. **P2:** deletion/insertion faithfulness 평가
12. **P2:** 인접 slice와 subject 수준 XAI 일관성 평가

작업 완료를 주장하기 전에 `OAS30100_MR_d0158_cor084.png`와 `OAS30100_MR_d0158_cor088.png`로 최소 회귀 사례를 통과하고, 이어서 전체 manifest와 전체 artifact index를 검사하여 모든 dataset·subject·scan·slice·model·target·run의 확률·원본·heatmap·metadata·PDF가 서로 교차되지 않는 것을 증명하라. 두 예시만 하드코딩해 통과시키는 수정은 완료로 인정하지 않는다.
