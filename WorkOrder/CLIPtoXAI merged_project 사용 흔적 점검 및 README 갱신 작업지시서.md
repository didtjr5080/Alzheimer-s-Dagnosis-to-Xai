# CLIPtoXAI merged\_project 사용 흔적 점검 및 README 갱신 작업지시서

Sep 23, 2026 · @Someone

## 1. 목적과 범위

`CLIPtoXAI` 저장소에서 `merged_project`(OASIS-3+ADNI 통합) 산출물이 이미 어디에 어떻게 쓰이고 있는지 점검하고, 그 결과를 `README.md`에 반영한다. 이번 작업은 **점검과 문서 갱신만** 하며, 코드 동작은 바꾸지 않는다.

현재 루트 구성(VS Code 탐색기 기준):

| 폴더/파일 | 추정 역할 | 비고 |
| --- | --- | --- |
| `clip_lr_grad_eclip_handoff_v1_20260812…` | OASIS-3 단독 CLIP+LR handoff 패키지 | 통합 모델 아님 |
| `merged_project` | 통합 프로젝트 산출물 | 점검 대상 |
| `merged_project-20260825T072246…` (3개) | Drive 다운로드 분할 압축본으로 보임 | 중복 여부 확인 필요 |
| `clip_xai_app` | XAI 앱 코드 | 모델 경로 참조 점검 |
| `graph_xai_extension` | XAI 확장 코드 | 모델 경로 참조 점검 |
| `scripts` | 보조 스크립트 | 모델 경로 참조 점검 |
| `artifacts` | 실행 산출물 | 어느 모델 결과인지 확인 |
| `WorkOrder` | 작업지시서 보관 | 기존 지시서와 충돌 여부 확인 |
| `README.md` | 저장소 설명 | 갱신 대상 |

"추정 역할"은 폴더 이름만 보고 적은 것이다. 실제 내용을 확인해 틀리면 보고서에 고쳐 적는다.

## 2. 점검 항목

아래 순서대로 진행하고, 단계마다 명령과 출력 요약을 기록한다.

### A. 전체 참조 검색

```bash
grep -rn --include=*.py --include=*.ipynb --include=*.md --include=*.json --include=*.yaml --include=*.txt \
  -e "merged_project" -e "OASIS3_project" -e "model_handoff" -e "clip_lr_grad_eclip_handoff" \
  -e "gbm_baseline" -e "gbm_stage" -e "clip_best.pt" -e "3dcnn" -e "clip_lr_classifier" \
  --exclude-dir=.pytest_cache --exclude-dir=.git .
```

- 결과를 파일·줄 번호·참조 경로 표로 정리한다.
- 하드코딩 경로(`/content/drive/MyDrive/...`)와 상대경로를 구분해 표시한다.

### B. 폴더별 확인

1. **`merged_project`**: 하위 구조(`models/`, `checkpoints/`, CSV 등)와 각 파일의 수정 시각·크기를 나열한다.
2. **`merged_project-20260825T072246…` 3개**: 압축 파일인지 풀린 폴더인지, `merged_project`와 내용이 겹치는지 파일 목록·SHA-256으로 비교한다.
3. **`clip_xai_app`, `graph_xai_extension`, `scripts`**: 모델을 로드하는 진입점(함수·설정 파일)을 찾고, 어떤 모델 파일을 어떤 방식(`lgb.Booster`, `torch.load`, `CLIPModel.from_pretrained`, `open_clip`, `joblib`)으로 여는지 적는다.
4. **`artifacts`**: 각 산출물이 어느 모델·데이터로 만들어졌는지 파일명·내부 메타데이터로 확인한다.
5. **`WorkOrder`**: 기존 지시서 목록과 각 지시서가 가리키는 모델 경로를 적는다.
6. **`README.md`**: 현재 내용 중 실제 코드와 어긋나는 문장을 표시한다.

### C. 실행 확인 (읽기 전용)

- `.pytest_cache`가 있으므로 테스트가 있다면 `pytest -q`를 실행해 통과/실패 수를 기록한다.
- 실패해도 이번 작업에서는 고치지 않고 원인만 적는다.

## 3. 판정 기준

모델 참조마다 아래 표로 "통합 / OASIS-3 단독 / 불명" 중 하나로 판정하고, 근거가 된 파일·줄을 적는다.

| 구분 | 통합 (OASIS-3+ADNI) | OASIS-3 단독 |
| --- | --- | --- |
| 경로 | `merged_project/models/`, `merged_project/checkpoints/` | `OASIS3_project/model_handoff/…`, `clip_lr_grad_eclip_handoff_v1_20260812…` |
| 모델 파일 | `gbm_baseline.txt`, `gbm_stage1.txt`, `gbm_stage2.txt`, `clip_best.pt`, `3dcnn_best.pt` | `clip_lr_classifier_new_run.joblib`, `clip_lr_weights.npz`, `clip_model/model.safetensors` |
| CLIP 백본 | BiomedCLIP (`open_clip`) | `openai/clip-vit-base-patch16` (`transformers.CLIPModel`) |
| 데이터 키 | `scan_id`, `dataset_source`, `is_adni` | `MR_ID` |
| 매니페스트 | `unified_manifest.csv`, `unified_slice_manifest.csv`, `unified_final_features_gbm.csv` | `slice_manifest_large*.csv` |
| 참고 지표 (test) | GBM baseline Acc 63.2% / Macro-F1 0.604, n=573 | subject Acc 72.2% / Macro-F1 0.435 |

- 한 코드에 양쪽 흔적이 섞여 있으면 "혼재"로 표시하고, 어느 쪽이 실제로 실행되는지(조건 분기·기본값)를 적는다.
- `dataset_source`에 ADNI 행이 실제로 있는지 CSV에서 `value_counts()`로 확인한다. 파일명만으로 통합이라고 판정하지 않는다.

## 4. README 갱신 규칙

점검에서 확인한 사실만 적는다. 확인하지 못한 내용은 `TODO(확인 필요): …`로 남긴다.

넣을 섹션(순서 고정):

1. **프로젝트 개요**: CLIP 기반 XAI 알츠하이머 진단 지원(CN/MCI/AD), OASIS-3+ADNI 통합 데이터
2. **디렉터리 구조**: 루트 폴더별 한 줄 설명. 분할 압축본(`merged_project-20260825T072246…`)은 원본과 관계를 명시
3. **모델 현황**: 통합 모델과 OASIS-3 단독 모델을 별도 표로 구분 (파일 경로, 백본, 학습 데이터, test 지표, 현재 사용 여부)
4. **코드별 사용 모델**: `clip_xai_app`, `graph_xai_extension`, `scripts`가 각각 어떤 모델을 로드하는지
5. **실행 방법**: 실제 동작을 확인한 명령만 기재 (필요 경로·환경 변수 포함)
6. **한계와 주의**: 연구용이며 임상 검증 없음, 히트맵은 진단 근거가 아님, 슬라이스 범위(cor069–092)에 해마 복셀이 거의 없음
7. **변경 이력**: 이번 갱신 날짜와 요약 한 줄

작성 원칙:

- 기존 README 문장 중 사실과 맞는 것은 유지하고, 틀린 것만 고친다. 삭제한 문장은 보고서에 원문을 남긴다.
- 지표는 출처 파일(노트북 셀 출력, 파일의 README 등)을 괄호로 명시한다.
- 절대경로(`/content/drive/...`)는 Colab 기준이라고 표기하고, 저장소 기준 상대경로를 함께 적는다.

## 5. 금지 사항과 완료 보고

이번 작업에서 수정하는 파일은 `README.md`와 새 보고서 하나뿐이다.

- 코드, 모델 파일, CSV, 기존 `WorkOrder` 문서를 수정·이동·삭제하지 않는다.
- 중복으로 보이는 `merged_project-20260825T072246…` 폴더도 지우지 않고, 정리 제안만 보고서에 적는다.
- `.gitignore`는 바꾸지 않는다. 대용량 파일이 추적 중이면 보고만 한다.
- README 갱신 전 원본을 `README_backup_20260923.md`로 복사해 둔다.
- 확인되지 않은 성능 수치나 경로를 README에 쓰지 않는다.

**완료 보고** — `WorkOrder/merged_project_점검보고서_20260923.md`로 저장:

1. 참조 검색 결과표 (파일 · 줄 · 경로 · 판정: 통합/단독/혼재/불명)
2. 분할 압축본 3개와 `merged_project` 비교 결과
3. `pytest` 결과 (있는 경우)
4. README 변경 요약 (추가·수정·삭제 문장)
5. 남은 `TODO`와 후속 작업 제안 (예: 단독 모델을 참조하는 코드의 교체)
