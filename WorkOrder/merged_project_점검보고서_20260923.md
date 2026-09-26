# merged_project 사용 흔적 점검 보고서 (2026-09-23)

작업지시서: `WorkOrder/CLIPtoXAI merged_project 사용 흔적 점검 및 README 갱신 작업지시서.md`

이번 작업에서 수정한 파일은 `README.md`와 이 보고서 파일 두 개뿐입니다. 코드, 모델 파일, CSV, 기존 `WorkOrder` 문서는 수정·이동·삭제하지 않았습니다. `README.md` 원본은 `README_backup_20260923.md`로 백업했습니다.

## 1. 참조 검색 결과표

지시서 2.A절의 grep 명령을 그대로 실행했습니다.

```bash
grep -rn --include=*.py --include=*.ipynb --include=*.md --include=*.json --include=*.yaml --include=*.txt \
  -e "merged_project" -e "OASIS3_project" -e "model_handoff" -e "clip_lr_grad_eclip_handoff" \
  -e "gbm_baseline" -e "gbm_stage" -e "clip_best.pt" -e "3dcnn" -e "clip_lr_classifier" \
  --exclude-dir=.pytest_cache --exclude-dir=.git .
```

총 952줄이 매칭되었습니다. `merged_project/`, `merged_project-2026...` 폴더 자체(데이터/체크포인트/이미지 덤프)에는 이 패턴들이 전혀 등장하지 않았습니다(코드·README·requirements 파일이 그 안에 없기 때문). 최상위 디렉터리별 분포:

| 디렉터리 | 매칭 줄 수 | 비고 |
| --- | --- | --- |
| `graph_xai_extension/` | 701 | 대부분 `clip_lr_grad_eclip_handoff`(660줄, 문서/테스트의 정상적인 참조). `merged_project`는 40줄 — 전부 파일-무결성 테스트가 "이 폴더는 스캔에서 제외한다"고 적은 것뿐, 모델 참조 아님 |
| `artifacts/` | 78 | `merged_project` 54줄 — 전부 `artifacts/merge_validation/*` 로그/리포트(병합·검증 스크립트 산출물) |
| `WorkOrder/` | 73 | 기존 지시서 6개가 각자 다른 모델 경로를 지시 |
| `clip_xai_app/` | 68 | 아래 표 참고 |
| `scripts/` | 19 | `merged_project` 26줄(중복 집계 있음) — 병합/검증 스크립트 자체 |
| `clip_lr_grad_eclip_handoff_v1_20260812_105027/` | 7 | 자기 자신 이름 언급 |
| `README.md` | 6 | 전부 `clip_lr_grad_eclip_handoff` — 갱신 전 README는 `merged_project`를 한 번도 언급하지 않았음(사실과 다른 누락) |

### 판정에 필요한 실제 코드 참조 (파일 · 줄 · 경로 · 판정)

프로즈(설명 문장) 매칭을 제외하고, 실제로 모델 경로를 로드/참조하는 코드 줄만 정리합니다.

| 파일 · 줄 | 참조 경로 | 판정 |
| --- | --- | --- |
| `clip_xai_app/src/model_loader.py:19-24` | `clip_lr_grad_eclip_handoff_v1_20260812_105027` (`config.handoff_root`) | 단독 — **실제 실행 경로** |
| `clip_xai_app/src/adapters/legacy_clip_lr.py:13-16` | 위와 동일(`model_loader.load_pipeline`) | 단독 — **실제 실행 경로** |
| `clip_xai_app/app.py:12-19` | `src.inference.analyze_subject` import (단독 경로로 연결) | 단독 — **실제 실행 경로** |
| `clip_xai_app/app.py:14, 207` | `src.model_registry.registry_rows()` | 혼재 표시용 — UI에 통합 모델 가용성 "표"만 그림, **호출은 하지 않음** |
| `clip_xai_app/src/model_registry.py:14-16` | `clip_lr_grad_eclip_handoff...`, `merged_project` 존재 여부 확인 | 혼재 — 두 경로 모두 파일 존재만 확인, 실행은 안 함 |
| `clip_xai_app/src/merged_artifacts.py:67-70` | `merged_project` | 통합 — 감사(audit)용, 실행 아님 |
| `clip_xai_app/src/adapters/merged_reference.py:91-97` | `merged_project/clip_test_probs.csv` 등 | 통합 — `predict_by_scan_id()`로 CSV 조회만 가능, **UI에서 호출하는 곳 없음(확인함)** |
| `clip_xai_app/configs/model_registry.yaml:14-59` | `merged_project/*.csv`, `checkpoints/*.pt` | 통합 — 상태값이 전부 `AVAILABLE(reference_csv_by_scan_id)` 아니면 `BLOCKED_MODEL_CONTRACT`/`BLOCKED_PREPROCESSING_CONTRACT` |
| `graph_xai_extension/graph_xai/legacy_adapter.py` (전체) | `clip_lr_grad_eclip_handoff...` | 단독 — **실제 실행 경로** |
| `graph_xai_extension/scripts/_regen_baseline_manifest.py:17-20` | `merged_project`, 분할 3폴더 | 무관 — 파일-해시 스캔 제외 목록일 뿐, 모델 참조 아님 |
| `scripts/merge_merged_parts.ps1`, `scripts/validate_merged_project.py` | `merged_project`, 분할 3폴더 | 무관 — 모델을 로드하지 않고 파일 구조/체크포인트 구조만 검사 |

**결론(혼재 상세)**: `clip_xai_app`는 코드베이스에 통합 모델 참조(레지스트리 표시 + scan_id 조회 어댑터)가 존재하지만, 실제로 사용자가 클릭하는 "분석" 버튼이 실행하는 경로(`app.py::analyze` → `analyze_subject` → `load_pipeline`)는 100% OASIS-3 단독 handoff만 호출합니다. 통합 어댑터를 호출하는 UI 입력(예: scan_id 검색창)이 없다는 것을 `app.py` 전체에서 `scan_id` 문자열을 검색해 확인했습니다(매칭 0건). 따라서 "혼재"이지만 **실행 시점 기준으로는 단독 모델만 동작**합니다.

## 2. 분할 압축본 3개와 merged_project 비교 결과

`artifacts/merge_validation/`에 2026-08-26에 이미 이 작업(Codex_merged_project_병합_검증_작업지시서.md)이 실행한 로그가 있어, 그 결과를 확인하고 `scripts/validate_merged_project.py`를 2026-09-23에 다시 읽기 전용으로 재실행해 최신 상태로 재확인했습니다(코드/데이터는 수정하지 않음).

- 각 분할 폴더는 내부에 자체 `merged_project/` 하위 폴더를 갖고 있으며(폴더인지 압축 파일인지: **압축 해제된 폴더**, `.zip`이 아님), 파일 수는 `merged_project-...-001`=33,283 / `-002`=31,826 / `-003`=30,316개로 서로 **겹치지 않습니다**(병합 dry-run 로그: `duplicate identical files to skip: 0`, `artifacts/merge_validation/merge_20260826_154400.txt`).
- 세 폴더의 합계(95,425개)는 `merged_project/`의 실제 파일 수(95,425개)와 정확히 일치합니다.
- 2026-09-23 재검증(`artifacts/merge_validation/validate_20260923_185436.json`): `Files expected/actual/missing/extra/size_mismatch: 95425/95425/0/0/0`. 2026-08-26 검증(`validate_20260826_160654.json`)과 동일한 결과로, 그 사이 어느 쪽도 변경되지 않았습니다.
- 결론: `merged_project/`는 세 분할 폴더의 **무손실 합집합**입니다. 세 분할 폴더는 이미 병합이 끝난 원본이며, 현재로서는 `merged_project/`만 있으면 충분합니다.
- **정리 제안**: 분할 폴더 3개(`merged_project-20260825T072246Z-1-00{1,2,3}`, 총 약 9.63GB)는 삭제해도 데이터 손실이 없습니다. 다만 이번 작업 지시(삭제 금지)에 따라 삭제하지 않았습니다. 삭제하려면 먼저 `merged_project/`를 별도 매체(외장 드라이브 등)에 백업한 뒤 삭제할 것을 권장합니다.

## 3. pytest 결과

두 프로젝트 모두 이 로컬 환경에 전역 자동로드되는 `pytest-qt` 플러그인이 Qt DLL 로드 실패로 즉시 크래시하므로, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`을 설정해 우회했습니다(코드 결함이 아니라 환경 문제이며, 이번 세션 이전부터 알려진 문제입니다). 고치지 않고 결과만 기록합니다.

### `clip_xai_app`

```text
16 passed, 1 warning, 4 errors in 53.89s
```

4건의 ERROR는 전부 `C:\Users\Public\Documents\ESTsoft\CreatorTemp\pytest-of-jocke`에 대한 `PermissionError: [WinError 5]`이며, pytest의 `tmp_path` 픽스처가 이 로컬 Windows 계정의 ESTsoft 소프트웨어가 만든 폴더에 접근하지 못해 발생합니다. 실패한 4개 테스트(`test_pdf_report_uses_research_and_model_output_language`, `test_cor084_and_cor088_create_distinct_artifact_stems_and_analysis_ids`, `test_pdf_report_marks_mci_to_cn_misclassification_and_xai_target`, `test_pdf_generation_aborts_on_manifest_probability_mismatch`)는 모두 `tmp_path`를 쓰는 테스트이고, 실제 테스트 로직은 한 줄도 실행되지 못한 채 setup 단계에서 막혔습니다. 이번 작업에서는 고치지 않았습니다(테스트 실행 후 생성된 `clip_xai_app/.pytest_cache/`는 이번 점검의 부산물이라 삭제해 두었습니다).

### `graph_xai_extension`

```text
3 failed, 148 passed, 1 deselected, 5 warnings in 82.36s
```

(`test_real_model_batch_evaluation_deduplicates_six_known_subjects`는 ~10분 걸리는 느린 테스트라 `-k "not real_model_batch_evaluation"`으로 제외했습니다.)

실패한 3건은 모두 `tests/test_existing_files_unchanged.py`의 원본 파일 무결성 보호 테스트이며, 원인은 프로젝트 결함이 아니라 **이 확장의 SHA-256 기준선(`docs/preexisting_files_manifest.json`, 2026-09-12 캡처)이 이번 작업 이전 세션에서 있었던 두 가지 정당한 변경 이후로 갱신되지 않았기 때문**입니다.

1. `test_no_preexisting_file_modified_deleted_or_moved` — `.gitignore`, `README.md` 두 파일의 해시가 기준선과 다르다고 보고합니다. 이 두 파일은 바로 이전 세션에서 사용자 승인 하에 의도적으로 수정하고 `origin/main`에 커밋·푸시한 파일입니다(커밋 `d00f165`). 실제 결함이 아닙니다.
2. `test_no_new_files_appeared_outside_extension_folder` — 이번 작업의 지시서 파일(`WorkOrder/CLIPtoXAI merged_project 사용 흔적 점검 및 README 갱신 작업지시서.md`)이 기준선 캡처 이후 사용자가 추가한 새 파일이라 감지됩니다. 실제 결함이 아닙니다. (`clip_xai_app/.pytest_cache/*`가 이 테스트의 첫 실행에서 함께 잡혔으나, 제가 방금 만든 pytest 캐시였으므로 삭제 후 재실행해 목록에서 제거했습니다.)
3. `test_git_diff_stat_of_tracked_files_matches_baseline` — 기준선 캡처 시점에는 `clip_xai_app`의 9개 파일이 "수정됐지만 미커밋" 상태였는데, 지금은 커밋되어 `git diff --stat`에 더 이상 나타나지 않습니다. 이 역시 정상적인 커밋의 결과입니다.

이번 작업 지시서는 "이번 작업에서 수정하는 파일은 README.md와 새 보고서 하나뿐"이라고 명시하므로, `graph_xai_extension`의 기준선 파일은 갱신하지 않았습니다. 실제로 `graph_xai_extension/` 폴더 자체나 원본 프로젝트 파일이 훼손된 것은 아니며, 위 3건은 전부 기준선이 최신 커밋 이력을 반영하지 못해서 생기는 오탐입니다. 다음에 `graph_xai_extension`을 다시 작업할 때 기준선을 v4로 갱신할 것을 권장합니다(아래 5절 TODO 참고).

## 4. README 변경 요약

전체 원문은 `README_backup_20260923.md`에 보존했습니다. 주요 변경 사항:

**추가**

- 프로젝트 개요를 "OASIS-3 단독(실제 실행)"과 "OASIS-3+ADNI 통합(검증되었으나 UI 미연결)" 두 갈래로 명시.
- 디렉터리 구조에 `merged_project/`, 분할 폴더 3개, `scripts/`, 최상위 `artifacts/`, `WorkOrder/`를 반영(기존 README는 이 5개를 전혀 언급하지 않았음).
- "모델 현황" 절: 통합/단독 모델을 별도 표로 구분하고, 이번 점검에서 CSV로 직접 재계산한 test 지표(CLIP Acc 60.91%/F1 0.5763, 3D CNN Acc 64.92%/F1 0.6217, GBM Acc 61.26%/F1 0.5450, 앙상블 Acc 65.62%/F1 0.6336, 전부 n=573)를 근거 파일과 함께 기재.
- "코드별 사용 모델" 절: `clip_xai_app`/`graph_xai_extension`/`scripts`가 각각 실제로 무엇을 로드하는지, 그리고 기존 `WorkOrder` 6개 지시서가 각각 어떤 모델을 대상으로 했는지 표로 정리.
- `merged_project` 재검증 명령과 2026-09-23 실행 결과를 "실행 방법"에 추가.
- "한계와 주의"에 통합 모델 라이브 추론 불가 사유, 해마 슬라이스 범위 주의, ESTsoft 임시폴더 테스트 환경 문제, 두 계열의 클래스 순서 검증 방식 차이를 추가.
- "변경 이력" 절 신설.
- 확인 불가 항목을 `TODO(확인 필요)`로 명시: (1) 통합 CLIP 체크포인트의 백본 정체(BiomedCLIP 여부), (2) OASIS-3 단독 모델의 전체 test-set(4,883 슬라이스) 정확도 — 임베딩 데이터가 로컬에 없어 재계산 불가, (3) 지시서에 언급된 "GBM baseline Acc 63.2%/F1 0.604", "subject Acc 72.2%/F1 0.435" 수치의 원출처를 저장소 안에서 찾지 못함.

**유지(사실과 일치, 그대로 보존)**

- 기존 "중요 주의사항"의 CLIP 512차원/L2 미정규화/클래스 순서/subject 평균/XAI target 정의 6개 항목 — 부록 절로 이동만 하고 문구는 그대로 유지.
- "지금까지 구현한 것" 1~6절(Handoff 검증, API 추가, Grad-ECLIP 수정, Gradio UI, PDF 보고서, 예측 근거 규칙), "주요 실행/검증 명령", "의존성", "남은 작업" — 부록 절로 이동, 문구는 그대로 유지(단, 단일/다중 슬라이스 예시 수치 옆에 "정확도 통계가 아니라 예시"라는 설명을 덧붙임).
- 이전 세션에서 추가한 "Graph XAI 확장" 소개 — 부록 절로 이동, 문구는 그대로 유지.

**수정(틀렸거나 불완전해서 고친 문장)**

- 원문: "이 저장소는 기존 handoff 패키지의 모델/가중치/전처리 기준을 재학습 없이 사용하고... Gradio UI를 제공합니다." → `merged_project` 통합 산출물의 존재를 전혀 언급하지 않아 불완전했으므로, 두 갈래 구성을 명시하는 문장으로 교체.
- Git/데이터 관리 절의 "무시 대상" 목록이 실제 `.gitignore`(그사이 갱신됨)와 어긋나 있었던 부분(`clip_xai_app/artifacts/*` 뭉뚱그림)을 실제 4개 세부 패턴 + `graph_xai_extension/outputs/`로 정정하고, `merged_project*`/`scripts`/`artifacts`/`WorkOrder`가 `.gitignore`에 등록되어 있지 않다는 사실을 새로 추가.

## 5. 남은 TODO와 후속 작업 제안

- TODO(확인 필요): 통합 CLIP 체크포인트(`clip_best.pt`)의 백본 정체. 체크포인트에는 `epoch`, `class_embeds`만 있어 backbone 식별자를 저장소 안에서 찾을 수 없습니다. 원본 학습 노트북(Colab, `/content/drive/MyDrive/...` 경로가 CSV에 남아 있음)을 확인해야 합니다.
- TODO(확인 필요): OASIS-3 단독 모델의 전체 test-set(4,883 슬라이스/194 subject) 정확도. `reports/validation_report.md`는 재현성(동일 임베딩 재계산 시 예측 194명 전원 일치)만 확인했을 뿐 실제 라벨 대비 정확도는 계산하지 않았고, 필요한 임베딩은 `.gitignore`로 제외되어 로컬에 없습니다.
- TODO(확인 필요): 이전 지시서에 언급된 "GBM baseline Acc 63.2%/Macro-F1 0.604", "subject Acc 72.2%/Macro-F1 0.435" 수치의 원출처.
- 후속 작업 제안 1: `graph_xai_extension/docs/preexisting_files_manifest.json`을 v4로 갱신해 3절에서 설명한 3건의 오탐을 해소할 것(이번 작업 범위 밖이라 진행하지 않음).
- 후속 작업 제안 2: `clip_xai_app`의 통합 모델 조회 어댑터(`merged_clip_reference`/`cnn3d_reference`/`gbm_reference`)를 실제로 UI에 연결할지(scan_id 검색 탭 추가) 결정. 현재는 코드만 있고 화면에 노출되지 않아 사용자가 존재를 알기 어렵습니다.
- 후속 작업 제안 3: 앙상블 결합 규칙 복원(`final_ensemble_results.csv`의 `pred`가 GBM/CNN/CLIP 확률의 단순 평균과 573건 중 9건에서 다름을 이번 점검에서 직접 확인) — 원 학습 코드나 노트북을 확보해야 재현 가능합니다.
- 후속 작업 제안 4: 대용량 미추적 데이터 정리. `merged_project*` 4개 폴더 합계 약 190,850개 파일·19.25GB가 git에 커밋되지 않은 채 로컬에만 존재합니다. 백업 정책과 `.gitignore` 등록 여부를 별도로 결정할 것을 제안합니다(이번 작업에서는 `.gitignore`를 변경하지 않았습니다).
- 후속 작업 제안 5: 분할 압축본 3개(`merged_project-20260825T072246Z-1-00{1,2,3}`) 삭제. 2절에서 확인했듯 `merged_project/`와 무손실 병합이 완료되었으므로 안전하게 삭제 가능합니다(이번 작업에서는 삭제 금지 지시에 따라 그대로 두었습니다).
