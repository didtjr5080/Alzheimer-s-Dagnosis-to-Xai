# CLIPtoXAI

CLIP 기반 XAI 알츠하이머 진단 지원 연구용 프로토타입입니다(CN/MCI/AD 3-class). 저장소에는 두 갈래 파이프라인이 모두 **라이브 추론**으로 들어 있습니다.

- **OASIS-3 단독**: `CLIP image embedding + Logistic Regression` handoff 패키지. `clip_xai_app`(legacy 탭)와 `graph_xai_extension` 두 Gradio 앱이 이 모델로 실시간 추론을 수행합니다. 신규 PNG 업로드를 지원하는 유일한 경로입니다.
- **통합(OASIS-3+ADNI)**: `merged_project/`의 CLIP(class-embedding 유사도) + 3D CNN + GBM baseline을 `clip_xai_app`(통합 모델 탭, 기본값)에서 `scan_id` 기준으로 라이브 추론합니다. 세 브랜치 모두 test-set(n=573) 재현 검증을 이 저장소의 CSV·체크포인트만으로 통과했습니다(아래 "모델 현황" 참고). 신규 업로드는 지원하지 않으며, 저장소에 캐시된 scan_id만 조회할 수 있습니다.

2026-09-23 최초 점검: [WorkOrder/merged_project_점검보고서_20260923.md](WorkOrder/merged_project_점검보고서_20260923.md). 같은 날 라이브 추론 전환 및 검증: [WorkOrder/통합모델_라이브추론_보고서_20260923.md](WorkOrder/통합모델_라이브추론_보고서_20260923.md).

## 디렉터리 구조

```text
CLIPtoXAI/
├─ README.md
├─ README_backup_20260923.md            # 이번 갱신 전 README 백업
├─ .gitignore
├─ clip_lr_grad_eclip_handoff_v1_20260812_105027/   # OASIS-3 단독 CLIP+LR handoff 패키지 (실제 사용 모델)
│  ├─ checksums.sha256
│  ├─ code/  (verify_handoff.py, inference_clip_lr.py, grad_eclip.py, gradient_smoke_test.py)
│  ├─ data/ metadata/ models/ reports/
├─ merged_project/                       # OASIS-3+ADNI 통합 산출물 (95,425 파일, 9.63GB, .gitignore 등록됨). 코드는 없고 CSV/checkpoint/이미지만 존재
│  ├─ unified_manifest.csv, unified_slice_manifest.csv, unified_final_features_gbm.csv
│  ├─ clip_test_probs.csv, 3dcnn_test_probs.csv, gbm_hierarchical_test_probs.csv, final_ensemble_results.csv
│  ├─ checkpoints/ (clip_best.pt, clip_latest.pt, 3dcnn_best.pt, 3dcnn_latest.pt)
│  ├─ models/ (gbm_baseline.txt, gbm_stage1.txt, gbm_stage2.txt)
│  ├─ mri_3d_cache/ slices_multi/ gradcam_results/
├─ merged_project-20260825T072246Z-1-001/  ┐
├─ merged_project-20260825T072246Z-1-002/  ├─ Google Drive 분할 다운로드 원본 3개(.gitignore 등록됨). 각 폴더 안에 자체 `merged_project/`
├─ merged_project-20260825T072246Z-1-003/  ┘  하위 폴더가 있으며, 세 폴더의 파일 집합은 서로 겹치지 않고(중복 0건) 합쳐서 위의
│                                             `merged_project/`와 바이트 단위로 완전히 일치합니다(2026-09-23 재검증). 이미 병합이
│                                             끝난 원본이라 삭제해도 데이터 손실은 없지만, 이번 작업 범위상 삭제하지 않았습니다.
├─ clip_xai_app/                         # 통합 모델(기본)/OASIS-3 단독(legacy) 2탭 Gradio 앱
│  ├─ app.py  README.md  requirements.txt
│  ├─ configs/ (explanation.yaml, model_registry.yaml)
│  ├─ src/ (config.py, model_loader.py, inference.py, xai.py, visualization.py, report.py,
│  │        schemas.py, warnings.py, contracts.py, model_registry.py, merged_artifacts.py,
│  │        merged_inference.py, merged_region_analysis.py, merged_pdf_report.py,
│  │        xai_artifacts.py, xai_methods.py,
│  │        adapters/ (legacy_clip_lr.py, merged_reference.py, merged_clip.py, merged_grad_eclip.py,
│  │                    merged_cnn3d.py, merged_cnn3d_gradcam.py, merged_gbm.py, merged_ensemble.py),
│  │        region_xai/ (3x3(x3) 구역 마스킹 민감도 분석 -- graph_xai_extension과 별개의 독립 구현:
│  │                      schemas.py, masking.py, region_grid_2d.py, region_grid_3d.py, perturbation.py,
│  │                      ranking.py, stability.py, agreement.py, graph_builder.py, visualization.py,
│  │                      clinical_wording.py))
│  ├─ tests/
│  ├─ artifacts/ (verification/, heatmaps/, reports/, xai/ —모두 생성물, git 추적 제외)
│  └─ handoff/
├─ graph_xai_extension/                  # OASIS-3 단독 모델만 사용하는 독립 XAI 확장 (읽기 전용, clip_xai_app 미수정)
│  ├─ run_graph_xai.py  README.md
│  ├─ graph_xai/  tests/  docs/  scripts/
│  └─ outputs/                            # 생성물, git 추적 제외
├─ scripts/                              # merged_project 분할본 병합·검증·점검 스크립트 (모델 실행 아님)
│  ├─ merge_merged_parts.ps1  validate_merged_project.py
│  └─ inspect_merged_parts.ps1  inspect_merged_parts_fast.py
├─ artifacts/                            # merged_project 병합/검증 로그 전용 (모델 추론 결과 아님)
│  └─ merge_validation/
└─ WorkOrder/                            # 작업지시서 보관 (7개, 아래 "코드별 사용 모델" 참고)
```

## 모델 현황

### 통합 (OASIS-3+ADNI, `merged_project/`) — 2026-09-23부로 라이브 추론

`통합(OASIS-3+ADNI) 모델 라이브 추론 전환 작업지시서.md`에 따라 아래 세 브랜치 + 고정 가중 앙상블을 저장된 가중치에서 직접 로드해 추론하도록 전환했습니다(재학습 없음). test-set(n=573) 재현 검증은 이 저장소의 CSV/체크포인트만으로 수행했고, 모든 확률·정확도·Macro-F1·혼동행렬이 예상값과 일치했습니다(증거: `WorkOrder/통합모델_라이브추론_보고서_20260923.md`).

| 모델 | 파일 경로 | 백본/알고리즘 | 학습 데이터 | test 지표 (이 세션에서 직접 재현) | 현재 사용 여부 |
| --- | --- | --- | --- | --- | --- |
| CLIP (class-embedding 유사도) | `merged_project/checkpoints/clip_best.pt` | `openai/clip-vit-base-patch16`(`transformers.CLIPModel`, 프리즈, OASIS-3 단독 handoff와 동일 로컬 스냅샷, commit `57c21647...`) — **BiomedCLIP 아님, 재현으로 확정**. 이미지 임베딩 L2 정규화 후 `class_embeds`(3,512, L2 정규화)와 코사인 유사도 × `logit_scale.exp()`(=100.0) | unified_manifest 기준 train 2,671 / val 572 / test 573 (OASIS3 1,294 + ADNI 2,522) | n=573, Acc=60.9%, Macro-F1=0.576, 혼동행렬 `[[212,47,7],[86,81,25],[18,41,56]]` — 지시서 기준값과 완전 일치, `clip_test_probs.csv` 대비 최대오차 1e-6 | `clip_xai_app` 통합 모델 탭에서 scan_id로 **라이브 추론** (checkpoint `epoch=13`) |
| 3D CNN | `merged_project/checkpoints/3dcnn_best.pt` | `Simple3DCNN_Stable`(4-block Conv3D+GroupNorm+ReLU+MaxPool, `strict=True` state_dict 로드), 입력 `merged_project/mri_3d_cache/{scan_id}.npy` (98,116,94) float16→float32, 추가 정규화 없음 | 위와 동일 | n=573, Acc=64.9%, Macro-F1=0.622, 혼동행렬 `[[224,39,3],[86,72,34],[6,33,76]]` — 완전 일치, `3dcnn_test_probs.csv` 대비 최대오차 1.8e-7 | `clip_xai_app` 통합 모델 탭에서 scan_id로 **라이브 추론** (checkpoint `epoch=56`) |
| GBM baseline | `merged_project/models/gbm_baseline.txt` | LightGBM Booster(`num_model_per_iteration=3`, 600 trees), 11개 피처(`Hippocampus_ratio` 등 구조 부피비 8종 + `AgeatEntry`/`GENDER`/`is_adni`); `is_adni`는 `unified_final_features_gbm.csv`에 없는 파생 컬럼이라 `dataset_source=="ADNI"`에서 직접 구성, 결측치 0건(원본 노트북처럼 별도 대치 없음) | 위와 동일 | n=573, Acc=63.2%, Macro-F1=0.604, 혼동행렬 `[[207,43,16],[65,80,47],[10,30,75]]` — 완전 일치, `final_ensemble_results.csv`의 `gbm_*` 대비 최대오차 0 | `clip_xai_app` 통합 모델 탭에서 scan_id로 **라이브 추론** |
| 앙상블 | 위 세 브랜치의 확률을 조합 | 고정 가중합 `GBM 0.35 + 3D CNN 0.45 + CLIP 0.20`(재정규화 없음, GBM은 반드시 baseline — 계층형을 쓰면 재현되지 않음을 확인) | 위와 동일 | n=573, Acc=65.6%, Macro-F1=0.634, 혼동행렬 `[[219,43,4],[79,77,36],[9,26,80]]` — 완전 일치, `final_ensemble_results.csv`의 `pred`와 573/573 일치 | `clip_xai_app` 통합 모델 탭에서 scan_id로 **라이브 추론**; 세 브랜치 중 하나라도 없는 scan은 앙상블도 "사용 불가" (재가중 없음) |

참고: `merged_clip_reference`/`cnn3d_reference`/`gbm_hierarchical_reference`(기존 test-set CSV 단순 조회 어댑터, `src/adapters/merged_reference.py`)는 위 라이브 어댑터와의 교차검증용으로 계속 남아 있습니다.

### OASIS-3 단독 (`clip_lr_grad_eclip_handoff_v1_20260812_105027/`)

| 모델 | 파일 경로 | 백본/알고리즘 | 학습 데이터 | test 지표 | 현재 사용 여부 |
| --- | --- | --- | --- | --- | --- |
| CLIP+LR | `models/clip_model/model.safetensors`, `models/clip_lr_classifier_new_run.joblib`, `models/clip_lr_weights.npz` | `openai/clip-vit-base-patch16`(`transformers.CLIPModel`, 프리즈) + sklearn LogisticRegression | OASIS-3 단독. 데이터 키 `MR_ID`(`data/manifests/slice_manifest_portable.csv` 헤더 직접 확인) | TODO(확인 필요): 저장소에 라벨이 있는 전체 test manifest(4,883행/194 subject, `reports/validation_report.md`에 행수만 기록)의 임베딩/이미지가 없어 subject-level accuracy를 직접 재계산할 수 없음. 아래 예시 값(단일/다중 슬라이스 1건)은 정확도 통계가 아니라 검증용 예시입니다. | `clip_xai_app`(legacy 탭, `analyze()` → `analyze_subject()` → `model_loader.load_pipeline()`)와 `graph_xai_extension` 둘 다 이 모델로 **실시간 추론**을 수행. 유일하게 신규 PNG 업로드를 지원하는 경로 |

## 코드별 사용 모델

| 코드 | 실제 실행 경로에서 사용하는 모델 | 근거 |
| --- | --- | --- |
| `clip_xai_app` | 탭 2개: **"통합 모델(OASIS-3+ADNI)"**(기본값) — `app.py::analyze_merged_ui()` → `src/merged_inference.py::analyze_merged_scan()` → `src/adapters/{merged_clip,merged_cnn3d,merged_gbm,merged_ensemble}.py` → `merged_project/`의 체크포인트·부스터를 scan_id로 **라이브 추론**. **"OASIS-3 단독(legacy)"** — `app.py::analyze()` → `src/inference.py::analyze_subject()` → `src/model_loader.py::load_pipeline()` → OASIS-3 단독 handoff. 두 탭 모두 UI에서 실제로 실행 가능 | `clip_xai_app/app.py`, `src/merged_inference.py`, `src/adapters/merged_*.py`, `src/model_loader.py:19-24` |
| `graph_xai_extension` | OASIS-3 단독 handoff만 사용(`graph_xai/legacy_adapter.py`). `merged_project`는 자체 파일-무결성 테스트에서 스캔 대상 제외 목록에만 등장하며(`scripts/_regen_baseline_manifest.py:17-20`), 모델로 로드하지 않음. 이번 통합 라이브 추론 전환 작업에서 수정하지 않음 | `graph_xai_extension/graph_xai/legacy_adapter.py`, `graph_xai_extension/scripts/_regen_baseline_manifest.py` |
| `scripts/` | 모델을 로드하지 않음. `merge_merged_parts.ps1`은 분할 3폴더를 해시 비교 후 `merged_project/`로 복사, `validate_merged_project.py`는 병합 결과의 파일 수·크기·CSV 스키마·체크포인트 `torch.load` 구조(추론이 아닌 구조 확인)를 검증 | `scripts/merge_merged_parts.ps1`, `scripts/validate_merged_project.py` |

`WorkOrder/` 지시서별 대상 모델:

| 지시서 | 대상 모델 경로 |
| --- | --- |
| `Codex_merged_project_병합_검증_작업지시서.md` | `merged_project*` (병합/검증 스크립트 작성 지시) |
| `Codex_XAI_다음단계_통합구현_작업지시서.md` | `merged_project/` + `clip_lr_grad_eclip_handoff...` (모델 레지스트리/어댑터 계층 도입 지시) |
| `Codex_XAI_히트맵_식별검증_개선_수정작업지시서.md` | `clip_lr_grad_eclip_handoff_v1_20260812_105027/` 단독 |
| `Alzheimer_XAI_GraphXAI_Codex_통합프롬프트.md`, `Graph_XAI_추가개선_수정작업지시서.md`, `Graph_XAI_의료진용_보고서_개선_작업지시서.md` | `clip_lr_grad_eclip_handoff_v1_20260812_105027/` 단독 (`graph_xai_extension` 구현) |
| `CLIPtoXAI merged_project 사용 흔적 점검 및 README 갱신 작업지시서.md` | 이 문서 자체 (점검 대상 지시서) |
| `통합(OASIS-3+ADNI) 모델 라이브 추론 전환 작업지시서.md` | `merged_project/` + `clip_lr_grad_eclip_handoff...`(백본 재사용) — 통합 모델 라이브 추론 전환 지시 |

## 실행 방법

가장 간단한 방법: 저장소 루트의 [run_app.bat](run_app.bat)을 더블클릭(또는 `run_app.bat` 실행). 의존성 설치 확인 후 앱을 바로 실행합니다. Python이 PATH에 없으면 안내 메시지를 출력하고 종료합니다.

수동 실행, `clip_xai_app` (통합 모델 탭 기본 표시 + OASIS-3 단독 legacy 탭):

```powershell
cd E:\Develop\CLIPtoXAI
python -m pip install -r clip_xai_app\requirements.txt
python clip_xai_app\app.py
```

접속: `http://127.0.0.1:7860/` — "통합 모델(OASIS-3+ADNI)" 탭에서 scan_id 입력(test split 예시 드롭다운 제공), "OASIS-3 단독(legacy)" 탭에서 PNG 업로드.

Graph XAI 확장(동일 모델, 3×3 구역 민감도 분석 + PDF 보고서):

```powershell
cd E:\Develop\CLIPtoXAI
python -m pip install -r graph_xai_extension\requirements_graph_xai.txt
python graph_xai_extension\run_graph_xai.py
```

`merged_project` 무결성 재검증(읽기 전용, 모델 실행 아님):

```powershell
cd E:\Develop\CLIPtoXAI
python scripts\validate_merged_project.py
```

2026-09-23 재실행 결과: `Files expected/actual/missing/extra/size_mismatch: 95425/95425/0/0/0` (`artifacts/merge_validation/validate_20260923_185436.json`).

`merged_project` 병합(이미 완료됨 — 대상 폴더가 비어 있지 않으면 안전하게 거부하고 종료 코드 3을 반환하므로 재실행해도 무해합니다):

```powershell
cd E:\Develop\CLIPtoXAI
powershell -ExecutionPolicy Bypass -File scripts\merge_merged_parts.ps1
```

## 한계와 주의

- 연구용 프로토타입입니다. 의료기기가 아니며 진단, 치료 결정, 개별 환자에 대한 임상 결론에 사용할 수 없습니다.
- 히트맵/Grad-ECLIP/Grad-CAM 결과는 모델이 참조한 영역을 근사적으로 보여줄 뿐, 진단 근거가 아니며 해부학적으로 검증되지 않았습니다(`clip_lr_grad_eclip_handoff_v1_20260812_105027/reports/grad_eclip_validation.md`).
- OASIS-3 coronal 슬라이스 인덱스 cor069–092 구간에는 해마 복셀이 거의 포함되지 않습니다. 이 범위의 슬라이스로 만든 히트맵을 해마 관련 소견으로 해석하지 마세요.
- 통합 모델은 `merged_project/`에 캐시된 scan_id만 조회할 수 있습니다(현재 3,816개 scan 전부 캐시 있음). 신규로 업로드한 원본 이미지는 이 파이프라인의 전처리(percentile 정규화, MNI 정합 등)가 적용되어 있지 않아 사용할 수 없습니다 — 신규 업로드는 "OASIS-3 단독(legacy)" 탭을 이용하세요.
- 3D CNN Grad-CAM은 마지막 conv block의 공간 해상도(6×7×5)에서 계산돼 캐시 볼륨 해상도(98×116×94)로 16~19배 업샘플링하므로 시각적으로 매우 흐릿하고 블록 형태입니다. 원본 스캔 해상도로는 되돌리지 않았습니다(작업지시서 3-3절 지시대로).
- 3D 볼륨의 axial/coronal/sagittal 축이 실제 해부학적 방향과 정확히 일치하는지는 배열 축 순서만으로 판단했으며, 참값(ground truth) 방향 라벨로 별도 검증하지 않았습니다. TODO(확인 필요).
- OASIS-3 단독 CLIP+LR 모델의 전체 test-set(4,883 슬라이스/194 subject) 정확도는 이 저장소 안의 파일만으로 재계산할 수 없습니다(임베딩 데이터가 `.gitignore`로 제외되어 있음). TODO(확인 필요).
- `clip_xai_app` 테스트 중 5건(라이브 추론 전환 이전 4건 + 이번에 추가된 PDF 보고서 테스트 1건)은 이 로컬 환경의 `C:\Users\Public\Documents\ESTsoft\CreatorTemp` 폴더 권한 문제로 `PermissionError`가 발생합니다(pytest의 `tmp_path` picker 문제이며 프로젝트 코드 결함이 아닙니다). `graph_xai_extension`은 동일 문제를 `tests/conftest.py`에서 우회하고 있습니다.
- 클래스 순서는 모든 브랜치가 `[CN, MCI, AD]`이지만 서로 다른 방식으로 검증됩니다: OASIS-3 단독은 `clf.classes_`, 통합 CSV/부스터는 컬럼 접두사(`clip_`, `prob_`, `gbm_`) 또는 `feature_name()`으로 재도출한 순서를 사용합니다. 어느 쪽도 인덱스를 하드코딩하지 않습니다.
- 통합 모델의 test 지표는 test-set(n=573) 재현 검증값이며, 개별 환자의 진단 정확도나 임상 성능을 의미하지 않습니다.

## 변경 이력

- **2026-10-01**: 통합 모델 탭에서 PDF 보고서와 별도로, 발표 자료에 바로 쓸 수 있도록 핵심 이미지 4종(원본 MRI, CLIP CAM 지도, 3D CNN CAM 중첩(coronal), Graph XAI(CLIP))을 개별 PNG + zip으로 다운로드하는 기능 추가(`merged_pdf_report.py::build_presentation_image_bundle`, UI에 "발표용 이미지 모음 (zip)" 다운로드 항목 신설). 기존에는 이 이미지들이 PDF 생성용 `TemporaryDirectory`에만 존재하다 삭제되어, PDF에서 스크린샷으로 잘라내는 것 외에는 재사용할 방법이 없었음. 3D CNN 데이터가 없는 이미지 업로드 경로에서는 3D CNN 이미지만 건너뛰고 나머지 3종은 그대로 제공(기존 None-안전 규칙과 동일). `tests/test_ui_e2e.py`에 zip 다운로드 검증 추가, 62개 테스트 전체 통과 확인.

- **2026-09-26**: `clip_xai_app` 통합 모델 탭의 PDF 보고서를 `graph_xai_extension`과 같은 구조(의료진용 3쪽 요약 + 연구자용 기술 부록)로 전면 교체하되, CLIP(3×3=9구역)과 3D CNN(3×3×3=27구역) 두 공간 브랜치 모두에 3x3 구역 마스킹(zero/mean/blur) 기반 CAM 민감도 분석을 적용하도록 확장(`graph_xai_extension`을 임포트하지 않는 독립 재구현, `src/region_xai/`). 지지/억제/절대 민감도 순위, 마스킹 방식 간 안정성, CAM-Perturbation 일치도, Graph XAI 네트워크(CLIP은 9노드 단일 그래프, 3D CNN은 27노드를 axis0 3개 층으로 나눈 소형 다중 패널)를 두 브랜치 각각에 대해 생성. 배경(뇌 밖) 마스킹을 CLIP 히트맵에도 적용(측정 결과 scan에 따라 배경에 투사되던 heat가 최대 96.4%에 달함을 확인). PyMuPDF로 12쪽 전 페이지 육안 검수를 거쳐 표 헤더 겹침 버그를 발견·수정. `src/report.py::create_merged_pdf_report`(구버전, 3쪽 요약뿐)는 미사용 코드로 삭제.
- **2026-09-26 (같은 날 후속)**: 통합 모델 탭에 scan_id 조회 외에 이미지 직접 업로드 분석을 추가(`merged_inference.py::analyze_merged_uploaded_image`). 업로드된 이미지는 3D CNN(MNI 정합 3D 볼륨 필요)과 GBM baseline(SynthSeg 구조적 부피비 필요) 입력을 만들 수 없으므로 CLIP 브랜치만 실행하고, 앙상블은 계산하지 않음(부재 브랜치를 재가중치로 메우지 않는 기존 규칙 유지). `merged_pdf_report.py`의 모든 섹션 빌더가 `cnn3d_analysis`/`cnn3d_volume`/`cnn3d_cam_volume`가 `None`인 경우를 안전하게 처리하도록 수정(3D CNN 관련 카드/이미지/순위/마스킹비교/안정성/일치도/그래프 절만 조건부로 생략, CLIP 절은 그대로 출력). PyMuPDF로 CLIP 전용 업로드 경로의 8쪽 보고서 전체를 육안 검수하여 정상 렌더링 확인. `app.py::analyze_merged_ui`의 갤러리 라벨이 업로드 이미지의 `rep_slice_index=None`을 `:03d` 형식으로 포맷하려다 `unsupported format string passed to NoneType` 예외로 죽던 버그를 발견·수정("CLIP 입력 이미지 (업로드됨)" 라벨로 분기 처리); 실제 `analyze_merged_ui`를 업로드 시나리오로 직접 실행해 에러 없이 상태/표/갤러리/PDF가 모두 생성됨을 확인.
- **2026-09-26 (같은 날 후속 2)**: `merged_project/`(체크포인트 포함 전체 ~9.6GB)는 git 대상이 아니라 팀원이 저장소만 clone해서는 통합 모델을 실행할 수 없으므로, 라이브 추론에 실제로 쓰이는 가중치 3개(`clip_best.pt`, `3dcnn_best.pt`, `gbm_baseline.txt`, 총 ~4.3MB)만 `merged_model_release/merged_model_files_20260926.zip`으로 압축해 배치 안내 README와 함께 별도 보관(팀원 공유용, git에는 포함하지 않음 — `.gitignore`에 `merged_model_release/` 추가). GBM/3D CNN의 scan_id 조회 기능은 이 3개 파일 외에 `unified_final_features_gbm.csv` 등 데이터 파일이 추가로 필요하지만, 이미지 업로드(CLIP 전용) 분석은 `clip_best.pt`만으로 동작한다. 그동안 커밋되지 않고 쌓여 있던 `WorkOrder/`, `scripts/`, 최상위 `artifacts/`(병합 검증 증거)와 이번 세션의 모든 `clip_xai_app` 변경사항을 함께 커밋·푸시(`b24c994`).
- **2026-09-26 (같은 날 후속 3)**: 부록 "남은 작업" 목록 중 테스트 관련 4개 항목을 정리. (1) `tests/test_report_helpers.py` 신설 — `src/report.py`의 PDF 조립용 순수 헬퍼 함수(구독자ID 파싱, 불확실성 등급 분류, 확률 합 검증, `_merged_*` 표 헬퍼 등) 17개 단위 테스트 추가. (2) `tests/test_ui_e2e.py` 신설 — `app.py::build_app()`의 실제 `gr.Blocks` 앱을 임시 포트로 기동해 `gradio_client`로 진짜 HTTP 요청을 보내는 3개 테스트(scan_id 분석, 이미지 업로드 분석, 잘못된 scan_id 처리) 추가, 기존 테스트들이 검증하지 않던 `.click()` 배선과 직렬화 경계까지 확인. (3) `clip_xai_app/pytest.ini` 신설 — 이 프로젝트가 쓰지 않는 `pytest-qt` 플러그인이 로컬 아나콘다 환경에 설치돼 있어 `pytest -q`가 즉시 `INTERNALERROR`로 죽던 문제, 그리고 로컬 환경에서 `tempfile.tempdir`이 권한 없는 폴더로 전역 오버라이드되어 `tmp_path`를 쓰는 모든 테스트가 `PermissionError`로 죽던 문제를 각각 `addopts`의 `-p no:pytest-qt`와 `--basetemp=.pytest_tmp`로 해결하고, 느린 전체 검증 테스트(`test_merged_full_validation.py`)에 `@pytest.mark.slow`를 붙여 기본 실행에서 제외되도록 정리. 결과적으로 `clip_xai_app/`에서 플래그 없이 `pytest -q`만 실행하면 62 passed, 4 deselected(약 80초)로 통과. (4) 테스트 파일에 실제 pytest 케이스가 없다는 항목은 재확인 결과 이미 해소된 상태였음(스텁/placeholder 0건).
- **2026-09-28 (2)**: 저장소 루트에 [run_app.bat](run_app.bat) 실행 파일을 추가. 더블클릭(또는 커맨드라인 실행)만으로 `clip_xai_app` 의존성 설치 확인 후 `python clip_xai_app\app.py`를 실행. 배치 파일에 한글 문자열을 넣었더니 cmd.exe가 UTF-8을 시스템 코드페이지로 잘못 해석해 `where python` 판정이 깨지고 이후 줄이 깨진 텍스트로 파싱되는 문제를 발견 — 배치 파일 특유의 인코딩 문제를 피하기 위해 모든 안내 문구를 영어로 작성. 실제로 배치 파일을 실행해 `http://127.0.0.1:7860` 응답을 확인.
- **2026-09-28**: 발표(비전문가 대상) 상황을 고려해, 의료진용 요약 2쪽("영상으로 보는 설명")에 CLIP·3D CNN 각각의 "처리 전/처리 후" 비교 패널과 쉬운 한글 캡션을 추가(`region_xai/clinical_wording.py::build_before_after_caption`, `merged_pdf_report.py::_clinical_before_after_panel`). 절대 민감도 1위 구역(패널 4에서 빨간 박스로 표시되는 구역과 동일)에 실제로 마스킹을 적용한 전/후 이미지를 나란히 보여주고, 그 아래에 "처리 전 모델 출력값 66.76% / 처리 후 모델 출력값 20.55% / 변화량 -46.21% / 적용한 처리 방식: 평균 밝기로 가림"처럼 숫자와 처리 방식을 그대로 문장으로 읽을 수 있게 표기(기존에는 이미지만 있고 수치 설명이 없어 히트맵을 직접 해석해야 했음). scan_id 조회 경로(3D CNN 포함, 13쪽)와 이미지 업로드 경로(CLIP 전용, 8쪽) 양쪽 모두 PyMuPDF로 육안 검수해 정상 렌더링 확인.
- **2026-09-23 (2)**: `통합(OASIS-3+ADNI) 모델 라이브 추론 전환 작업지시서.md`에 따라 `merged_project`의 CLIP/3D CNN/GBM baseline을 저장된 가중치에서 직접 로드하는 라이브 추론으로 전환(재학습 없음). 세 브랜치 + 고정 가중 앙상블 모두 test-set(n=573) 재현 검증에서 지시서의 모든 기준값(확률/argmax/Acc/Macro-F1/혼동행렬/소스별 분해)과 완전히 일치. CLIP class-embedding 및 3D CNN용 Grad-CAM XAI 추가. `clip_xai_app` UI를 통합 모델(기본)/legacy 두 탭으로 재구성. 통합 CLIP 백본이 `openai/clip-vit-base-patch16`(BiomedCLIP 아님)임을 재현으로 확정. 상세 근거는 `WorkOrder/통합모델_라이브추론_보고서_20260923.md` 참고.
- **2026-09-23 (1)**: `merged_project` 사용 흔적을 전수 점검하고 README를 갱신. 통합(OASIS-3+ADNI) 산출물이 실제로 존재·검증되었으나 당시에는 UI 경로에 연결되어 있지 않았다는 점(이후 위 항목에서 연결됨), `merged_project`가 3개 분할 폴더의 무손실 병합 결과라는 점을 명시. 상세 근거는 `WorkOrder/merged_project_점검보고서_20260923.md` 참고.
- **(이전)**: `graph_xai_extension`(3×3 구역 민감도 분석 + Graph XAI + 의료진용/기술용 PDF 보고서) 추가, `clip_xai_app`에 모델 레지스트리/어댑터 계층과 XAI 아티팩트 식별·색인 체계 추가.

---

## 부록: Graph XAI 확장 (`graph_xai_extension/`) 상세

`clip_xai_app`과 동일한 OASIS-3 단독 CLIP+LR 분류기와 Grad-ECLIP CAM을 **읽기 전용**으로 재사용해서 만든 독립 확장입니다. 원본 `clip_xai_app` 코드/모델/가중치는 전혀 수정하지 않으며, 자체 폴더 안에서만 동작합니다. `merged_project`는 사용하지 않습니다.

주요 기능:

- 3×3 공간 구역 분할 + zero/mean/blur 마스킹 기반 perturbation 분석 (구역이 모델 출력에 얼마나 민감한지 측정 — 해부학적 의미 아님)
- 지지 근거(probability_drop > 0) / 억제 근거(< 0) / 절대 민감도(방향 무관) 3종 분리 순위
- 마스킹 방식 간 안정성 지표(Spearman/Kendall, 부호 일치율)와 CAM-perturbation 탐색적 일치도
- NetworkX/Plotly 기반 Graph XAI(구역 간 공간 인접성 그래프, 실제 신경 연결 아님)
- CN/MCI/AD 클래스 순서를 분류기의 `classes_`에서 직접 검증(하드코딩 금지) + 다중 샘플 배치 평가(피험자 단위 중복 제거)
- PDF 보고서 3종 모드: `clinical_summary`(의료진용 2~3쪽 요약), `technical_full`(연구자용 상세), `combined`(기본값, 요약 + 기술 부록), 연구 검토자 확인란 포함
- CSV/JSON/HTML 내보내기, 독립 Gradio UI

자세한 구현 내역, 테스트 결과, 원본 파일 무결성 검증 기록은
[graph_xai_extension/README.md](graph_xai_extension/README.md)와
[graph_xai_extension/docs/IMPLEMENTATION_REPORT.md](graph_xai_extension/docs/IMPLEMENTATION_REPORT.md)를 참고하세요.

## 부록: OASIS-3 단독 파이프라인 구현 상세

다음 내용은 `clip_xai_app`이 실제로 실행하는 OASIS-3 단독 CLIP+LR 파이프라인에 대한 기존 구현 기록입니다.

### 중요 주의사항

- 모델은 `openai/clip-vit-base-patch16` CLIP image encoder와 sklearn Logistic Regression 조합입니다.
- CLIP 임베딩은 512차원이며 L2 정규화를 추가로 적용하지 않습니다.
- 클래스 순서는 저장된 `clf.classes_` 기준으로 `[0, 1, 2] = [CN, MCI, AD]`를 검증해 사용합니다.
- subject 예측은 동일 subject의 슬라이스별 `predict_proba`를 단순 평균합니다.
- XAI target은 선택 클래스의 LR logit `z_c = W_c f_CLIP(x) + b_c`입니다.
- XAI forward에서는 `torch.no_grad()`를 사용하지 않습니다.

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
- 모델 가용성 표(통합 산출물 레지스트리 상태 표시, 조회 기능은 아직 없음)
- traceback은 UI에 노출하지 않고 `clip_xai_app/artifacts/verification/ui_errors.log`에 기록

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

단일 슬라이스 기준 검증 샘플(정확도 통계가 아니라 재현성 확인용 단일 예시입니다):

```text
OAS30009_MR_d2457_cor094.png

CN  = 0.9180
MCI = 0.0619
AD  = 0.0200
Top-1 = CN
Top-2 = MCI
Margin = 85.61%p
```

다중 슬라이스 subject 평균 검증 샘플(위와 동일 subject, 정확도 통계 아님):

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

### 주요 실행/검증 명령

전체 테스트(빠른 스위트만, 기본값 — `clip_xai_app/pytest.ini` 참고):

```powershell
cd clip_xai_app
python -m pytest -q
```

느린 전체 검증(573-scan 재현, 약 7-8분)까지 포함:

```powershell
python -m pytest tests/test_merged_full_validation.py -v -m slow
```

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

### 의존성

주요 설치 패키지:

```text
gradio 6.4.0
fastapi 0.141.1
starlette 0.52.1
python-multipart 0.0.32
torch 2.5.1+cu121
transformers 4.56.2
scikit-learn 1.5.2
lightgbm 4.7.0        # 통합 GBM baseline 라이브 추론 (2026-09-23 추가)
reportlab 4.2.2
pypdf
PyMuPDF
```

설치:

```powershell
python -m pip install -r clip_xai_app\requirements.txt
```

### Git/데이터 관리

[.gitignore](.gitignore)는 생성 artifact와 대용량/민감 가능성이 있는 로컬 payload를 제외하도록 설정했습니다.

무시 대상:

- `clip_xai_app/artifacts/verification/*`, `heatmaps/*`, `reports/*`, `xai/*`, `xai_artifact_index.csv` 생성물
- `clip_xai_app/handoff/*`
- handoff 패키지의 `models/`
- handoff 패키지의 `data/embeddings/`
- handoff 패키지의 `data/xai_samples/images/`
- `graph_xai_extension/outputs/` 생성물

`merged_project/`와 3개 분할 폴더(`merged_project-20260825T...-1-*`)는 대용량 데이터(총 약 190,850 파일/약 19.3GB)라 `.gitignore`에 등록되어 git에 커밋되지 않습니다. 팀 공유용으로 라이브 추론에 실제 필요한 가중치 3개(`clip_best.pt`, `3dcnn_best.pt`, `gbm_baseline.txt`, ~4.3MB)만 압축한 `merged_model_release/`도 같은 이유로 `.gitignore`에 등록되어 있습니다(배치 안내는 `merged_model_release/README_모델파일_배치.txt` 참고). 반면 최상위 `artifacts/`(병합 검증 증거), `scripts/`(병합/검증 스크립트), `WorkOrder/`(작업지시서 기록)는 2026-09-26에 git에 커밋되었습니다.

### 남은 작업

아래 5개 항목은 이 부록이 처음 작성됐던 시점(OASIS-3 단독 handoff 검증 단계)의 목록이며, 통합 모델 라이브 추론 전환(2026-09-23)과 이번 정리(2026-09-26) 이후로는 모두 해소되었습니다:

- ~~테스트 파일에 실제 pytest 케이스 작성~~ — 2026-09-26 확인: 전체 테스트 파일(`tests/*.py`)에 스텁/placeholder 없이 실제 assertion이 있는 케이스만 존재함을 재확인(`pass`/`assert True`/`TODO`/`NotImplementedError` 패턴 grep 결과 0건).
- ~~PDF 보고서 생성 로직의 단위 테스트 추가~~ — 2026-09-26: `tests/test_report_helpers.py` 신설. `src/report.py`의 순수 헬퍼(`infer_subject_id`, `validate_same_mr_id`, `classify_output_separation`, `_assert_probabilities_sum_to_one`, `_short_digest`, `_prediction_status_text`, `_artifact_slice_token`, `_merged_cell`/`_merged_branch_prob_table`/`_merged_kv_table`)를 대상으로 17개 단위 테스트 추가(기존에는 `create_basic_pdf_report`/`build_merged_full_pdf_report` 같은 최종 조립 함수를 통해서만 간접 검증됨).
- ~~UI end-to-end 테스트 자동화~~ — 2026-09-26: `tests/test_ui_e2e.py` 신설. `app.py::build_app()`이 반환하는 실제 `gr.Blocks` 앱을 임시 포트로 실제 기동한 뒤 `gradio_client`로 실 HTTP 요청을 보내 `.click()` 배선·컴포넌트 직렬화까지 검증(scan_id 조회, 이미지 업로드, 잘못된 scan_id 처리 3케이스). 기존 테스트들은 `analyze_merged_scan` 등 내부 함수를 직접 호출해 UI 배선 자체는 검증하지 않았음.
- ~~전체 `pytest -q` 통과 기준 정리~~ — 2026-09-26: `clip_xai_app/pytest.ini` 신설. (1) 이 프로젝트가 쓰지 않는 `pytest-qt` 플러그인이 이 저장소가 아닌 로컬 아나콘다 환경에 깔려 있어 `pytest -q` 실행 시 `INTERNALERROR`로 즉시 죽던 문제를 `addopts = -p no:pytest-qt`로 해결. (2) 느린 전체 검증(`test_merged_full_validation.py`, 약 7-8분)에 `@pytest.mark.slow`를 붙이고 `addopts`에 `-m "not slow"`를 추가해 기본 실행에서 자동 제외(명시 실행은 `pytest tests/test_merged_full_validation.py -v -m slow`). (3) 이 개발 환경에서 ESTsoft 계열 도구로 추정되는 무언가가 `tempfile.tempdir`을 프로세스 전역으로 권한 없는 폴더(`.../ESTsoft/CreatorTemp`)로 덮어써 `tmp_path`를 쓰는 모든 테스트가 `PermissionError`로 죽던 문제를, 저장소 상대 경로(`--basetemp=.pytest_tmp`, `.gitignore` 등록)로 고정해 해결. 결과적으로 `clip_xai_app/`에서 별도 플래그 없이 `pytest -q`만 실행하면 통과함(확인: 62 passed, 4 deselected, ~80초).

여전히 별도 연구 과제로 남아 있는 항목:

- XAI 전경/배경 QC 수치 연동 가능 여부 검토
- 전체 슬라이스 XAI 일관성 검증(별도 연구 과제로 분리)
