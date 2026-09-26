# 통합 모델 XAI 후속 수정 작업지시서

Sep 23, 2026 · @Someone

## 0. 목적과 범위

`통합모델_라이브추론_보고서_20260923.md` 7절의 남은 문제 중 세 가지를 처리한다. 모델 추론 결과(확률·정확도)는 바꾸지 않으며, 전환 작업의 전체 재현 테스트 4개(`test_merged_full_validation.py`)가 작업 후에도 그대로 통과해야 한다.

| 항목 | 내용 | 수정 대상 |
| --- | --- | --- |
| A | 3D Grad-CAM 축 라벨 수정, 균일 히트맵 원인 점검 | `src/adapters/merged_cnn3d_gradcam.py` |
| B | `tmp_path` 권한 오류 우회 | `clip_xai_app/tests/conftest.py` |
| C | README 문구 정리, 통합 탭 scan\_id 선택 범위 확장 | `README.md`, `clip_xai_app/app.py` |

A → B → C 순서로 진행한다. A의 원인 점검 결과는 수정 전에 먼저 보고서에 기록한다.

## A. 3D Grad-CAM 축 라벨과 균일 히트맵

`axis_check.png`(scan `OAS30564_MR_d0000`)로 캐시 볼륨 `(98, 116, 94)`의 축을 실제 영상으로 확인했다. 배열이 표준 `(x, y, z)` 순서가 아니므로, 기존 `(d,h,w)` 가정과 이전 판의 매핑을 모두 아래 표로 바꾼다.

**주의**: `slices_multi`의 `cor` 파일은 이름과 달리 **axial 단면**이다(axis 1 절단, 90° 회전).

| 배열 축 | 캐시 크기 | 실제 단면 (영상 확인) | 표시 방법 | 근거 |
| --- | --- | --- | --- | --- |
| axis 0 | 98 | coronal | `vol[i, :, :]` | 좌우 대칭, 아래쪽에 소뇌, 측뇌실 보임 |
| axis 1 | 116 | axial | `np.rot90(vol[:, j, :])` | 타원형 윤곽, 대뇌 종렬이 가로로 지남, `slices_multi`와 동일 |
| axis 2 | 94 | sagittal | `vol[:, :, k]` | 뇌량 아치와 소뇌가 옆모습으로 보임 |

### A-1. 원인 점검 (수정 전, 결과를 보고서에 기록)

test scan 3개(CN·MCI·AD 각 1개, 기존 PDF에 쓴 `OAS30564_MR_d0000` 포함)에 대해:

1. 업샘플 전 CAM 원값의 shape, min, max, mean, std를 출력한다. std가 max의 5% 미만이면 "거의 상수"로 판정한다.
2. 활성화와 기울기 각각의 shape와 채널 평균 가중치(`alpha`) 분포(양수 비율)를 출력한다.
3. ReLU 적용 전후 0이 아닌 voxel 비율을 출력한다.
4. 업샘플 방식이 `F.interpolate(cam[None, None], size=vol.shape, mode="trilinear", align_corners=False)`인지 확인한다. 다르면 기존 방식을 적는다.
5. 축 매핑 수정 전후의 3단면 이미지를 나란히 저장해 비교한다.

### A-2. 수정

- 위 표의 축 매핑으로 단면 추출과 라벨을 고친다.
- 업샘플은 A-1의 4번 방식으로 통일한다.
- 정규화는 볼륨 전체 기준 min-max로 한 번만 하고, 단면별로 다시 정규화하지 않는다.
- 비교용 옵션으로 `target_layer="block3"`(약 12×14×11)을 추가한다. 기본값은 원본 셀 52와 같은 `block4`로 둔다.
- 원본 뇌 볼륨의 배경(0 근처)에서는 히트맵을 투명하게 처리해, 뇌 밖이 붉게 칠해지지 않도록 한다.

### A-3. 판정

- 수정 후 라벨은 위 표를 따라야 한다. axial 패널은 같은 scan의 `slices_multi` PNG와 방향이 같아야 하고, coronal 패널에는 소뇌가 아래쪽에, sagittal 패널에는 뇌량 아치가 보여야 한다.
- 히트맵이 여전히 절반을 균일하게 덮으면 A-1 수치와 함께 "해상도 한계"로 결론짓고, `block3` 결과를 함께 보고한다. 임의로 보기 좋게 만드는 후처리(스무딩, 임계값 조정)는 하지 않는다.
- 좌우(L/R) 방향 검증은 이번 범위 밖이다. 라벨 옆에 "좌우 미검증"을 표기한다.

### A-4. PDF 표기 수정 (`merged_1ae74a37de03.pdf` 검수 결과)

| 위치 | 현재 | 변경 |
| --- | --- | --- |
| 3D Grad-CAM 1번 그림 | "axial" | "coronal" (소뇌가 아래에 보임) |
| 3D Grad-CAM 2번 그림 | "coronal" | "axial" (A 표의 axis 1, `rot90` 적용) |
| 3D Grad-CAM 3번 그림 | "sagittal" | 유지 |
| CLIP 그림 캡션 | "CLIP cor081 중첩" | "CLIP axial 슬라이스 (파일명 cor081) 중첩" |
| 히트맵 설명 문장 | "selected LR class logit" | 통합 모델 기준으로 교체: CLIP은 class-embedding 유사도 logit, 3D CNN은 fc logit. 각 히트맵의 target class를 캡션에 표기 |
| 최종 앙상블 예측 | `**CN**`이 별표 그대로 출력 | 굵게 렌더링하거나 별표 제거 |
| 모델 정보: GBM | "11 구조 부피 피처 + AgeatEntry/GENDER/is\_adni" | "구조 부피 비율 8개 + AgeatEntry, GENDER, is\_adni (총 11개)" |
| 캡션 배치 | 캡션이 그림 위에 있어 1쪽 끝에 캡션만 남음 | 캡션을 그림 아래로 옮기고 `KeepTogether`로 묶음 |

- 2D CLIP 히트맵에서 뇌 밖 배경(모서리)에도 붉은 점이 보인다. 3D와 같이 배경 마스크를 적용하고, 마스크 전후의 뇌 안/밖 heat 비율을 보고한다.
- 3D 히트맵이 axis 2 끝쪽(그림 오른쪽 가장자리)에만 몰려 뇌 밖까지 번져 있다. A-1 점검을 test scan 3개 이상에서 하고, 스캔·클래스와 무관하게 항상 같은 가장자리에 몰리면 경계 패딩 artifact로 판정해 보고한다.

## B. 테스트 tmp\_path 우회

`graph_xai_extension/tests/conftest.py`에 이미 있는 `basetemp` 우회 로직을 읽고, 같은 방식으로 `clip_xai_app/tests/conftest.py`에 적용한다. 테스트 코드 자체는 고치지 않는다.

- 임시 폴더는 저장소 안의 `clip_xai_app/.pytest_tmp/`로 지정하고 `.gitignore`에 추가한다.
- `graph_xai_extension` 파일은 읽기만 하고 수정하지 않는다.

판정:

```text
pytest -q -k "not merged_full_validation"   → ERROR 0건 (현재 5건)
pytest tests/test_merged_full_validation.py → 4 passed 유지
```

새로 통과하게 된 5개 테스트(특히 `test_create_merged_pdf_report_produces_a_valid_pdf`) 중 실패가 나오면 그건 실제 결함이다. 고치기 전에 실패 내용을 보고한다.

## C. README 문구와 scan\_id 선택 범위

### C-1. README 수정

| 위치 | 현재 | 변경 |
| --- | --- | --- |
| 통합 CLIP 백본 | "사실상 확정" | "확정: `학습3.ipynb` 셀 42·46의 `MODEL_NAME = \"openai/clip-vit-base-patch16\"`" |
| 체크포인트 epoch | 지시서와 1 차이로 보고 | "체크포인트 `epoch`는 0-index(CLIP 13, 3D CNN 56). 노트북은 `epoch+1`로 출력(14, 57)" |
| CLIP 입력 단면 명칭 | coronal 슬라이스 | **axial 슬라이스**로 정정. 파일명의 `cor`는 원본 코드의 명명일 뿐이라는 설명을 덧붙인다(`axis_check.png` 근거) |
| 한계와 주의: 슬라이스 범위 | OASIS-3 단독 쪽에만 적혀 있을 수 있음 | 통합 CLIP도 axis 1의 30\~40% 구간(파일명 cor069–092)을 쓴다고 명시. 해마 복셀이 거의 없어 통합 CLIP 히트맵도 해마 소견으로 해석할 수 없다고 적는다 |
| 한계와 주의: 3D Grad-CAM | 저해상도 때문에 절반이 붉게 보임 | A 결과로 교체(축 매핑, 원인 점검 수치, block3 비교 여부) |
| 한계와 주의: pytest | ESTsoft 권한 오류 5건 | B가 통과하면 삭제 |

- 슬라이스 범위는 실제 파일로 검증한다. `slices_multi/`의 파일명에서 `cor` 번호의 최솟값과 최댓값을 세어 README에 적는다. cor069–092와 다르면 실제 값으로 쓰고 보고한다.

### C-2. 통합 탭 scan\_id 선택

- 현재 test 예시 50개만 보여주는 드롭다운을 test split 전체 573개로 늘리고, 입력해서 검색할 수 있게 한다(Gradio `Dropdown(filterable=True)` 또는 동등한 방식).
- 항목 라벨은 `scan_id · dataset_source · 실제 라벨` 형식으로 표시한다.
- train/val scan은 직접 입력하면 허용하되, 기존 경고("학습에 사용된 데이터")를 그대로 띄운다.

## 금지 사항과 완료 보고

모델 가중치, 추론 경로, 앙상블 가중치는 건드리지 않는다.

- `merged_project/`, handoff 패키지, `graph_xai_extension/`은 읽기 전용이다.
- 수정 전 Grad-CAM 산출물은 지우지 않는다. 새 결과는 `_axisfix_v1` 접미사로 저장한다.
- Grad-CAM을 보기 좋게 만드는 스무딩이나 임계값 조정을 하지 않는다.
- 기존 테스트를 통과시키려고 테스트 기준값이나 assert를 바꾸지 않는다.

완료 보고 — `WorkOrder/통합모델_XAI_후속수정_보고서_YYYYMMDD.md`:

1. A-1 원인 점검 수치표(scan 3개 × CAM 통계, alpha 양수 비율, 0이 아닌 voxel 비율, 기존 업샘플 방식)
2. 축 매핑 수정 전후 3단면 비교 이미지 경로, `slices_multi` coronal과의 방향 일치 여부
3. `block3` vs `block4` 비교 결과와 결론(원인 해결 / 해상도 한계)
4. `pytest` 두 명령의 결과
5. README 변경 전후 문장, `slices_multi` 실제 cor 범위
6. 드롭다운 동작 확인(검색, train/val 경고)
