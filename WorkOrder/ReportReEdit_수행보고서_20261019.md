# ReportReEditWorkOrder 수행 보고서 (2026-10-19)

## 변경 파일

- `clip_xai_app/src/merged_pdf_report.py`: 단일 PDF의 제목·요약·표·장 번호와 페이지 배치 정리. 첫 페이지에 보고서 ID/일시/검사 ID/출처 및 앙상블 클래스별 점수 추가. 스캔 단위 분류와 대표 단면 XAI 대상을 구분. 검토 확인란 유지.
- `clip_xai_app/src/region_xai/clinical_wording.py`: 요약 문장의 일관성 평가 제거. 기존 `stability_verdict` 인수를 계속 허용. 전후 변화량 단위를 %p로 수정.
- `clip_xai_app/app.py`: MediOn 제목, 짧은 연구·검토 목적 안내, 모델 출력 점수 및 PDF 다운로드 라벨 정리. 기존 분석 버튼·갤러리·ZIP 연결 유지.
- `clip_xai_app/tests/test_merged_live_adapters.py`: 변경된 보고서 제목/검토란 및 출력 제외 항목 검사.
- `clip_xai_app/tests/test_ui_e2e.py`: 실제 HTTP 분석에 대표 AD 사례 추가, 다운로드 PDF의 핵심 수치 및 제외 문구 검사.
- `clip_xai_app/tests/test_report_reedit.py`: 스캔/단면 구분, 단일 PDF, 제외 절 호출 차단, 기존 함수 인수 호환성 및 %p 검사.

## 화면·PDF 변경

설명 일관성 카드·문장·경고, 안정성 평가 표, CAM-Perturbation 일치도 절, zero/mean/blur 상세 비교표의 최종 PDF 호출을 제외했다. 관련 계산 함수와 분석 결과는 유지했다. 모델 출력·CAM·대표 마스킹·지지/억제/절대 민감도 순위·Graph XAI는 계속 제공한다. 연구용 기술 부록 표제는 상세 분석 및 재현 정보로 정리했고, 실제 제공되는 PDF는 분석 실행당 하나다.

## 값 보존 검증

변경 전에 실제 GPU 추론 결과를 저장하고 변경 후 같은 스캔을 다시 추론했다. 브랜치/앙상블 출력과 CLIP·3D CNN의 모든 마스킹 방식별 `RegionPerturbationResult`가 변경 전과 정확히 같았다. PDF 생성 전후에도 수치와 CAM 배열 바이트가 같음을 확인했다.

대표 스캔: `131_S_0691_71681`, ADNI / test / 실제 AD.

| 항목 | 변경 전 | 변경 후 |
|---|---:|---:|
| 앙상블 AD 출력 점수 | 76.69% | 76.69% |
| CLIP 스캔 단위 AD | 44.13% | 44.13% |
| 3D CNN AD | 85.73% | 85.73% |
| GBM baseline AD | 83.66% | 83.66% |
| 대표 단면 CLIP XAI 대상 | MCI 50.21% | MCI 50.21% |
| CLIP 대표 마스킹 후 MCI | 58.44% | 58.44% |
| 3D CNN `block_1_2_1` 마스킹 후 AD | 6.35% | 6.35% |

3D CNN의 처리 후−처리 전 변화량은 −79.38%p다. 순위 표의 `출력 감소 (%p)`는 기존 `probability_drop = 처리 전−처리 후` 부호를 보존한다. 가중치 CLIP 0.20 / CNN 0.45 / GBM 0.35, 클래스 매핑, 추론·CAM·마스킹·안정성/일치도 계산 코드는 수정하지 않았다.

## 실제 PDF 및 시각 검토

참고 PDF는 14페이지였고, 새로 생성한 다음 세 사례는 각각 11페이지다. 모든 페이지를 PNG로 렌더링하고 페이지 모음과 주요 페이지를 직접 검토했다. 한글 깨짐·표 잘림·문장 겹침·빈 페이지·제목만 있는 페이지는 확인되지 않았다. 마지막 안내만 남는 페이지와 영상 제목/그림 분리도 조정했다.

| 스캔 | 실제 클래스 | 앙상블 분류 | 해당 출력 점수 | PDF |
|---|---|---|---:|---|
| `131_S_0691_71681` | AD | AD | 76.69% | [AD 보고서](../clip_xai_app/artifacts/verification/report_reedit/reports/merged_full_a0699e4de514.pdf) |
| `OAS30564_MR_d0000` | CN | CN | 43.29% | [CN 보고서](../clip_xai_app/artifacts/verification/report_reedit/reports/merged_full_f687b61542f8.pdf) |
| `OAS31230_MR_d0000` | MCI | CN | 53.88% | [MCI 표본 보고서](../clip_xai_app/artifacts/verification/report_reedit/reports/merged_full_9e1231d30bf6.pdf) |

MCI 표본의 CN 예측을 그대로 표시했다. 오분류 사례를 제외하거나 출력값을 교체하지 않았다.

검증 산출물은 `clip_xai_app/artifacts/verification/report_reedit/` 아래에 저장했다. `validation.json`은 최종 파일 경로·클래스별 출력을, 스캔별 `text.txt`/`page_01.png`…`page_11.png`/`contact_sheet.png`는 텍스트와 시각 검토 근거를 담는다. 이 디렉터리는 기존 정책상 Git 추적에서 제외된다.

## 테스트 결과

- Gradio HTTP 및 신규 보고서 검사: 6개 통과. CN/AD 스캔 → 분석 → 갤러리 → PDF·ZIP 다운로드, 이미지 업로드의 CLIP 단독 분석, 잘못된 스캔 ID 처리를 확인했다.
- 전체 애플리케이션 회귀: **65 passed, 4 deselected, 실패 0, skip 0** (357.89초). 제외된 4개는 기존 slow 전체 데이터셋 재현 검사다. 실행: `python -m pytest -q -c clip_xai_app/pytest.ini clip_xai_app/tests -o faulthandler_timeout=60 --basetemp=clip_xai_app/artifacts/verification/pytest_final --junitxml=clip_xai_app/artifacts/verification/report_reedit/pytest.xml`. 데이터 감사 테스트도 통과했다. 경고 9개는 외부 라이브러리 deprecation, legacy LogisticRegression 저장/실행 sklearn 버전 차이(1.6.1/1.5.2), 신규 단위 픽스처의 상수 입력 Spearman 경고다. 관련 모델 출력 검사는 통과했으며 모델·의존성 버전은 바꾸지 않았다.

## 한계 및 남은 작업

브라우저를 직접 클릭하는 수동 테스트는 수행하지 않았고, 실제 Gradio 서버를 띄워 HTTP 클라이언트로 컴포넌트 연결과 다운로드를 검증했다. 신규 2D 업로드는 기존과 같이 CLIP만 분석하며 3D CNN·GBM·앙상블은 제공하지 않는다. 573개 전체 테스트셋 재현 검사는 기존 `slow` 설정으로 일반 회귀 실행에서 제외되며, 이번 작업의 실제 GPU 검증은 위 세 스캔과 HTTP 분석 사례에 한정된다. 과거 독립 구현인 `graph_xai_extension/`은 수정하지 않았으며 해당 별도 테스트 묶음은 이번 실행 범위에 포함하지 않았다.


## 추가 배치 수정

사용자 요청에 따라 보고서 검토 확인란 뒤에 페이지 나눔을 적용했다. 상세 분석 및 재현 정보는 반드시 새 페이지에서 시작한다. 페이지 분리 회귀 검사 2개가 통과했고, CN/MCI/AD PDF를 다시 생성했다. 대표 AD 보고서의 검토란(4페이지)과 상세 분석 시작(5페이지)을 PNG로 확인했다. 전체 65개 통과 기록은 이 배치 수정 전 결과이며, 이후에는 관련 보고서 테스트 2개를 실행했다.

## 요약 문단 제거 및 실제 실행 경로 반영

추가 요청에 따라 `_clinical_page1()`에서 점수 해석 안내와 CLIP/3D CNN 요약 설명 세 문단의 생성을 제거했다. 보고서 모듈의 `build_branch_summary_sentence` import와 호출도 제거했으며, 표·영상·마스킹 전후 수치 및 원래 분석 함수는 유지한다. `app.py::analyze_merged_ui()`가 이 수정된 `build_merged_full_pdf_report()`를 직접 호출하므로 이후 정상 분석과 업로드에서도 같은 구성이 적용된다. 이전 PDF 파일을 편집하거나 정적 결과물로 대체하지 않았다. 관련 보고서 테스트 2개가 통과했고, CN/MCI/AD 세 PDF를 재생성했다. 추가로 Gradio 실제 HTTP 실행에서 세 문구의 부재 및 검토란/상세 분석의 페이지 분리를 검사하도록 테스트를 보강했다. 이미 실행 중인 앱 프로세스는 수정된 모듈을 읽도록 재시작해야 한다.

실제 앱 경로 검증 결과: `test_ui_e2e.py` **4 passed** (101.52초). CN·대표 AD 스캔 분석, CLIP 단독 업로드, 잘못된 스캔 ID 처리를 실제 HTTP로 검증했다. 다운로드된 PDF 전체 텍스트에서 제거한 세 문구가 없고, 상세 분석이 검토란보다 뒤 페이지에 있는 것을 확인했다. 이번 후속 변경 후에는 보고서 테스트 2개와 UI 테스트 4개를 실행했으며 전체 회귀 묶음은 다시 실행하지 않았다.
