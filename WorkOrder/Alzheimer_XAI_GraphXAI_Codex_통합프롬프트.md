# Alzheimer MRI Graph XAI 확장 구현 — Codex 통합 프롬프트

다음 GitHub 프로젝트를 읽기 전용으로 분석하고, 기존 파일을 전혀 변경하지 않은 상태에서 독립 실행형 Graph XAI 확장 기능을 구현하세요.

저장소: `https://github.com/didtjr5080/Alzheimer-s-Dagnosis-to-Xai`

## 1. 목표

현재 저장소의 CLIP 기반 AD/CN MRI 분류 및 CAM XAI 기능을 보존하면서 별도 확장 앱을 구현합니다.

1. 기존 모델·전처리·추론 인터페이스를 읽기 전용으로 연결
2. MRI와 CAM을 3×3 공간 구역으로 분할
3. 구역별 CAM 중요도 계산
4. `zero`, `mean`, `blur` 마스킹 기반 perturbation 분석
5. 원본 예측 클래스 확률의 마스킹 전후 변화 계산
6. 9개 구역을 노드로 표현하는 Graph XAI 생성
7. 독립 UI와 CSV·JSON·HTML 내보내기 구현
8. 단위·통합·무결성 테스트와 문서 작성

초파리 MaleCNS 커넥톰은 노드·간선·가상 제거 분석이라는 **방법론적 아이디어만 참고**합니다. 초파리 데이터를 MRI 분류 입력으로 사용하거나 초파리와 사람의 신경계를 생물학적으로 동일시하지 마세요.

## 2. 최우선 원칙: 기존 파일 변경 절대 금지

작업 시작 전에 존재하던 모든 파일과 디렉터리를 읽기 전용으로 취급하세요. 이 규칙은 다른 모든 요구보다 우선합니다.

금지 사항:

- 기존 파일 수정·삭제·이동·이름 변경·덮어쓰기
- 기존 파일 자동 포맷팅, 인코딩 또는 줄바꿈 변경
- 기존 `app.py`, 소스, 테스트, README, `requirements.txt`, 설정 파일 수정
- 기존 모델·임베딩·결과 파일 교체
- 기존 폴더에 캐시·로그·결과·`__pycache__` 생성
- 사용자의 기존 Git 변경사항 복구·숨김·삭제
- 저장소 전체 포맷팅
- 사용자 승인 없는 commit, push, 배포

기존 오류를 발견해도 수정하지 말고 신규 문서에만 기록하세요.

## 3. 쓰기 허용 범위

모든 신규 파일은 충돌하지 않는 새로운 최상위 폴더 하나에만 생성하세요.

기본값: `graph_xai_extension/`

이미 존재하면 수정하지 말고 `graph_xai_extension_v2/`, `graph_xai_extension_v3/`처럼 사용되지 않은 이름을 선택하세요. 저장소 루트나 기존 폴더에는 새 파일을 만들지 마세요.

권장 구조:

```text
graph_xai_extension/
├── README.md
├── requirements_graph_xai.txt
├── run_graph_xai.py
├── config.example.yaml
├── graph_xai/
│   ├── __init__.py
│   ├── legacy_adapter.py
│   ├── model_adapter.py
│   ├── cam_adapter.py
│   ├── region_grid.py
│   ├── masking.py
│   ├── perturbation.py
│   ├── graph_builder.py
│   ├── visualization.py
│   ├── exporter.py
│   └── schemas.py
├── tests/
│   ├── test_existing_files_unchanged.py
│   ├── test_region_grid.py
│   ├── test_masking.py
│   ├── test_perturbation.py
│   ├── test_graph_builder.py
│   └── test_integration.py
└── docs/
    ├── preexisting_files_manifest.json
    ├── IMPLEMENTATION_REPORT.md
    ├── LEGACY_ISSUES.md
    └── INTEGRATION_GUIDE.md
```

## 4. 파일 보호 기준선

신규 폴더 생성 전에 다음을 수집해 운영체제 임시 디렉터리에 보관하세요.

- Git 저장소 루트와 현재 브랜치
- `git status --short`
- Git 추적 파일과 기존 미추적 파일
- `.git`을 제외한 기존 파일의 상대 경로, 크기, 수정 시각, SHA-256, 추적 여부

신규 폴더를 만든 뒤 기준선 사본을 `docs/preexisting_files_manifest.json`에 저장하세요. 작업 전부터 존재한 변경은 사용자 변경사항이므로 반드시 보존하세요.

Python 실행 시 기존 폴더에 쓰기가 발생하지 않도록 `PYTHONDONTWRITEBYTECODE=1`을 설정하고 `HF_HOME`, `TRANSFORMERS_CACHE`, `TORCH_HOME`, `MPLCONFIGDIR`, `XDG_CACHE_HOME`, 임시 경로를 신규 폴더 또는 OS 임시 디렉터리로 지정하세요. import 시 자동 다운로드·로그·결과 생성 부작용이 의심되면 실행하지 말고 정적 분석만 하세요.

먼저 다음 형식으로 보고하세요.

```text
[작업 전 보호 기준선]
- 저장소 루트:
- 현재 브랜치:
- 기존 파일 수:
- Git 추적 파일 수:
- 기존 미추적 파일 수:
- 기존 변경 파일:
- 사용할 확장 폴더:
- 기준선 생성 여부:
```

## 5. 기존 프로젝트 읽기 전용 검사

구현 전에 다음을 확인하세요.

- 실행 진입점과 UI 프레임워크
- CLIP 모델·전처리·임베딩 차원
- classifier 종류, 입력 특징 차원, 클래스 순서
- 모델 로딩 코드와 경로
- Grad-CAM·Score-CAM·ViT CAM 위치와 출력 형태
- MRI와 CAM 좌표 정렬 방식
- 추론 함수의 입력·출력
- 의존성, 테스트, 결과 저장 부작용
- `clip_lr_classifier_new_run.joblib` 존재 여부

결과 형식:

```text
[기존 프로젝트 분석]
- 실행 진입점:
- UI 프레임워크:
- CLIP 모델:
- classifier:
- 클래스 순서:
- 모델 파일 경로 및 존재 여부:
- XAI 구현 및 CAM 형태:
- 안전하게 재사용 가능한 인터페이스:
- import 부작용:
- 실제 추론 가능 여부:
- 위험 요소:
```

## 6. 모델 파일 누락 처리

`clip_lr_classifier_new_run.joblib`이 없으면 가짜 모델 파일을 만들지 마세요.

- 신규 앱에서 모델 경로 입력과 `GRAPH_XAI_CLASSIFIER_PATH` 지원
- 파일이 없으면 실제 분석 버튼 비활성화
- mock classifier는 테스트에서만 사용
- mock 결과에 `mock_mode: true` 표시
- mock 결과를 실제 MRI 결과나 성능으로 보고하지 않기
- 클래스 순서를 확인할 수 없으면 실제 분석 중단

표시 문구:

> 분류 모델 파일을 찾을 수 없어 실제 Graph XAI 분석을 실행할 수 없습니다. 학습된 분류기 경로를 지정해 주세요. 테스트용 mock 모델 결과는 실제 의료영상 분석 결과가 아닙니다.

## 7. Adapter 연동

기존 파일은 수정하지 않고 다음 순서로 연결하세요.

1. 부작용 없는 기존 공개 함수·클래스를 읽기 전용 import
2. 기존 모델 파일을 읽기 전용 로드
3. 기존 전처리와 추론 인터페이스 호출
4. import가 위험하면 신규 `legacy_adapter.py`에서 호환 인터페이스 구현
5. 실제 모델이 없으면 실제 분석 비활성화

기존 코드를 대량 복제하지 마세요. 참조한 원본 경로와 함수, 재사용 방식을 문서화하세요. 기존 앱 내부 통합은 수행하지 말고 `docs/INTEGRATION_GUIDE.md`에 수동 적용 방법만 기록하세요.

## 8. 테스트 우선 구현

신규 기능을 구현하기 전에 아직 존재하지 않는 함수의 요구사항 테스트를 먼저 작성·실행하여 예상 실패를 확인하세요. 기존 코드를 고의로 망가뜨려 실패를 만들지 마세요.

초기 실패 대상:

- 3×3 구역 생성
- 마스킹
- probability drop 계산
- 그래프 생성
- 클래스 매핑 검증
- CAM 크기 불일치 처리

실패 결과 보고 후 구현하고 동일 테스트를 다시 실행하세요.

## 9. 3×3 구역 분석

2D 관상면 MRI를 다음 공간 구역으로 나누세요.

```text
top_left, top_center, top_right
middle_left, middle_center, middle_right
bottom_left, bottom_center, bottom_right
```

해부학적 영역 명칭을 붙이지 마세요. 각 구역에서 CAM 평균·최댓값·합계·전체 대비 비율, 이미지 평균 밝기, 좌표, 면적을 계산하세요.

요구사항:

- 모든 픽셀이 정확히 한 구역에 포함
- 영상 크기가 3으로 나누어지지 않아도 누락·중복 없음
- 빈 배열, NaN, Inf, 음수, 상수 CAM, 합계 0 처리
- CAM과 영상 크기가 다르면 명시된 보간법으로 리사이즈
- 원본 배열 in-place 수정 금지

## 10. 마스킹

각 구역에 `zero`, `mean`, `blur` 방식을 지원하고 기본값은 `mean`으로 하세요.

- grayscale/RGB NumPy, PIL 및 기존 파이프라인 Tensor 규격 처리
- 원본 이미지 불변성 보장
- dtype과 값 범위 보존
- 전처리·정규화 중복 적용 금지
- 마스크 경계 검증
- 방식과 파라미터 기록

## 11. Perturbation 분석

원본과 9개 마스킹 영상에 동일 모델·전처리를 사용하고 가능하면 배치 추론하세요.

각 구역 결과:

```text
region_name
original_predicted_class
original_class_probability
masked_predicted_class
masked_original_class_probability
probability_drop
absolute_probability_change
masking_method
cam_mean
cam_ratio
```

계산식:

```text
probability_drop = original_class_probability - masked_original_class_probability
```

마스킹 후 최고 클래스 확률이 아니라 **원본에서 선택된 클래스의 전후 확률**을 비교하세요. `AD=0`, `CN=1`을 임의로 가정하지 말고 `classes_`, 설정, 학습 코드 또는 메타데이터에서 확인하세요. 확인 불가 시 실제 분석을 중단하세요.

## 12. Graph XAI

NetworkX로 9개 공간 구역을 노드로 생성하세요.

- 기본 간선: 상하좌우 공간 인접성
- 대각선 간선: 설정으로 선택, 기본 비활성화
- 노드 속성: 구역명, 행·열, CAM 평균·비율, probability drop, 절대 변화량
- 노드 크기: CAM 비율
- 노드 색상: probability drop
- hover 정보, 범례, 중요도 순위 제공

간선은 실제 신경·구조·기능 연결이 아니라 공간적 인접성일 뿐임을 UI와 문서에 표시하세요. Plotly를 우선 사용하세요.

## 13. 독립 UI

기존 `app.py`를 수정하지 말고 신규 `run_graph_xai.py`로 별도 실행 앱을 만드세요. 기존 UI가 Gradio면 가능하면 Gradio를 사용하세요.

화면 구성:

1. MRI 입력
2. classifier 경로와 로딩 상태
3. CAM 방법 또는 CAM 배열 입력
4. 마스킹 방법 선택
5. 분석 실행
6. 원본 예측·CAM·3×3 오버레이
7. 구역별 중요도 표
8. 확률 변화 막대그래프
9. Graph XAI
10. 선택 구역 마스킹 영상
11. CSV·JSON·HTML 다운로드

경고 문구:

> 이 분석은 모델이 특정 영상 구역을 얼마나 참고했는지 평가하는 설명가능 AI 결과입니다. 표시된 구역은 실제 병변 위치, 알츠하이머병의 원인 또는 임상 진단을 의미하지 않습니다. 3×3 구역은 해부학적 뇌 영역이 아닙니다.

## 14. 결과 저장 및 설정

기본 출력은 신규 확장 폴더의 `outputs/`에만 저장하세요. CSV, JSON, HTML을 지원하고 JSON에는 다음을 포함하세요.

```text
input_filename, timestamp, model_identifier, predicted_class,
original_probability, cam_method, masking_method, grid_size,
region_results, warning, mock_mode
```

환자 식별자·전체 로컬 경로·민감한 메타데이터는 저장하지 마세요.

기존 의존성 파일 대신 `requirements_graph_xai.txt`, 기존 설정 대신 `config.example.yaml`을 만드세요. `pathlib.Path`를 사용하고 Windows·Linux·Colab을 고려하세요.

환경변수:

```text
GRAPH_XAI_CLASSIFIER_PATH
GRAPH_XAI_OUTPUT_DIR
GRAPH_XAI_DEVICE
GRAPH_XAI_GRID_SIZE
```

## 15. 성능 요구사항

- CLIP과 classifier는 각각 한 번만 로드
- 원본 및 마스킹 영상 배치 추론 우선
- CPU 실행 지원, CUDA 미존재 처리
- `torch.inference_mode()` 또는 `torch.no_grad()` 사용
- 실행 장치·시간 기록
- 사용자 동의 없는 모델 다운로드 금지

## 16. 테스트 요구사항

단위 테스트:

- 3×3 전체 픽셀 커버, 중복·누락 없음
- 3으로 나누어지지 않는 크기
- CAM 정규화, NaN·Inf·크기 불일치
- 세 가지 마스킹과 원본 불변성
- probability drop 계산 및 실제 클래스 순서 사용
- 그래프 노드 9개, 기본 간선, 대각선 설정
- JSON 직렬화 및 mock 표시

통합 테스트:

- 합성 영상과 명시적 mock classifier 전체 파이프라인
- 실제 모델 존재 시에만 smoke test, 없으면 명확히 skip
- 독립 UI import smoke test
- CPU 모드, 모델 누락 오류, 결과 내보내기
- 기존 파일 무결성 검사

`test_existing_files_unchanged.py`는 작업 전 기준선과 종료 상태를 비교하여 기존 파일의 변경·삭제·이동·크기·SHA-256 변화, 기존 Git 상태 대비 새로운 diff, 기존 미추적 파일 변경, 확장 폴더 외부 신규 파일을 검사하세요. 하나라도 발견되면 전체 검증 실패입니다.

## 17. 문서화

신규 README에 설치·실행·모델 경로·Graph XAI 해석·테스트·구조·한계를 작성하세요.

`LEGACY_ISSUES.md`에는 기존 오류의 파일, 재현 조건, 오류, 원인, 권장 수정, 영향을 기록하되 실제 수정하지 마세요.

`INTEGRATION_GUIDE.md`에는 나중에 사용자가 기존 앱에 수동 통합할 파일·import·삽입 위치·연결 함수·충돌·테스트·되돌리기 방법을 기록하세요.

## 18. 과학적·의료적 제한

다음을 주장하지 마세요.

- CAM이 실제 병변을 정확히 찾았음
- 마스킹이 인과관계를 입증함
- 초파리와 사람의 신경회로가 동일함
- 초파리 커넥톰으로 알츠하이머를 진단함
- 2D MRI 한 장으로 임상 진단 가능
- mock 모델로 실제 성능 검증 완료

CAM은 근사 설명이며, 마스킹 방식에 따라 결과가 달라지고, 3×3 공간 구역은 해부학적 영역이 아니며, 본 결과는 임상 진단용이 아님을 명시하세요.

## 19. 금지 명령

다음을 실행하지 마세요.

```text
git reset
git reset --hard
git checkout --
git restore
git clean
git stash
git commit
git push
```

## 20. 작업 순서

1. Git·파일 기준선 생성 및 보고
2. 기존 프로젝트 정적 분석 및 모델 검색
3. 안전한 실행 가능성 검사
4. 구현 계획 보고
5. 충돌 없는 신규 폴더 생성
6. 실패 테스트 작성·실행·보고
7. 구역 분석, 마스킹, perturbation, 그래프 구현
8. adapter, 독립 UI, 내보내기 구현
9. 단위·통합·보호 테스트
10. 문서 작성
11. 작업 전후 SHA-256과 Git 상태 비교
12. 최종 보고

## 21. 체크리스트

- [ ] 작업 전 Git 상태와 SHA-256 기준선 생성
- [ ] 기존 변경사항 보존
- [ ] 모델·클래스·CAM 인터페이스 확인
- [ ] 모든 신규 파일을 확장 폴더에만 생성
- [ ] 실패 테스트 먼저 실행
- [ ] 3×3 분석과 세 가지 마스킹 구현
- [ ] 원본 클래스 확률 기준 perturbation 구현
- [ ] Graph XAI와 독립 UI 구현
- [ ] CSV·JSON·HTML 저장
- [ ] 단위·통합·무결성 테스트
- [ ] 기존 파일 변경·삭제 0개 확인
- [ ] 확장 폴더 외부 신규 파일 0개 확인
- [ ] 실제 모델과 mock 결과 구분
- [ ] 의료적 한계 표시

## 22. TODO

- [ ] P0: 기준선·Git 상태·모델 파일 검사
- [ ] P0: 기존 추론·CAM 인터페이스 정적 분석
- [ ] P0: 신규 확장 폴더 결정
- [ ] P1: 실패 테스트
- [ ] P1: 구역·마스킹·perturbation·Graph XAI
- [ ] P1: adapter·독립 UI·내보내기
- [ ] P2: 전체 테스트와 무결성 검증
- [ ] P2: README·통합 안내서·오류·구현 보고서

## 23. 중단 조건

다음 상황에서는 추정하거나 기존 파일을 수정하지 말고 보고하세요.

- classifier가 없거나 클래스 순서를 확인할 수 없음
- classifier 입력 차원 또는 CLIP 전처리가 불명확함
- CAM과 MRI 좌표 정렬 확인 불가
- 기존 import가 파일 생성·변경 또는 자동 다운로드 수행
- 구현에 기존 파일 수정이 필수
- 기존 파일 해시 변화 또는 확장 폴더 외부 파일 생성

모델이 없어도 순수 모듈, 합성·mock 테스트, 독립 UI 비활성 상태, 오류 처리와 문서는 계속 구현할 수 있지만 실제 MRI 검증 완료라고 보고하지 마세요. 기존 파일이 변경되면 자동 복원하지 말고 변경 파일과 해시 차이를 보고한 뒤 중단하세요.

## 24. 완료 조건과 최종 보고

종료 전 `git status --short`, `git diff --name-only`, `git diff --stat` 및 기준선 비교를 수행하세요.

완료 조건:

```text
기존 파일 변경 0개
기존 파일 삭제·이동 0개
확장 폴더 외부 신규 파일 0개
기존 기준선 대비 새로운 Git diff 0개
단위 테스트 통과
mock 통합 테스트 통과
모델 누락 처리 통과
실제 모델 테스트는 모델이 존재할 때만 통과
```

최종 보고 형식:

```text
[기존 파일 보호 결과]
- 저장소 루트/확장 폴더:
- 작업 전 기존 파일 수:
- 변경·삭제·이동된 기존 파일:
- 기존 파일의 새로운 Git diff:
- 확장 폴더 외부 신규 파일:
- 보호 규칙 통과 여부:

[기존 프로젝트 분석]
- 실행 진입점/UI/CLIP/classifier:
- 클래스 순서와 모델 파일 상태:
- XAI 구현과 재사용 인터페이스:

[신규 구현 결과]
- 신규 파일과 구현 기능:
- 미구현 기능:
- 독립 앱 실행 방법:
- 실제 모델/mock 테스트 여부:

[테스트 결과]
- 전체/통과/실패/skip:
- 실패 원인과 실행 명령:
- 앱 smoke test:

[제한사항]
- 무변경 조건으로 통합하지 못한 기능:
- 실제 검증에 필요한 사항:
- 수동 통합 사항:

[Git 상태]
- 작업 전후 변경:
- 기존 변경사항 보존 여부:
- 확장 폴더 외 변경 여부:
```

기존 파일이 하나라도 변경되었거나 테스트가 실패했다면 작업 완료라고 표시하지 말고 사실과 오류를 정확히 보고하세요.
