# 통합(OASIS-3+ADNI) 모델 라이브 추론 전환 작업지시서

Sep 23, 2026 · @Someone

## 0. 목적과 공통 원칙

`clip_xai_app`이 OASIS-3 단독 handoff 대신 `merged_project`의 통합 학습 모델(CLIP·GBM·3D CNN·앙상블)로 **라이브 추론과 XAI**를 수행하도록 바꾼다. 재학습은 하지 않고, 저장된 가중치를 로드해 학습 당시 test 결과를 재현하는 것이 목표다. CLIP은 캡스톤 요구사항에 따라 중심 브랜치로 둔다.

원본 정의는 모두 Colab 노트북 `학습3.ipynb`(Drive file ID `1CXu63ES0rvImpI6DGvKBsd6rDmtdsIgs`)에 있다. 코드를 옮길 때는 해당 셀을 그대로 따르고, 임의로 고치지 않는다.

| 단계 | 브랜치 | 원본 셀 | 가중치 | 재현 기준 (test n=573) |
| --- | --- | --- | --- | --- |
| 1 | 통합 CLIP + XAI | 40(슬라이스), 42(학습), 46(평가) | `checkpoints/clip_best.pt` | `clip_test_probs.csv`, Acc 60.9% / Macro-F1 0.576 |
| 2 | GBM baseline | 30 | `models/gbm_baseline.txt` | Acc 63.2% / Macro-F1 0.604 |
| 3 | 3D CNN | 확인 후 기재(3절) | `checkpoints/3dcnn_best.pt` | `3dcnn_test_probs.csv`, Acc 64.9% / Macro-F1 0.622 |
| 4 | 앙상블 | 46 | 가중합 규칙 | `final_ensemble_results.csv`, Acc 65.6% / Macro-F1 0.634 |

공통 규칙:

- 클래스 순서는 `[CN, MCI, AD] = [0, 1, 2]`로 고정하고, 로드 시점에 assert한다.
- 식별 키는 `scan_id`다. 통합 경로에 `MR_ID`를 쓰지 않는다.
- 경로는 `clip_xai_app/configs/model_registry.yaml` 한 곳에서 관리하고, `merged_project` 루트는 저장소 기준 상대경로로 쓴다.
- 각 단계는 앞 단계 검증이 통과한 뒤에만 시작한다. 실패하면 멈추고 보고한다.
- 기존 OASIS-3 단독 경로는 지우지 않고 `legacy`로 남겨, UI에서 선택할 수 있게 한다. 기본값은 통합 모델이다.

## 1단계. 통합 CLIP 어댑터와 XAI

통합 CLIP은 기존 handoff와 같은 백본·전처리를 쓰고 분류 헤드만 다르다. 새 어댑터 `clip_xai_app/src/adapters/merged_clip.py`를 만들어 헤드만 교체한다.

### 1-1. 모델 구조 (셀 42 `CLIPSimilarityClassifierFixed2` 그대로)

```python
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")  # 프리즈
ckpt = torch.load("merged_project/checkpoints/clip_best.pt", map_location=device)
class_embeds = ckpt["class_embeds"]                  # shape (3, 512)
logit_scale = clip.logit_scale.exp().item()          # 학습 때와 동일한 고정값

img = clip.get_image_features(pixel_values=pv)       # (n, 512)
img = img / img.norm(dim=-1, keepdim=True)
ce = class_embeds / class_embeds.norm(dim=-1, keepdim=True)
logits = logit_scale * img @ ce.T
probs = logits.softmax(dim=-1)
```

- 백본은 handoff에 로컬 저장된 `clip_lr_grad_eclip_handoff_v1_20260812_105027/models/clip_model/`(commit `57c21647…`)을 재사용해도 된다. 단, 로드 후 `logit_scale.exp()` 값을 출력해 보고한다.
- handoff와 달리 **이미지 임베딩을 L2 정규화한다.** handoff의 "L2 미적용" 규칙을 가져오지 않는다.
- `class_embeds.shape == (3, 512)`, `ckpt["epoch"]`(셀 46 출력 기준 epoch 14 표기)를 로드 시 출력한다.

### 1-2. 입력 슬라이스

- 통합 모델 입력은 `merged_project/slices_multi/{scan_id}_cor{idx:03d}.png`다. 셀 40이 MNI 정합 볼륨에서 1\~99 percentile 정규화, 깊이 30\~40% coronal 구간, 90도 회전, 224×224 bilinear로 만든 것이다.
- handoff `data/xai_samples/` 이미지는 전처리가 다르므로 통합 모델에 넣지 않는다.
- `unified_slice_manifest.csv`의 `slice_path`는 Drive 절대경로다. `slices_multi/` 파일명 기준으로 로컬 경로로 바꾼다. 매니페스트 파일은 수정하지 않고 메모리에서만 변환한다.
- 셀 46은 존재가 확인된 test 슬라이스 13,752장으로 평가했다. 로컬 test 슬라이스 수를 세어 다르면 보고한다.

### 1-3. subject 집계

슬라이스별 softmax 확률을 `scan_id`로 묶어 **단순 평균**한다(셀 46 `groupby("scan_id").agg(mean)`). 출력 컬럼은 `clip_CN`, `clip_MCI`, `clip_AD`로 한다.

### 1-4. XAI (Grad-ECLIP 응용형)

- 기존 `grad_eclip.py`의 hook 위치(`vision_model.encoder.layers[-1].self_attn.v_proj`)와 attention 결합 제거 방식은 그대로 쓴다.
- target만 바꾼다. 기존 LR logit `W_c f(x) + b_c` 대신 아래 값을 역전파한다.

```latex
z_c = s \cdot \frac{f(x)}{\lVert f(x) \rVert} \cdot \frac{e_c}{\lVert e_c \rVert}
```

- 학습 코드는 image 임베딩을 `torch.no_grad()`로 만들었지만, XAI 경로에서는 `no_grad`를 쓰지 않는다.
- 히트맵 이름은 "통합 CLIP class-embedding 유사도 기반 gradient heatmap"으로 표기한다.

### 1-5. 검증 (통과해야 2단계 진행)

| 항목 | 기준 |
| --- | --- |
| test scan 수 | 573 |
| 확률 vs `clip_test_probs.csv` (scan\_id별) | 최대 절대오차 < 1e-3, argmax 100% 일치 |
| Acc / Macro-F1 | 60.9% / 0.576 |
| 혼동행렬 (행=실제 CN/MCI/AD) | \[212 47 7\] / \[86 81 25\] / \[18 41 56\] |
| 소스별 | OASIS3 n=194 Acc 72.7% / F1 0.439, ADNI n=379 Acc 54.9% / F1 0.556 |
| XAI | 히트맵이 전부 0이 아님, 슬라이스 3장 이상에서 NaN 없음 |

오차 1e-3은 환경 차이(GPU→CPU, 라이브러리 버전)를 감안한 값이다. 넘으면 늘리지 말고 최대 오차와 원인 후보를 보고한다.

## 2단계. GBM baseline

GBM은 이미지가 아닌 구조 부피 비율 표 피처를 쓰며, 로컬에 필요한 파일이 모두 있다. 기존 `gbm_reference` 어댑터의 `BLOCKED` 상태를 해제하고 라이브 추론 어댑터 `merged_gbm.py`를 만든다.

- 로더: `lgb.Booster(model_file="merged_project/models/gbm_baseline.txt")`. 로드 후 `num_model_per_iteration() == 3`을 확인한다.
- 피처 원천: `merged_project/unified_final_features_gbm.csv`에서 `scan_id`로 행을 찾는다.
- 피처 순서(모델 `feature_name()`과 일치 assert):

```python
FEATURE_COLS = ['Hippocampus_ratio', 'Amygdala_ratio', 'LateralVentricle_ratio',
                'InfLatVent_ratio', 'Entorhinal_ratio', 'Thalamus_ratio',
                'Accumbens_ratio', 'BrainStem_ratio', 'AgeatEntry', 'GENDER', 'is_adni']
df['is_adni'] = (df['dataset_source'] == 'ADNI').astype(int)
probs = booster.predict(df[FEATURE_COLS])   # (n, 3) = [gbm_CN, gbm_MCI, gbm_AD]
```

- 원본 셀 30은 결측값을 따로 채우지 않았다. 같은 방식을 따르되, 결측이 있는 행 수를 보고한다.
- `gbm_stage1/2.txt`(계층형)는 앙상블에 쓰이지 않았다. 이번 작업에서는 연결하지 않는다.
- 신규 MRI는 SynthSeg 부피가 없어 GBM 피처를 만들 수 없다. `unified_final_features_gbm.csv`에 없는 scan이면 GBM 결과를 "사용 불가"로 표시한다.

검증:

| 항목 | 기준 |
| --- | --- |
| test Acc / Macro-F1 | 63.2% / 0.604 |
| 혼동행렬 | \[207 43 16\] / \[65 80 47\] / \[10 30 75\] |
| 소스별 | OASIS3 Acc 72.2% / F1 0.459, ADNI Acc 58.6% / F1 0.591 |
| 확률 vs `final_ensemble_results.csv`의 `gbm_*` | 최대 절대오차 < 1e-6 (이 CSV에 컬럼이 있는 경우) |

## 3단계. 3D CNN

3D CNN은 통합 모델 중 단일 성능이 가장 높다. 셀 39의 `Simple3DCNN_Stable` 정의를 `clip_xai_app/src/adapters/merged_cnn3d.py`로 그대로 옮기고 `3dcnn_best.pt`를 로드한다.

### 3-1. 아키텍처 (셀 39 원문 그대로)

```python
class Simple3DCNN_Stable(nn.Module):
    def __init__(self, num_classes=3, dropout=0.4, num_groups=8):
        super().__init__()
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups, out_ch), nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups, out_ch), nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2, stride=2))
        self.block1 = conv_block(1, 16)
        self.block2 = conv_block(16, 32)
        self.block3 = conv_block(32, 64)
        self.block4 = conv_block(64, 128)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.block4(self.block3(self.block2(self.block1(x))))
        x = self.global_pool(x).view(x.size(0), -1)
        return self.fc(self.dropout(x))
```

- 로드: `model.load_state_dict(ckpt["model_state"])`, `strict=True`. 누락·초과 키가 있으면 멈추고 보고한다.
- `ckpt["epoch"]`를 출력한다. 셀 39 출력 기준 표기는 epoch 57이다.
- `eval()` 모드로 추론한다(dropout 비활성).

### 3-2. 입력

- 입력은 `merged_project/mri_3d_cache/{scan_id}.npy`(float16)다. 셀 33이 MNI 볼륨에 1\~99 percentile clip, min-max 정규화, `zoom(0.5, order=1)`을 적용해 만들었다.
- 로드 시 `float32`로 바꾸고 `unsqueeze(0)`으로 채널 축을 붙여 `(1, 1, D, H, W)`로 넣는다. 추가 정규화는 하지 않는다.
- 캐시에 없는 scan은 3D CNN 결과를 "사용 불가"로 표시한다. 새 볼륨 전처리(ANTs 필요)는 이번 범위 밖이다.

### 3-3. XAI (3D Grad-CAM)

- 셀 52 방식대로 마지막 conv block(`block4`)의 활성화와 기울기로 3D 히트맵을 만든다.
- 원본 해상도로 되돌리지 말고, 캐시 볼륨 해상도에서 axial/coronal/sagittal 중앙 단면 오버레이를 만든다.
- `merged_project/gradcam_results/`의 기존 결과는 덮어쓰지 않는다. 새 결과는 `clip_xai_app/artifacts/xai/`에 저장한다.

### 3-4. 검증

| 항목 | 기준 |
| --- | --- |
| test scan 수 | 573 |
| 확률 vs `3dcnn_test_probs.csv` | 최대 절대오차 < 1e-4, argmax 100% 일치 |
| Acc / Macro-F1 | 64.9% / 0.622 |
| 혼동행렬 | \[224 39 3\] / \[86 72 34\] / \[6 33 76\] |
| 소스별 | OASIS3 Acc 76.8% / F1 0.577, ADNI Acc 58.8% / F1 0.596 |

CPU 추론은 느릴 수 있다. 전체 검증은 GPU가 없으면 batch 단위 진행률을 출력하고, 시간이 부족하면 중단 지점과 처리 건수를 보고한다.

## 4단계. 앙상블과 UI 연결

앙상블은 세 브랜치 확률의 **고정 가중합**이다(셀 46). 결합 규칙을 새로 찾지 않고 아래 값을 그대로 쓴다.

```python
W = {"gbm": 0.35, "cnn": 0.45, "clip": 0.20}
ens = W["gbm"] * P_gbm_baseline + W["cnn"] * P_cnn + W["clip"] * P_clip   # 각 (n, 3)
pred = ens.argmax(axis=1)
```

- GBM은 **baseline** 확률을 쓴다. 계층형(`gbm_hierarchical_test_probs.csv`)을 쓰면 재현되지 않는다.
- 가중치 합이 1이므로 재정규화하지 않는다. UI에 보여줄 앙상블 확률은 `ens` 값을 그대로 쓴다.
- 브랜치 하나가 "사용 불가"면 앙상블도 "사용 불가"로 표시한다. 남은 브랜치로 재가중하지 않는다.

### 4-1. 앙상블 검증

| 항목 | 기준 |
| --- | --- |
| `final_ensemble_results.csv`의 `pred`와 일치 | 573건 중 573건 |
| Acc / Macro-F1 | 65.6% / 0.634 |
| 혼동행렬 | \[219 43 4\] / \[79 77 36\] / \[9 26 80\] |
| 소스별 | OASIS3 Acc 76.3% / F1 0.549, ADNI Acc 60.2% / F1 0.611 |

지난 점검에서 나온 "단순 평균으로는 9건 불일치"는 가중치와 GBM 종류가 달라서 생긴 것이다. 위 규칙으로 다시 비교한다.

### 4-2. UI 변경 (`clip_xai_app/app.py`)

1. 모델 선택: "통합 모델(OASIS-3+ADNI)"(기본값)과 "OASIS-3 단독(legacy)".
2. 통합 모델 입력: `scan_id` 입력란 또는 드롭다운. 입력하면 `slices_multi/`에서 해당 scan의 슬라이스를 자동으로 불러온다.
3. 결과 표: CLIP, GBM, 3D CNN, 앙상블의 CN/MCI/AD 확률을 한 표로 보여주고, 최종 예측은 앙상블 기준으로 표시한다.
4. XAI: CLIP 2D 히트맵(대표 슬라이스 3장)과 3D CNN Grad-CAM 3단면을 보여준다.
5. 해당 scan의 `split`이 train/val이면 "학습에 사용된 데이터로, 성능 평가 근거가 아님" 경고를 띄운다.
6. PDF 보고서의 모델 정보 표를 통합 모델 기준으로 바꾸고, 4-1의 test 지표와 앙상블 가중치를 기재한다.

기존 PNG 업로드 경로는 legacy 모드에서만 동작하게 둔다. 통합 모드에서 업로드 이미지를 받는 기능은 전처리 재현 보장이 없어 이번 범위에서 제외한다.

## 5. 금지 사항, 문서 갱신, 완료 보고

`merged_project/`와 handoff 패키지 안의 파일은 읽기 전용이다.

- 모델 재학습, 가중치·하이퍼파라미터·앙상블 가중치·클래스 순서 변경을 하지 않는다.
- `merged_project/`의 CSV·체크포인트·`gradcam_results/`를 덮어쓰지 않는다. 새 산출물은 `clip_xai_app/artifacts/` 아래에 `_merged_v1` 접미사를 붙여 저장한다.
- test를 튜닝이나 가중치 탐색에 쓰지 않는다. 재현 검증에만 쓴다.
- 허용 오차를 늘리지 않는다. 검증 실패 시 다음 단계로 넘어가지 않고 보고한다.
- `.gitignore`에 `merged_project/`, `merged_project-20260825T072246Z-1-*/`를 추가한다. 이 폴더들을 커밋하지 않는다.
- `graph_xai_extension`은 이번 작업에서 수정하지 않는다.

문서 갱신:

- `README.md`의 "모델 현황"에서 통합 CLIP 백본 TODO를 `openai/clip-vit-base-patch16`(프리즈 + `class_embeds`)으로 확정하고, GBM baseline 행과 앙상블 가중치를 추가한다.
- 각 통합 어댑터의 레지스트리 상태를 `BLOCKED_*`에서 검증 결과에 맞는 값으로 바꾼다.
- "코드별 사용 모델" 표에서 `clip_xai_app`의 실행 경로를 통합 모델로 갱신한다.
- 변경 이력에 이번 작업 날짜와 요약을 추가한다.

완료 보고 — `WorkOrder/통합모델_라이브추론_보고서_YYYYMMDD.md`:

1. 단계별 검증 표(기준값, 실제값, 통과 여부)
2. 추가·수정한 파일 목록
3. 로드 시 출력값: `logit_scale`, `class_embeds.shape`, 각 체크포인트 `epoch`, LightGBM·torch·transformers 버전
4. 사용 불가로 처리된 scan 수(브랜치별)
5. `pytest` 결과(`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 사용)
6. UI 스크린샷 또는 PDF 1건(test split scan 기준)
7. 남은 문제와 제안
