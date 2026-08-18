# Validation Report

```
======================================================================
독립 프로세스 검증 시작: /content/drive/MyDrive/OASIS3_project/model_handoff/clip_lr_grad_eclip_handoff_v1_20260812_105027
======================================================================
[PASS] 필수 파일 전체 존재
[PASS] CLIP model/processor 로드 성공
[PASS] joblib LR 로드 성공
[PASS] NPZ coef_ == joblib coef_
[PASS] NPZ intercept_ == joblib intercept_
[PASS] class order 정확히 일치 — npz=[0 1 2], joblib=[0 1 2]
[PASS] XAI 샘플 이미지 존재 — 152개 발견
[PASS] 전처리 tensor shape 정상 — 실제: (1, 3, 224, 224)
[PASS] 전처리 tensor dtype 정상 — 실제: torch.float32
[PASS] 재추출 CLIP embedding이 기준과 일치 — 최대오차=2.6226043701171875e-06
[PASS] sklearn slice probability가 기준과 일치 — 최대오차=6.620863168826929e-08
[PASS] torch softmax == sklearn predict_proba — 최대오차=8.619260638553783e-08
[PASS] 동일 embedding에 대한 LR 결과 결정성 — 최대오차=0.0
[PASS] test manifest 행수 == embedding 행수 — manifest=4883, embeds=4883
[PASS] MR_ID 개수 일치 (194명) — merged=194, ref=194, recomputed=194
[PASS] 저장된 embedding으로 subject 확률 재현 — 최대오차=1.1102230246251565e-16
[PASS] 예측 클래스 194명 전부 일치 — 불일치 0명
✅ PASSED: gradient가 마지막 vision attention 출력까지 정상적으로 도달했습니다.
{
  "target_class_idx": 0,
  "target_logit_value": 3.665052890777588,
  "attn_output_shape": [
    1,
    197,
    768
  ],
  "attn_output_grad_is_none": false,
  "attn_output_grad_norm": 20.963199615478516
}
[PASS] gradient smoke test 성공
[SKIP] checksum 검증 — checksums.sha256이 아직 생성되지 않음 (단계 6에서 생성 후 재검증 예정)

======================================================================
검증 결과 요약
======================================================================
  [PASS] 필수 파일 전체 존재
  [PASS] CLIP model/processor 로드 성공
  [PASS] joblib LR 로드 성공
  [PASS] NPZ coef_ == joblib coef_
  [PASS] NPZ intercept_ == joblib intercept_
  [PASS] class order 정확히 일치
  [PASS] XAI 샘플 이미지 존재
  [PASS] 전처리 tensor shape 정상
  [PASS] 전처리 tensor dtype 정상
  [PASS] 재추출 CLIP embedding이 기준과 일치
  [PASS] sklearn slice probability가 기준과 일치
  [PASS] torch softmax == sklearn predict_proba
  [PASS] 동일 embedding에 대한 LR 결과 결정성
  [PASS] test manifest 행수 == embedding 행수
  [PASS] MR_ID 개수 일치 (194명)
  [PASS] 저장된 embedding으로 subject 확률 재현
  [PASS] 예측 클래스 194명 전부 일치
  [PASS] gradient smoke test 성공

EXPORT_OK: CLIP+LR handoff package is reproducible and Grad-ECLIP-ready

```

## STDERR

```

Loading weights:   0%|          | 0/398 [00:00<?, ?it/s]
Loading weights:  76%|███████▌  | 302/398 [00:00<00:00, 2989.50it/s]
Loading weights: 100%|██████████| 398/398 [00:00<00:00, 2991.07it/s]

Loading weights:   0%|          | 0/398 [00:00<?, ?it/s]
Loading weights: 100%|██████████| 398/398 [00:00<00:00, 4337.94it/s]

```
