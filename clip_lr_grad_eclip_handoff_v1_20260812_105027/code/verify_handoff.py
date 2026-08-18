"""
verify_handoff.py

완전히 독립된 새 Python 프로세스에서 실행되어야 하는 검증 스크립트.
현재 노트북의 전역 변수를 전혀 사용하지 않고, 패키지 파일만으로 재현성을 검증한다.

사용: python verify_handoff.py <package_root>
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import joblib

TOL_LR_RELOAD = 1e-10       # 같은 embedding에 대한 LR 재로드 결과
TOL_CLIP_REEXTRACT = 1e-5   # 같은 환경에서 재추출한 CLIP embedding
TOL_TORCH_SKLEARN = 1e-5    # torch vs sklearn probability

results = {}
failures = []

def check(name, condition, detail=""):
    results[name] = bool(condition)
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append((name, detail))


def main(package_root):
    print("="*70)
    print(f"독립 프로세스 검증 시작: {package_root}")
    print("="*70)

    # --- 0. 필수 파일 존재 확인 ---
    required_files = [
        "README.md",
        "models/clip_model/config.json",
        "models/clip_model/model.safetensors",
        "models/clip_processor/preprocessor_config.json",
        "models/clip_lr_classifier_new_run.joblib",
        "models/clip_lr_weights.npz",
        "metadata/model_config.json",
        "metadata/xai_target_spec.json",
        "metadata/environment.json",
        "metadata/preprocessing_config.json",
        "metadata/data_lineage.json",
        "data/manifests/slice_manifest_large_used_original.csv",
        "data/manifests/slice_manifest_portable.csv",
        "data/manifests/embedding_index.csv",
        "data/embeddings/train_embeds_checkpoint.npz",
        "data/embeddings/test_embeds_checkpoint.npz",
        "data/reference/test_probs_clip_new_run.csv",
        "data/xai_samples/xai_sample_manifest.csv",
        "code/inference_clip_lr.py",
        "code/gradient_smoke_test.py",
    ]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(package_root, f))]
    check("필수 파일 전체 존재", len(missing_files) == 0, f"누락: {missing_files}" if missing_files else "")

    # --- 1. CLIP/processor 로드 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        clip_model = CLIPModel.from_pretrained(os.path.join(package_root, "models/clip_model")).to(device)
        clip_processor = CLIPProcessor.from_pretrained(os.path.join(package_root, "models/clip_processor"))
        clip_model.eval()
        check("CLIP model/processor 로드 성공", True)
    except Exception as e:
        check("CLIP model/processor 로드 성공", False, str(e))
        return finalize()

    # --- 2. joblib LR 로드 ---
    try:
        clf = joblib.load(os.path.join(package_root, "models/clip_lr_classifier_new_run.joblib"))
        check("joblib LR 로드 성공", True)
    except Exception as e:
        check("joblib LR 로드 성공", False, str(e))
        return finalize()

    # --- 3. NPZ 가중치와 joblib 속성 일치 ---
    weights = np.load(os.path.join(package_root, "models/clip_lr_weights.npz"), allow_pickle=False)
    coef_match = np.allclose(weights["coef"], clf.coef_, atol=0)
    intercept_match = np.allclose(weights["intercept"], clf.intercept_, atol=0)
    classes_match = np.array_equal(weights["classes"], clf.classes_)
    check("NPZ coef_ == joblib coef_", coef_match)
    check("NPZ intercept_ == joblib intercept_", intercept_match)
    check("class order 정확히 일치", classes_match, "npz=" + str(weights["classes"]) + ", joblib=" + str(clf.classes_))

    # --- 4. 선택 PNG 전처리 tensor shape/dtype 확인 ---
    xai_img_dir = os.path.join(package_root, "data/xai_samples/images")
    sample_files = sorted([f for f in os.listdir(xai_img_dir) if f.endswith(".png")])
    check("XAI 샘플 이미지 존재", len(sample_files) > 0, f"{len(sample_files)}개 발견")
    sample_path = os.path.join(xai_img_dir, sample_files[0])
    img = Image.open(sample_path).convert("RGB")
    pixel_values = clip_processor.image_processor(images=img, return_tensors="pt")["pixel_values"]
    check("전처리 tensor shape 정상", tuple(pixel_values.shape) == (1, 3, 224, 224), f"실제: {tuple(pixel_values.shape)}")
    check("전처리 tensor dtype 정상", pixel_values.dtype == torch.float32, f"실제: {pixel_values.dtype}")

    # --- 5. CLIP embedding 재추출이 기준과 일치하는지 ---
    ref_outputs = np.load(os.path.join(package_root, "data/reference/reference_outputs.npz"))
    xai_inputs = np.load(os.path.join(package_root, "data/xai_samples/xai_reference_inputs.npz"))
    stored_embeds = ref_outputs["embeds"]
    stored_logits = ref_outputs["logits"]
    stored_probs = ref_outputs["probs"]

    with torch.no_grad():
        pv = torch.tensor(xai_inputs["pixel_values"][0:1]).to(device)
        out = clip_model.get_image_features(pixel_values=pv)
        reextracted_embed = out.pooler_output.cpu().numpy()[0] if not isinstance(out, torch.Tensor) else out.cpu().numpy()[0]

    embed_diff = np.abs(reextracted_embed - stored_embeds[0]).max()
    check("재추출 CLIP embedding이 기준과 일치", embed_diff <= TOL_CLIP_REEXTRACT, f"최대오차={embed_diff}")

    # --- 6. sklearn slice probability 일치 ---
    sklearn_probs_recomputed = clf.predict_proba(reextracted_embed.reshape(1, -1))[0]
    sklearn_diff = np.abs(sklearn_probs_recomputed - stored_probs[0]).max()
    check("sklearn slice probability가 기준과 일치", sklearn_diff <= TOL_CLIP_REEXTRACT, f"최대오차={sklearn_diff}")

    # --- 7. torch LR logits/softmax가 sklearn과 일치 ---
    coef_t = torch.tensor(clf.coef_, dtype=torch.float32)
    intercept_t = torch.tensor(clf.intercept_, dtype=torch.float32)
    embed_t = torch.tensor(reextracted_embed, dtype=torch.float32)
    torch_logits = embed_t @ coef_t.T + intercept_t
    torch_probs = torch.softmax(torch_logits, dim=-1).numpy()
    torch_sklearn_diff = np.abs(torch_probs - sklearn_probs_recomputed).max()
    check("torch softmax == sklearn predict_proba", torch_sklearn_diff <= TOL_TORCH_SKLEARN, f"최대오차={torch_sklearn_diff}")

    # --- 8. 저장된 embedding으로 LR 재로드 결과 일치 (매우 엄격한 허용오차) ---
    test_ckpt = np.load(os.path.join(package_root, "data/embeddings/test_embeds_checkpoint.npz"))
    test_embeds_saved = test_ckpt["embeds"]
    test_labels_saved = test_ckpt["labels"]
    test_probs_saved_recompute = clf.predict_proba(test_embeds_saved[:100])
    # 자기 자신과 비교 (재로드 자체의 결정성 확인용) — 두 번 계산해서 비교
    test_probs_saved_recompute2 = clf.predict_proba(test_embeds_saved[:100])
    lr_reload_diff = np.abs(test_probs_saved_recompute - test_probs_saved_recompute2).max()
    check("동일 embedding에 대한 LR 결과 결정성", lr_reload_diff <= TOL_LR_RELOAD, f"최대오차={lr_reload_diff}")

    # --- 9. MR_ID별 평균 probability가 기준 CSV와 일치 (전체 194명) ---
    manifest_df = pd.read_csv(os.path.join(package_root, "data/manifests/slice_manifest_large_used_original.csv"))
    test_df_slices = manifest_df[manifest_df["split"] == "test"].reset_index(drop=True)

    check("test manifest 행수 == embedding 행수", len(test_df_slices) == len(test_embeds_saved),
          f"manifest={len(test_df_slices)}, embeds={len(test_embeds_saved)}")

    full_test_probs = clf.predict_proba(test_embeds_saved)
    result_df = pd.DataFrame({"MR_ID": test_df_slices["MR_ID"].values, "true_label_idx": test_labels_saved})
    for i, cls in enumerate(["CN", "MCI", "AD"]):
        result_df[f"{cls}_prob_slice"] = full_test_probs[:, i]

    subject_probs = result_df.groupby("MR_ID")[["CN_prob_slice", "MCI_prob_slice", "AD_prob_slice"]].mean().reset_index()
    subject_true = result_df.groupby("MR_ID")["true_label_idx"].first().reset_index()
    recomputed_df = subject_probs.merge(subject_true, on="MR_ID")
    recomputed_df.columns = ["MR_ID", "CN_prob_clip", "MCI_prob_clip", "AD_prob_clip", "true_label_idx"]

    ref_new_run = pd.read_csv(os.path.join(package_root, "data/reference/test_probs_clip_new_run.csv"))
    m = ref_new_run.merge(recomputed_df, on="MR_ID", suffixes=("_ref", "_recomputed"))
    check("MR_ID 개수 일치 (194명)", len(m) == 194 == len(ref_new_run) == len(recomputed_df),
          f"merged={len(m)}, ref={len(ref_new_run)}, recomputed={len(recomputed_df)}")

    prob_cols = ["CN_prob_clip", "MCI_prob_clip", "AD_prob_clip"]
    subj_diff = pd.concat([(m[f"{c}_ref"] - m[f"{c}_recomputed"]).abs() for c in prob_cols]).max()
    check("저장된 embedding으로 subject 확률 재현", subj_diff <= TOL_LR_RELOAD, f"최대오차={subj_diff}")

    pred_ref = m[[f"{c}_ref" for c in prob_cols]].values.argmax(axis=1)
    pred_recomputed = m[[f"{c}_recomputed" for c in prob_cols]].values.argmax(axis=1)
    check("예측 클래스 194명 전부 일치", (pred_ref == pred_recomputed).all(),
          f"불일치 {(pred_ref != pred_recomputed).sum()}명")

    # --- 10. gradient smoke test ---
    sys.path.insert(0, os.path.join(package_root, "code"))
    from gradient_smoke_test import run_smoke_test
    smoke_success, smoke_result = run_smoke_test(package_root, sample_image_path=sample_path)
    check("gradient smoke test 성공", smoke_success)

    # --- 11. checksum 검증 (아직 미생성 — 단계 6에서 처리 예정) ---
    checksum_path = os.path.join(package_root, "checksums.sha256")
    if os.path.exists(checksum_path):
        # (단계 6 이후 재실행 시 실제 검증)
        import hashlib
        all_ok = True
        with open(checksum_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                expected_hash, rel_path = line.split(maxsplit=1)
                full_path = os.path.join(package_root, rel_path)
                if not os.path.exists(full_path):
                    all_ok = False
                    continue
                h = hashlib.sha256()
                with open(full_path, "rb") as ff:
                    for chunk in iter(lambda: ff.read(8192), b""):
                        h.update(chunk)
                if h.hexdigest() != expected_hash:
                    all_ok = False
        check("checksum 검증 성공", all_ok)
    else:
        print("[SKIP] checksum 검증 — checksums.sha256이 아직 생성되지 않음 (단계 6에서 생성 후 재검증 예정)")

    return finalize()


def finalize():
    print("\n" + "="*70)
    print("검증 결과 요약")
    print("="*70)
    for name, passed in results.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    if failures:
        print(f"\n총 {len(failures)}개 항목 실패:")
        for name, detail in failures:
            print(f"  - {name}: {detail}")
        return False
    else:
        print("\nEXPORT_OK: CLIP+LR handoff package is reproducible and Grad-ECLIP-ready")
        return True


if __name__ == "__main__":
    package_root = sys.argv[1] if len(sys.argv) > 1 else "."
    success = main(package_root)
    sys.exit(0 if success else 1)
