"""
gradient_smoke_test.py

Grad-ECLIP 구현 전, gradient 흐름이 정상 작동하는지만 확인하는 준비 검사 스크립트.
주의: 이 스크립트는 전체 Grad-ECLIP 히트맵을 계산하지 않는다. gradient가 마지막
vision attention 모듈까지 도달하는지만 확인한다.

핵심: torch.no_grad()를 사용하지 않는다 (gradient가 필요하므로).
"""

import os
import sys
import json
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def run_smoke_test(package_root, sample_image_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_model_dir = os.path.join(package_root, "models", "clip_model")
    clip_processor_dir = os.path.join(package_root, "models", "clip_processor")
    weights_path = os.path.join(package_root, "models", "clip_lr_weights.npz")

    clip_model = CLIPModel.from_pretrained(clip_model_dir).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_processor_dir)
    clip_model.eval()  # eval 모드이지만 gradient 계산은 여전히 가능 (no_grad를 안 쓰면 됨)

    weights = np.load(weights_path, allow_pickle=False)
    coef = torch.tensor(weights["coef"], dtype=torch.float32, device=device)       # (3, 512)
    intercept = torch.tensor(weights["intercept"], dtype=torch.float32, device=device)  # (3,)
    classes = weights["classes"]

    # 샘플 이미지 준비
    if sample_image_path is None:
        xai_dir = os.path.join(package_root, "data", "xai_samples", "images")
        candidates = [f for f in os.listdir(xai_dir) if f.endswith(".png")]
        assert candidates, "XAI 샘플 이미지가 없습니다."
        sample_image_path = os.path.join(xai_dir, candidates[0])

    img = Image.open(sample_image_path).convert("RGB")
    pixel_values = clip_processor.image_processor(images=img, return_tensors="pt")["pixel_values"].to(device)
    pixel_values.requires_grad_(False)  # 입력 자체는 grad 불필요, 모델 파라미터/activation에 grad 필요

    # 마지막 vision attention의 출력에 gradient가 도달하는지 확인하기 위한 hook
    captured = {}
    def hook_fn(module, input, output):
        # CLIPAttention의 forward 출력은 (attn_output, attn_weights) 형태일 수 있음
        out = output[0] if isinstance(output, tuple) else output
        out.retain_grad()
        captured["attn_output"] = out

    last_attn_module = clip_model.vision_model.encoder.layers[-1].self_attn
    handle = last_attn_module.register_forward_hook(hook_fn)

    # *** torch.no_grad() 사용하지 않음 — gradient 계산 위해 필수 ***
    image_outputs = clip_model.get_image_features(pixel_values=pixel_values)
    image_embeds = image_outputs.pooler_output if not isinstance(image_outputs, torch.Tensor) else image_outputs

    # LR class logit 계산: z_c = W_c f_CLIP(x) + b_c
    logits = image_embeds @ coef.T + intercept  # (1, 3)

    target_class_local_idx = int(logits.argmax(dim=-1).item())
    target_logit = logits[0, target_class_local_idx]

    clip_model.zero_grad()
    target_logit.backward()

    handle.remove()

    result = {
        "target_class_idx": int(classes[target_class_local_idx]),
        "target_logit_value": float(target_logit.item()),
        "attn_output_shape": list(captured["attn_output"].shape) if "attn_output" in captured else None,
        "attn_output_grad_is_none": captured["attn_output"].grad is None if "attn_output" in captured else True,
    }

    if result["attn_output_grad_is_none"]:
        print("❌ FAILED: 마지막 attention 출력에 gradient가 도달하지 않았습니다.")
        return False, result

    grad_norm = captured["attn_output"].grad.norm().item()
    result["attn_output_grad_norm"] = grad_norm

    if grad_norm == 0.0:
        print("❌ FAILED: gradient가 도달했지만 norm이 0입니다 (흐름이 끊겼을 가능성).")
        return False, result

    print("✅ PASSED: gradient가 마지막 vision attention 출력까지 정상적으로 도달했습니다.")
    print(json.dumps(result, indent=2))
    return True, result


if __name__ == "__main__":
    package_root = sys.argv[1] if len(sys.argv) > 1 else "."
    success, result = run_smoke_test(package_root)
    if not success:
        sys.exit(1)
