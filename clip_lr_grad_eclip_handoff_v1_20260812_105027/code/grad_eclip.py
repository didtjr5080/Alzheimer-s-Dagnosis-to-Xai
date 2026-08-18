"""
grad_eclip.py

Grad-ECLIP 응용형(application variant) 히트맵 계산.
목표 점수는 원 논문의 이미지-텍스트 유사도가 아니라 LR class logit:
    z_c = W_c f_CLIP(x) + b_c
방법: 마지막 vision encoder layer 출력 토큰에 대한 gradient(Grad-CAM 채널 가중치)와
      CLS->patch attention weight를 결합.

주의: torch.no_grad()를 사용하지 않는다.
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class GradECLIPExplainer:
    def __init__(self, package_root, device=None):
        self.package_root = package_root
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.clip_model = CLIPModel.from_pretrained(
            os.path.join(package_root, "models", "clip_model"),
            attn_implementation="eager",  # sdpa는 output_attentions를 지원하지 않아 Grad-ECLIP에 attn weight가 필요하므로 eager 사용
        ).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(
            os.path.join(package_root, "models", "clip_processor")
        )
        self.clip_model.eval()

        weights = np.load(os.path.join(package_root, "models", "clip_lr_weights.npz"), allow_pickle=False)
        self.coef = torch.tensor(weights["coef"], dtype=torch.float32, device=self.device)      # (3, 512)
        self.intercept = torch.tensor(weights["intercept"], dtype=torch.float32, device=self.device)  # (3,)
        self.classes_idx = weights["classes"]

        with open(os.path.join(package_root, "metadata", "model_config.json"), "r", encoding="utf-8") as f:
            self.idx_to_name = json.load(f)["class_name_mapping"]

        vision_cfg = self.clip_model.config.vision_config
        self.grid_size = vision_cfg.image_size // vision_cfg.patch_size  # 14
        self.image_size = vision_cfg.image_size  # 224

    def _image_to_pixel_values(self, image):
        img = image.convert("RGB") if isinstance(image, Image.Image) else Image.open(image).convert("RGB")
        return self.clip_processor.image_processor(
            images=img, return_tensors="pt"
        )["pixel_values"].to(self.device)

    def _forward_with_hooks(self, pixel_values):
        last_layer_output_holder = {}

        def capture_hook(module, inp, out):
            # v_proj 출력 (reshape 전 Value 벡터). CLS의 최종 표현은 마지막 레이어
            # 출력 hidden state가 아니라, self-attention 내부에서 patch들의 V가
            # attention 가중치로 결합되어 만들어지므로 gradient가 흐르는 지점은 여기다.
            out.retain_grad()
            last_layer_output_holder["tokens"] = out

        last_attn_module = self.clip_model.vision_model.encoder.layers[-1].self_attn
        handle = last_attn_module.v_proj.register_forward_hook(capture_hook)

        vision_outputs = self.clip_model.vision_model(
            pixel_values=pixel_values, output_attentions=True
        )
        handle.remove()

        pooled_output = vision_outputs.pooler_output
        image_embeds = self.clip_model.visual_projection(pooled_output)

        last_attn_weights = vision_outputs.attentions[-1]  # (1, num_heads, 197, 197)
        last_layer_tokens = last_layer_output_holder["tokens"]  # (1, 197, 768) - V vectors

        return image_embeds, last_attn_weights, last_layer_tokens

    def explain(self, image_path, target_class_name=None):
        pixel_values = self._image_to_pixel_values(image_path)
        return self.explain_pixel_values(pixel_values, target_class_name=target_class_name)

    def explain_image(self, image, target_class_name=None):
        pixel_values = self._image_to_pixel_values(image)
        return self.explain_pixel_values(pixel_values, target_class_name=target_class_name)

    def explain_pixel_values(self, pixel_values, target_class_name=None):
        image_embeds, attn_weights, last_layer_tokens = self._forward_with_hooks(pixel_values)

        logits = image_embeds @ self.coef.T + self.intercept
        probs = torch.softmax(logits, dim=-1)[0]

        class_names_ordered = [self.idx_to_name[str(c)] for c in self.classes_idx]
        pred_local_idx = int(logits.argmax(dim=-1).item())
        predicted_class = class_names_ordered[pred_local_idx]

        if target_class_name is None:
            target_local_idx = pred_local_idx
        else:
            target_local_idx = class_names_ordered.index(target_class_name)

        target_logit = logits[0, target_local_idx]

        self.clip_model.zero_grad()
        target_logit.backward()

        grad = last_layer_tokens.grad
        if grad is None:
            raise RuntimeError("gradient가 계산되지 않았습니다.")

        tokens = last_layer_tokens.detach()[0]
        grads = grad.detach()[0]

        patch_tokens = tokens[1:]
        patch_grads = grads[1:]

        channel_weights = patch_grads.mean(dim=0)
        cam = (patch_tokens * channel_weights).sum(dim=-1)
        cam = F.relu(cam)

        cls_to_patch_attn = attn_weights.detach()[0, :, 0, 1:].mean(dim=0)

        # attention 성분 제거: 배경(뇌 없는 검은 영역)에 쏠리는 attention sink head가
        # 12개 중 9개로 다수였음이 진단으로 확인됨. gradient(cam) 신호는 배경 쏠림이
        # 없는 것으로 확인되어, 이 신호만 사용한다 (Grad-CAM 스타일).
        combined = cam
        if combined.max() > 0:
            combined = combined / combined.max()

        heatmap_grid = combined.reshape(self.grid_size, self.grid_size).cpu().numpy()

        heatmap_tensor = torch.tensor(heatmap_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        heatmap_224 = F.interpolate(
            heatmap_tensor, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )[0, 0].numpy()

        return {
            "heatmap_224": heatmap_224,
            "predicted_class": predicted_class,
            "target_class": class_names_ordered[target_local_idx],
            "logits": dict(zip(class_names_ordered, logits[0].detach().cpu().tolist())),
            "probs": dict(zip(class_names_ordered, probs.detach().cpu().tolist())),
        }

    def verify_embedding_path_matches(self, image_path):
        pixel_values = self._image_to_pixel_values(image_path)

        with torch.no_grad():
            official_out = self.clip_model.get_image_features(pixel_values=pixel_values)
            official = official_out.pooler_output if not isinstance(official_out, torch.Tensor) else official_out
            vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values, return_dict=True)
            manual = self.clip_model.visual_projection(vision_outputs.pooler_output)

        diff = (official - manual).abs().max().item()
        return diff


if __name__ == "__main__":
    import sys
    package_root = sys.argv[1] if len(sys.argv) > 1 else "."
    image_path = sys.argv[2]
    explainer = GradECLIPExplainer(package_root)
    diff = explainer.verify_embedding_path_matches(image_path)
    print(f"임베딩 경로 일치 검증 최대 오차: {diff}")
    result = explainer.explain(image_path)
    print("predicted_class:", result["predicted_class"])
    print("probs:", result["probs"])
