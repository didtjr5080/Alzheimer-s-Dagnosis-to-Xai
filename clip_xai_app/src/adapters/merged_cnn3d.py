"""Live-inference adapter for the merged (OASIS-3+ADNI) 3D CNN branch.

`Simple3DCNN_Stable` is copied verbatim from the original training notebook
cell 39 (work order section 3-1) and loaded with
`merged_project/checkpoints/3dcnn_best.pt`'s `model_state` via
`strict=True`. Input is the cached, already min-max-normalized MNI volume
(`merged_project/mri_3d_cache/{scan_id}.npy`, float16, shape (98, 116, 94));
this adapter only casts to float32 and adds batch/channel dims, with no
further normalization, per work order section 3-2.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

CLASS_NAMES = ["CN", "MCI", "AD"]


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


class MergedCnn3dBundle:
    def __init__(self, model: Simple3DCNN_Stable, epoch: int, device: torch.device, cache_dir: Path):
        self.model = model
        self.epoch = epoch
        self.device = device
        self.cache_dir = cache_dir

    def has_scan(self, scan_id: str) -> bool:
        return (self.cache_dir / f"{scan_id}.npy").exists()

    def predict_by_scan_id(self, scan_id: str):
        path = self.cache_dir / f"{scan_id}.npy"
        if not path.exists():
            return None  # "사용 불가": no cached MNI volume for this scan
        volume = np.load(path).astype(np.float32)
        probs = self.predict_array(volume)
        return {name: float(p) for name, p in zip(CLASS_NAMES, probs)}

    def predict_array(self, volume: np.ndarray) -> np.ndarray:
        """(3,) CN/MCI/AD probabilities for a raw (98, 116, 94)-shaped
        volume array (e.g. a region-perturbation-masked volume). No
        gradients tracked."""
        tensor = torch.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(tensor).softmax(dim=-1)[0].cpu().numpy()


def load_merged_cnn3d_bundle(repo_root: Path, device: str | None = None) -> MergedCnn3dBundle:
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    merged_root = repo_root / "merged_project"
    ckpt_path = merged_root / "checkpoints" / "3dcnn_best.pt"
    cache_dir = merged_root / "mri_3d_cache"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing merged 3D CNN checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device_obj, weights_only=True)
    model = Simple3DCNN_Stable(num_classes=3)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={missing} unexpected={unexpected}")
    model.to(device_obj)
    model.eval()

    return MergedCnn3dBundle(model=model, epoch=int(ckpt["epoch"]), device=device_obj, cache_dir=cache_dir)
