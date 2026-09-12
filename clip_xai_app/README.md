# Alzheimer MRI XAI Research Prototype

This app wraps the verified `CLIP image embedding + LogisticRegression` handoff package without retraining or fine-tuning.

## Handoff Root

By default the app uses:

```powershell
E:\Develop\CLIPtoXAI\clip_lr_grad_eclip_handoff_v1_20260812_105027
```

Override with:

```powershell
$env:CLIP_XAI_HANDOFF_ROOT="E:\path\to\clip_lr_grad_eclip_handoff_v1_20260812_105027"
```

## Current Stage

The legacy CLIP+LR path is available for live PNG inference and CLIP ViT gradient heatmaps.

`E:\Develop\CLIPtoXAI\merged_project` has been audited and connected as verified reference CSV outputs:

- `merged_clip_reference`
- `cnn3d_reference`
- `gbm_reference`

The merged CLIP checkpoint, 3D CNN checkpoint, GBM live inference, and final ensemble are blocked for live execution until the missing model/preprocessing/ensemble contracts are supplied. See `artifacts/verification/model_contracts.md` and `artifacts/verification/missing_handoff_requirements.md`.

XAI outputs are saved with exact slice provenance under `clip_xai_app/artifacts/xai/` and indexed in `clip_xai_app/artifacts/xai_artifact_index.csv`. Each generated slice XAI bundle stores:

- exact original slice PNG copy
- raw heatmap NPY
- normalized heatmap NPY
- display heatmap PNG
- overlay PNG
- metadata JSON with `analysis_id`, true label, predicted class, target class, target score type, model/checkpoint digest, and spatial QC metrics

The artifact stem includes dataset, scan, slice index, true label, predicted class, target class, correctness, method, model, and a short `analysis_id`, so `cor084` and `cor088` no longer collide under a subject-only name.

## Validation

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest clip_xai_app\tests -q
```

The global environment currently autoloads `pytest-qt`, which fails before test collection because Qt DLLs are unavailable. Disabling third-party plugin autoloading isolates this project test suite.
