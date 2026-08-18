# Alzheimer MRI XAI Decision Support Prototype

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

This stage provides the project structure and source-module boundaries for model loading, inference, XAI, visualization, warnings, and reports. The Gradio UI and PDF report workflow are added in later stages.
