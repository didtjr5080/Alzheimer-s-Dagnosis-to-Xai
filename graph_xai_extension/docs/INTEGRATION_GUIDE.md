# Integration Guide (manual, not applied)

This extension was built and tested as a fully standalone app
(`run_graph_xai.py`, port 7861) precisely so that `clip_xai_app/app.py` never
had to be touched. This document is a recipe for a maintainer who later
*chooses* to add a "Graph XAI" tab to the existing Gradio app by hand. Nothing
here has been applied to the repo.

## 1. Files to reference (read-only, do not copy their contents wholesale)

| Existing file | What it's used for |
|---|---|
| `clip_xai_app/src/model_loader.py` | pattern for lazily loading + caching the pipeline |
| `clip_xai_app/app.py` | existing `gr.Blocks` structure to add a tab to |
| `graph_xai_extension/graph_xai/*.py` | new logic to import as-is |

## 2. New imports to add to `clip_xai_app/app.py`

```python
import sys
from pathlib import Path

_GRAPH_XAI_ROOT = Path(__file__).resolve().parents[1] / "graph_xai_extension"
if str(_GRAPH_XAI_ROOT) not in sys.path:
    sys.path.insert(0, str(_GRAPH_XAI_ROOT))

from graph_xai.cam_adapter import get_original_prediction_and_cam
from graph_xai.exporter import build_export_payload, export_csv, export_html, export_json
from graph_xai.graph_builder import build_region_graph
from graph_xai.masking import apply_region_mask
from graph_xai.model_adapter import GraphXAIModel, check_model_available
from graph_xai.pdf_report import build_graph_xai_pdf_report
from graph_xai.perturbation import run_region_perturbation
from graph_xai.region_grid import split_into_regions
from graph_xai.visualization import overlay_grid_on_image, plot_probability_bar_chart, plot_region_graph
```

Note this imports the **extension's own package**, not the other way around
-- `graph_xai_extension` still never needs to import anything from
`clip_xai_app`.

## 3. Where to insert a new tab

In `build_app()` (`clip_xai_app/app.py`), wrap the existing single-page
`gr.Blocks` body in a `gr.Tabs()` with the current content as "Subject
Analysis" and a new tab "Graph XAI (Beta)" containing the same components
`run_graph_xai.build_app()` defines (image input, classifier status, masking
dropdown, analyze button, overlay/table/bar/graph outputs, CSV/JSON/HTML
files). The simplest low-risk approach is actually to reuse
`run_graph_xai.build_app()`'s inner components function-by-function rather
than hand-copying the layout, since it already returns a working `gr.Blocks`.

## 4. Connecting to the already-loaded model instead of reloading it

The existing app's `analyze_subject()` already loads a `PipelineBundle` via
`clip_xai_app/src/model_loader.load_pipeline()`. To avoid loading CLIP+LR
twice in the same process, a maintainer could add a thin wrapper in
`graph_xai/model_adapter.py` (or a new function next to
`GraphXAIModel`) that accepts an already-constructed `PipelineBundle` instead
of a classifier path, e.g.:

```python
class GraphXAIModelFromBundle:
    def __init__(self, bundle, module):
        self._bundle = bundle
        self._module = module  # the inference_clip_lr module object
    # same predict_proba / compute_cam methods as GraphXAIModel, delegating
    # to self._module.predict_slices / generate_xai with self._bundle
```

This is **not implemented** in this extension because doing so would require
either importing `clip_xai_app.src` (adding a dependency this extension was
asked not to have) or passing the bundle object across the app boundary,
which only makes sense once both are the same process/app.

## 5. Known conflicts / things to check before merging

- **Port**: `run_graph_xai.py` defaults to `7861` specifically so it can run
  side-by-side with `app.py`'s `7860` during a transition period; once merged
  into one app this is moot.
- **Two `CLIPModel` loads**: see `docs/LEGACY_ISSUES.md` #3 -- if a
  maintainer wires the CAM explainer in through `GraphXAIModel` as-is, this
  extension will load its own `CLIPModel` + `CLIPProcessor` on top of
  whatever `clip_xai_app` already loaded, doubling GPU/CPU memory use. Fix by
  sharing the bundle first (see step 4).
- **`sys.modules["grad_eclip"]`**: `legacy_adapter._import_inference_clip_lr`
  inserts the handoff `code/` directory onto `sys.path` so that
  `inference_clip_lr.py`'s internal `from grad_eclip import ...` resolves.
  `clip_xai_app/src/model_loader.add_handoff_code_to_path` does the exact
  same thing for the same directory, so running both in one process is safe
  (same file, same module identity) as long as `GRAPH_XAI_CLASSIFIER_PATH`
  (if set) points at the same handoff root `clip_xai_app` is using --
  pointing them at two *different* handoff roots in the same process would
  make the second `sys.path` insertion irrelevant (first one wins) and could
  silently load the wrong `grad_eclip.py`. Keep them pointed at the same root
  until step 4's shared-bundle approach is implemented.

## 6. Tests to run after any manual integration

```powershell
cd E:\Develop\CLIPtoXAI
python -m pytest clip_xai_app\tests -q
python -m pytest graph_xai_extension\tests -q
```

Both suites should stay green independently; a manual integration should not
require changing either suite's contents, only `clip_xai_app/app.py` itself.

## 7. Rollback

Since integration is manual and additive (a new tab + new imports), rollback
is simply reverting the edits to `clip_xai_app/app.py`. No other existing
file needs to change for step 2/3 above.
