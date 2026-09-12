# Graph XAI Extension (Standalone)

A read-only, additive extension to the existing CLIPtoXAI Alzheimer MRI
CLIP+LR classifier. It never modifies, imports destructively, or writes into
`clip_xai_app/` or `clip_lr_grad_eclip_handoff_v1_20260812_105027/`. Everything
this extension produces lives under this folder (`graph_xai_extension/`).

It divides a coronal MRI slice (and its CAM) into a 3x3 grid of **purely
spatial** regions, masks each region in turn with **all three** methods
(`zero`, `mean`, `blur`), re-runs the existing classifier, and measures how
much the probability of the model's own predicted class changes. A region
where masking *decreased* that probability is **지지 근거 (supporting
evidence)**; one where masking *increased* it is **억제 근거 (suppressing
evidence)** -- these are always reported as two separate rankings, never
merged into one "importance" list, alongside a direction-agnostic **절대
민감도 (absolute sensitivity)** ranking. The 9 regions are also shown as a
graph whose edges are spatial adjacency only.

Because `zero`/`mean`/`blur` can disagree (sometimes even flip which
direction a region points), every run also computes cross-method stability
metrics (Spearman/Kendall rank correlation, top-k overlap, sign agreement)
and CAM-vs-perturbation agreement, and reports a stability verdict that is
explicitly labeled as a project engineering threshold, not a medically
validated one. See `docs/IMPLEMENTATION_REPORT.md` for a real example where
this mattered (5 of 9 regions flipped sign between masking methods on one
real MRI slice, correctly driving the "높음" stability claim down to "낮음").

## What this is not

- Not a medical device. Research use only, not for diagnosis or treatment.
- The 3x3 regions are **not** anatomical brain regions.
- Graph edges are **not** neural, structural, or functional brain
  connections -- they only mean "these two grid cells are next to each
  other". The fruit-fly MaleCNS connectome referenced in the original work
  order was used only as methodological inspiration (nodes / edges / virtual
  ablation) for this graph; no fly data or fly/human equivalence is used
  anywhere in this code.
- A probability drop after masking is not proof that the masked region
  *causes* the prediction, and a CAM heatmap is an approximate,
  masking-method-dependent explanation, not ground-truth pathology.

## Install

```powershell
python -m pip install -r graph_xai_extension\requirements_graph_xai.txt
```

(The existing `clip_xai_app/requirements.txt` already provides everything
except `networkx` and `plotly`; if you already have that environment set up,
`pip install networkx plotly` is enough.)

## Model path

Real analysis needs the existing CLIP model, CLIP processor, and trained LR
classifier. By default this extension looks for the repo's own
`clip_lr_grad_eclip_handoff_v1_20260812_105027/` folder. To point at a
different location, set:

```powershell
$env:GRAPH_XAI_CLASSIFIER_PATH = "E:\path\to\clip_lr_classifier_new_run.joblib"
# or the handoff root directory directly:
$env:GRAPH_XAI_CLASSIFIER_PATH = "E:\path\to\handoff_root"
```

If the classifier/CLIP files cannot be found, the standalone UI disables the
"5. 분석 실행" button and shows:

> 분류 모델 파일을 찾을 수 없어 실제 Graph XAI 분석을 실행할 수 없습니다. 학습된 분류기 경로를 지정해 주세요. 테스트용 mock 모델 결과는 실제 의료영상 분석 결과가 아닙니다.

There is no UI fallback to a mock model -- mock classifiers are only used in
`tests/_mock_model.py` for automated tests, and every export always carries
an explicit `mock_mode` flag.

## Run

```powershell
cd E:\Develop\CLIPtoXAI
python graph_xai_extension\run_graph_xai.py
```

Opens at `http://127.0.0.1:7861/` (a different port from the existing app's
`7860`, so both can run at once). Optional environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `GRAPH_XAI_CLASSIFIER_PATH` | joblib file or handoff root | repo's own handoff dir |
| `GRAPH_XAI_OUTPUT_DIR` | where CSV/JSON/HTML are written | `graph_xai_extension/outputs` |
| `GRAPH_XAI_DEVICE` | `cpu` / `cuda` | auto-detect |
| `GRAPH_XAI_GRID_SIZE` | grid cells per axis | `3` |

## Interpreting the Graph XAI output

- **Node size** = the region's share of total CAM activation (`cam_ratio`).
- **Node color** = `probability_drop` for the model's original predicted
  class after masking that region -- red (+) = masking DECREASED the
  probability (supporting evidence), blue (-) = masking INCREASED it
  (suppressing evidence). Both the bar chart and the graph's colorbar carry
  this legend explicitly.
- **Edges** = 4-neighbor (rook) spatial adjacency by default; enable
  diagonal (king-move) adjacency with `build_region_graph(..., diagonal_edges=True)`.
  Edges never represent a real anatomical or neural connection -- only that
  two grid cells are next to each other.
- **`probability_drop`** = `original_class_probability - masked_original_class_probability`,
  always evaluated for the class the model originally predicted -- never the
  class the masked image happens to score highest afterward.
- A single masking method's numbers are never presented as "the" answer:
  every UI run and PDF report shows `zero`/`mean`/`blur` side by side plus
  the stability metrics between them (see `graph_xai/stability.py`).

## PDF report (prediction + model-output-sensitivity reasoning + research sign-off)

`build_graph_xai_pdf_report` takes a `report_mode` argument with three
values (default `"combined"`, selectable in the UI as "통합"/"의료진용"/
"기술 상세"):

- `"clinical_summary"`: a 2-3 page, plain-Korean-only report for medical
  staff -- key-result cards, four side-by-side images, a masked before/after
  pair, an up-to-3-row plain-language change table, and the reviewer
  sign-off. No Spearman/Kendall/Jaccard/SHA-256/git/classes_/mock_mode text.
- `"technical_full"`: the full section-by-section report described below,
  with no clinical pages.
- `"combined"`: the clinical pages, then an explicit "연구자용 기술 부록"
  divider page, then the full technical report. The reviewer sign-off
  appears exactly once, at the end of the clinical section.

Every "5. 분석 실행" run writes this PDF report (`graph_xai/pdf_report.py`)
next to the CSV/JSON/HTML. The technical section (present in
`technical_full` and `combined`) has 12 sub-sections:

1. 입력 · 모델 · 버전 · 클래스 매핑 (run id, git commit, software version,
   input content hash, and the class-order validation evidence -- never a
   hardcoded CN=0/MCI=1/AD=2 assumption)
2. 모델 예측 확률 (never called a "판단"/judgment -- always "예측"/prediction)
3. 원본 · CAM · 중첩 영상 (original, CAM-only heatmap, MRI+CAM overlay,
   grid+CAM overlay, and a top-3-CAM-region highlight -- all aligned to the
   same size, with CAM value range/colormap/alpha/interpolation recorded)
4. 지지 근거 순위 (supporting: `probability_drop > 0`)
5. 억제 근거 순위 (suppressing: `probability_drop < 0`) -- plus before/after
   masking snapshots of the #1 supporting and #1 suppressing region
6. 절대 민감도 순위 (direction-agnostic `|probability_drop|`)
7. 세 마스킹 방식 비교 (zero/mean/blur side by side + the stability verdict,
   explicitly labeled as a project threshold, not a medical one)
8. Graph XAI (static graph render with an explicit sign/size legend)
9. CAM-Perturbation 일치도 (Spearman correlations, top-3 overlap/Jaccard;
   always labeled exploratory since n=9 regions)
10. 다중 샘플 요약 (if a `BatchSummary` is passed in; otherwise says so)
11. 테스트 및 재현 정보 (reproduction command + pointer to
    `docs/IMPLEMENTATION_REPORT.md` for the full test/integrity evidence)
12. 제한사항 (medical/CAM/graph caveats; an extra bold warning is added
    automatically whenever `mock_mode=True`)

...followed by **연구 검토자 확인** (renamed from "의료진 서명란"): reviewer
name, affiliation/title, review date, and a signature line, with an explicit
disclaimer that signing only records that someone reviewed the AI output --
it does not confirm a diagnosis, a treatment decision, model approval, or
medical-device performance validation.

This is a new module, independent of `clip_xai_app/src/report.py` -- it is
not imported from there and does not modify it, though it follows a similar
reportlab/Korean-font/footer/signature pattern for visual consistency. Every
PDF page has been rendered to PNG and visually checked for clipping/overlap
at least once per structural change (see `docs/IMPLEMENTATION_REPORT.md`).

## Stability, CAM-perturbation agreement, and class-mapping validation

- `graph_xai/ranking.py` -- the three separated rankings (competition
  ranking, so ties share a rank).
- `graph_xai/masking_compare.py` -- runs `zero`/`mean`/`blur` on the same
  image and builds the region-by-region comparison table.
- `graph_xai/stability.py` -- pairwise Spearman/Kendall correlation, top-1
  overlap, top-3 Jaccard, and a sign-agreement rate across masking methods,
  with a configurable (not hardcoded) `높음`/`보통`/`낮음` verdict.
- `graph_xai/agreement.py` -- CAM-ratio vs. `probability_drop` /
  `|probability_drop|` Spearman correlation for one masking method; a region
  with a high CAM ratio but a negative drop is kept and flagged
  (`high_cam_negative_drop_regions`), never deleted or sign-flipped.
- `graph_xai/class_mapping.py` -- `resolve_class_mapping` turns a
  classifier's raw `classes_` into names using only its own reported mapping
  (never assumes CN=0/MCI=1/AD=2); `validate_class_mapping` /
  `validate_probabilities` reject binary models (unless explicitly allowed),
  unexpected class names, and non-finite or non-normalized probability
  vectors -- wired into `GraphXAIModel.__init__` / `.predict_original`, so a
  bad mapping stops the analysis instead of silently mislabeling probabilities.

## Batch evaluation

`graph_xai/batch.py::run_batch_evaluation(source, model, masking_method=...)`
accepts a list of file paths, a directory, or a CSV manifest (`image_path`,
optional `true_label` columns) and returns a `BatchSummary`: per-image and
per-**subject** class distributions (subject id inferred from the
`OASxxxxx_MR_dxxxx_corNNN.png` naming convention), supporting/suppressing
region frequency by predicted class, mean probability drop per region,
average CAM-perturbation agreement, and -- only if ground-truth labels were
supplied -- subject-level accuracy/balanced-accuracy/macro-F1/confusion
matrix. Repeated slices of one subject are averaged into a single subject
entry before any distribution or metric is computed, so they are never
double-counted as independent samples. Exposed in the standalone UI as a
"배치 평가" tab, and validated end-to-end against the 152 real slices / 6
subjects available locally (see `docs/IMPLEMENTATION_REPORT.md`).

## Package layout

```text
graph_xai_extension/
├── run_graph_xai.py          # standalone Gradio app (port 7861)
├── requirements_graph_xai.txt
├── config.example.yaml
├── graph_xai/
│   ├── schemas.py             # dataclasses shared across modules
│   ├── region_grid.py         # 3x3 spatial split + CAM/brightness stats
│   ├── masking.py             # zero / mean / blur region masking
│   ├── masking_compare.py     # runs all 3 masking methods + comparison table
│   ├── perturbation.py        # per-region probability-drop analysis
│   ├── ranking.py             # supporting / suppressing / absolute-sensitivity rankings
│   ├── stability.py           # cross-masking-method stability metrics + verdict
│   ├── agreement.py           # CAM-vs-perturbation agreement (n=9, exploratory)
│   ├── class_mapping.py       # class-order validation (never assumes CN=0/MCI=1/AD=2)
│   ├── batch.py                # multi-sample batch evaluation, subject-level dedup
│   ├── run_metadata.py        # run_id / git commit / version / input-hash metadata
│   ├── graph_builder.py       # NetworkX graph over the 9 regions
│   ├── cam_visuals.py         # CAM-only heatmap, MRI+CAM / grid+CAM overlays, top-CAM highlight
│   ├── visualization.py       # Plotly graph / bar chart / grid overlay (+ static PNG renderers)
│   ├── exporter.py            # CSV / JSON / HTML export
│   ├── pdf_report.py          # 12-section PDF: prediction + sensitivity reasoning + sign-off
│   ├── legacy_adapter.py      # read-only reuse of the existing pipeline
│   ├── model_adapter.py       # GraphXAIModel: predict_proba / compute_cam
│   └── cam_adapter.py         # ties prediction + CAM to the SAME class
├── tests/
│   ├── conftest.py            # cache/tmp isolation, makes `graph_xai` importable
│   ├── _mock_model.py         # deterministic mock classifier, tests only
│   ├── test_region_grid.py
│   ├── test_masking.py
│   ├── test_masking_compare.py
│   ├── test_perturbation.py
│   ├── test_ranking.py
│   ├── test_stability.py
│   ├── test_agreement.py
│   ├── test_class_mapping.py
│   ├── test_batch.py
│   ├── test_run_metadata.py
│   ├── test_graph_builder.py
│   ├── test_cam_visuals.py
│   ├── test_pdf_report.py
│   ├── test_integration.py
│   └── test_existing_files_unchanged.py
├── docs/
│   ├── preexisting_files_manifest.json        # current baseline (v2)
│   ├── preexisting_files_manifest_v1_20260912.json  # archived pre-v2 baseline
│   ├── graph_xai_source_baseline_pre_v2.json  # this extension's own file hashes before round 2
│   ├── IMPLEMENTATION_REPORT.md
│   ├── LEGACY_ISSUES.md
│   └── INTEGRATION_GUIDE.md
└── outputs/                   # generated CSV/JSON/HTML/PDF land here
```

## Tests

```powershell
cd E:\Develop\CLIPtoXAI\graph_xai_extension
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"   # see note below
python -m pytest tests -q
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` works around an unrelated environment
issue on this machine: the globally installed `pytest-qt` plugin crashes
with a Qt DLL load error during `pytest_configure`, before any test runs.
This is not caused by this extension; if your environment doesn't have that
problem, plain `pytest tests -q` works too.

`test_real_model_smoke_end_to_end_on_cpu` and
`test_real_model_batch_evaluation_deduplicates_six_known_subjects` only run
if the real classifier and real sample slices are found locally; otherwise
they are skipped (not failed) with a message saying why. The batch one
processes all 152 real slices and takes roughly 10 minutes on this machine's
CPU -- deselect it for a quick run with `-k "not real_model_batch_evaluation"`.
All other tests either use synthetic images with the explicit
`MockGraphXAIModel` (marked `mock_mode: true` everywhere it appears) or pure
numpy fixtures.

`test_existing_files_unchanged.py` re-hashes every pre-existing file recorded
in `docs/preexisting_files_manifest.json`, checks the 4 large untracked data
dump folders' file counts, and scans the whole repo (outside `.git`,
`merged_project*`, and this extension folder) for any new file -- including
gitignored ones like stray `__pycache__` directories, which plain
`git status` would silently miss.

## Limitations

- Region perturbation requires one forward pass per region (9 masked images
  + 1 original) plus one backward pass for the CAM; this is slower than a
  single forward pass but was kept simple rather than batched, since the
  reused `GradECLIPExplainer` computes one image at a time.
- CAM generation always explains the model's own top-1 predicted class. This
  extension does not let a user pick an arbitrary target class, by design
  (see work order section 11): explaining a class the model didn't predict
  would make the probability-drop numbers hard to interpret.
- Only 3x3 grids have named regions (`top_left` ... `bottom_right`); other
  grid sizes fall back to generic `rowR_colC` names and are not wired into
  the standalone UI's fixed dropdown.
- The stability verdict's "보통" (moderate) tier requires a minimum sign
  agreement in addition to the mean-Spearman threshold the work order's
  example gave -- Spearman on `|probability_drop|` alone can score "perfect"
  even when every region's sign flipped between masking methods, which would
  otherwise mislabel a fully sign-flipped result as merely "보통". See
  `graph_xai/stability.py`'s module docstring and
  `docs/IMPLEMENTATION_REPORT.md` for the worked example.
- CAM-perturbation agreement and all masking-comparison statistics are
  computed over 9 regions per image and are always exploratory -- never
  presented as a validated statistical result.
- See `docs/LEGACY_ISSUES.md` for pre-existing issues found in the original
  project during read-only analysis (not fixed here, per the work order).
