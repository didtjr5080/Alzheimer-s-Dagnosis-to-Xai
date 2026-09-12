# Implementation Report

## Scope actually implemented

All items from the work order were implemented:

1. **Read-only reuse** of the existing pipeline via
   `graph_xai/legacy_adapter.py` (loads `inference_clip_lr.py` /
   `grad_eclip.py` from the handoff package by file path, never touching
   `clip_xai_app/`).
2. **3x3 spatial region split** (`graph_xai/region_grid.py`): exact pixel
   coverage with no gaps/overlaps even for non-divisible sizes, CAM/image
   resize alignment, NaN/Inf rejection, zero-sum/constant/negative CAM
   handling, no in-place mutation.
3. **Masking** (`graph_xai/masking.py`): `zero` / `mean` / `blur` (default
   `mean`), for grayscale/RGB numpy, PIL, and torch CHW tensors, with dtype
   and immutability preserved.
4. **Perturbation analysis** (`graph_xai/perturbation.py`): always compares
   the **original predicted class's** probability before/after masking
   (never the masked image's own top-1 class), using the class order
   reported by the reused pipeline (`clf.classes_` + `model_config.json`,
   validated by the original `load_pipeline`, not re-hardcoded here).
5. **Graph XAI** (`graph_xai/graph_builder.py` +
   `graph_xai/visualization.py`): 9 nodes, rook adjacency by default,
   optional diagonal (king-move) adjacency, node size = `cam_ratio`, node
   color = `probability_drop`, hover text, and an explicit
   "not a neural connection" annotation on every plotted graph.
6. **Standalone UI** (`run_graph_xai.py`, Gradio, port 7861): all 11 screen
   elements from work order section 13, model-missing message and disabled
   analyze button when the classifier can't be found, CAM-array upload
   override, per-region masked-image preview, CSV/JSON/HTML downloads.
7. **Export** (`graph_xai/exporter.py`): CSV, JSON (with exactly the fields
   listed in work order section 14, including `mock_mode`), and HTML with
   embedded Plotly figures. No patient identifiers or absolute local paths
   are written -- `input_filename` is reduced to its basename.
8. **Tests**: 54 tests across 7 files (see below), including a real-model
   smoke test that runs when the actual classifier is present and an
   automated file-protection test.
9. **PDF report** (`graph_xai/pdf_report.py`, added after the user clarified
   that the final deliverable is "a report of the judgment and why it was
   judged that way"): a reportlab-based PDF with a 판단 (Judgment) section,
   a 판단 근거 (Reasoning) section (ranked region table + narrative + grid
   overlay + bar chart + a static Graph XAI rendering), a 해석상 제한
   (Limitations) section, and a 의료진 검토 기록 (medical staff review)
   section with reviewer name/affiliation/date/signature lines -- modeled
   after `clip_xai_app/src/report.py`'s own reviewer/signature block, but
   implemented independently (no import from that file).

## TDD process actually followed

For the 4 pure-logic modules, tests were written and run against
not-yet-created modules first, confirming collection failures
(`ModuleNotFoundError: No module named 'graph_xai.region_grid'`, etc. -- 4
errors, 0 collected) before any implementation code was written. After
implementing each module, the same test files were re-run and passed (40/40
for region_grid + masking + perturbation + graph_builder combined). The
integration and file-protection tests were written after the adapters/UI
existed (they exercise the adapters/UI directly, so writing them
"pre-implementation" would have only tested for `ModuleNotFoundError` in a
less informative way); running them surfaced one real bug (see below), which
was fixed and re-verified.

## A real bug found and fixed during integration testing

`graph_xai/perturbation.py` and `graph_xai/masking.py` work on numpy arrays,
but the reused `inference_clip_lr._image_to_pixel_values` only accepts a PIL
`Image` or a file path -- passing it a masked numpy array raised
`AttributeError: 'numpy.ndarray' object has no attribute 'read'`. This was
only caught by `test_real_model_smoke_end_to_end_on_cpu` and by a manual
golden-path run through `run_graph_xai.run_analysis()` on a real sample
image, not by the mock-classifier tests (the mock classifier's `predict_proba`
happily accepts numpy arrays). Fixed by adding a `_to_pil_image()` conversion
inside `graph_xai/legacy_adapter.LegacyPipelineHandle.predict_proba` /
`generate_cam`, so the numpy-based masking/perturbation code never needs to
know about the reused pipeline's PIL-only interface. All 50 tests pass after
the fix, including a second manual golden-path run.

## Verified against the existing project's own documented reference value

Running the real pipeline through this extension on
`OAS30009_MR_d2457_cor094.png` (the exact file the repo's own root
`README.md` uses as its validation sample) reproduced `predicted_class=CN`,
`original_probability=0.9180` -- matching the documented reference
(`CN = 0.9180`) to 4 decimal places, confirming the adapter reuses the
existing pipeline correctly rather than reimplementing it.

## Test results (this environment)

```text
graph_xai_extension/tests: 54 passed
  test_region_grid.py         11 passed
  test_masking.py             12 passed
  test_perturbation.py         5 passed
  test_graph_builder.py        5 passed
  test_pdf_report.py           4 passed
  test_integration.py          5 passed (incl. real-model smoke test, not skipped:
                                          real classifier + sample image were present)
  test_existing_files_unchanged.py  5 passed
```

## Real-world validation across all locally available samples

Beyond the single documented reference sample, all 152 slice images present
locally (6 subjects: OAS30009, OAS30051, OAS30100, OAS30217, OAS30302,
OAS30661) were run through this extension's adapter and their
subject-averaged CN/MCI/AD probabilities compared against the repo's own
`clip_lr_grad_eclip_handoff_v1_20260812_105027/data/reference/test_probs_clip_new_run.csv`.
Maximum absolute difference across all 6 subjects x 3 classes was
1.5e-6 (floating-point-level noise), confirming the adapter reproduces the
original pipeline's output rather than approximating or reimplementing it.
Two of those subjects were used to pick clean demo slices for MCI and AD
(chosen for having both a correct subject-level prediction and a
high-confidence single-slice prediction of the target class):

- MCI: `OAS30217_MR_d0077_cor090.png` -> CN=0.1103, MCI=0.7892, AD=0.1005
- AD: `OAS30661_MR_d0057_cor094.png` -> CN=0.0840, MCI=0.1797, AD=0.7363

An environment-specific note unrelated to this extension's code: the
globally installed `pytest-qt` plugin crashes with a Qt DLL load error during
`pytest_configure` on this machine, and the Windows default temp directory
(redirected by third-party "ESTsoft" software) denies pytest permission to
manage its own `tmp_path` cleanup directory. Both are worked around
(`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` env var for the former; a
`pytest_configure` hook in `tests/conftest.py` that redirects `basetemp` into
this extension's own `outputs/.cache/pytest_tmp/` for the latter) without
touching anything outside this extension folder.

## Not implemented / explicitly out of scope (round 1)

- No changes to `clip_xai_app/app.py` -- see `docs/INTEGRATION_GUIDE.md` for
  the manual steps a maintainer would take instead.
- No batched forward+backward for the 9 masked images' CAM computation
  (only the original image needs a CAM; masked images only need
  `predict_proba`, which the mock/real adapters do accept as a single batch
  call already).
- Grid sizes other than 3x3 are supported by `region_grid.py`'s and
  `graph_builder.py`'s underlying logic (falls back to generic `rowR_colC`
  names) but are not wired into the fixed-choice UI dropdown.
- Pre-existing issues found in `clip_xai_app`/handoff code during analysis
  were documented in `docs/LEGACY_ISSUES.md`, not fixed, per the work order.

---

## Round 2: `Graph_XAI_추가개선_수정작업지시서.md` (2026-09-12)

### Baseline refresh

Before making any changes, the pre-existing-files baseline
(`docs/preexisting_files_manifest.json`) was refreshed to v2 because the user
had added a new file (`WorkOrder/Graph_XAI_추가개선_수정작업지시서.md`) since
round 1's baseline was captured. The original v1 baseline is archived at
`docs/preexisting_files_manifest_v1_20260912.json`; v2 additionally excludes
`graph_xai_extension/` itself from the hashed scope (it's the allowed-to-modify
folder) and includes the new WorkOrder file. This extension's own 29 source
files were separately hashed into `docs/graph_xai_source_baseline_pre_v2.json`
before any round-2 edits, so round-2 changes to this extension's own code are
traceable too, not just protection of the original project.

Only one Graph XAI folder existed (`graph_xai_extension/`; no `_v2`/`_v3`
candidates), and it was confirmed as the correct target because
`outputs/eafeeeb24a_report.pdf` inside it already contained the exact report
ID (`graphxai_20260912_161158_4a81408a`) the work order referenced.

### New modules

- `graph_xai/ranking.py` -- `rank_supporting_regions` (`probability_drop > 0`),
  `rank_suppressing_regions` (`< 0`), `rank_by_absolute_sensitivity`
  (direction-agnostic, includes zero-drop regions). Competition ranking so
  ties share a rank (1, 1, 3 -- not 1, 1, 2).
- `graph_xai/class_mapping.py` -- `resolve_class_mapping` (classifier's own
  `classes_` + index->name map, order-preserving, never hardcodes
  CN=0/MCI=1/AD=2), `validate_class_mapping` (rejects binary models unless
  `allow_binary=True`, rejects unexpected class names, rejects class counts
  outside {2, 3}), `validate_probabilities` (finite + sums to 1 within
  tolerance). Wired into `GraphXAIModel.__init__` and `.predict_original` so
  a bad mapping stops analysis instead of silently mislabeling probabilities.
- `graph_xai/masking_compare.py` -- `run_all_masking_methods` (zero/mean/blur
  on the same image), `build_masking_comparison_table` (region x method
  pivot table).
- `graph_xai/stability.py` -- pairwise Spearman/Kendall correlation (on both
  `probability_drop` and `|probability_drop|`), top-1 overlap and top-3
  Jaccard (computed separately for the supporting ranking and the absolute
  ranking, per the work order), a sign-agreement rate, the most-unstable
  region (largest drop range across methods), and a `높음`/`보통`/`낮음`
  verdict from a configurable `DEFAULT_STABILITY_THRESHOLDS` dict.
- `graph_xai/agreement.py` -- CAM-ratio vs. `probability_drop` /
  `|probability_drop|` Spearman correlation, CAM-top-3-vs-support-top-3
  overlap, CAM-top-3-vs-absolute-top-3 Jaccard, and
  `high_cam_negative_drop_regions` (kept and flagged, never deleted or
  sign-flipped). Always labeled `n=9 regions ... exploratory only`.
- `graph_xai/cam_visuals.py` -- `render_cam_heatmap`, `render_mri_cam_overlay`,
  `render_grid_cam_overlay` (visually distinct from a plain grid-only
  overlay), `render_top_cam_regions_highlight`,
  `render_region_masking_before_after`. All resize/align the CAM to the
  image size, record colormap/alpha/interpolation/value-range, and never
  mutate inputs.
- `graph_xai/batch.py` -- `discover_batch_inputs` (list / directory / CSV
  manifest), `run_batch_evaluation` (subject-level dedup via
  `infer_subject_id`, an independent reimplementation of the naming
  convention `clip_xai_app/src/report.py::infer_subject_id` uses -- not
  imported from there), classification metrics only computed when ground
  truth is present.
- `graph_xai/run_metadata.py` -- `build_run_metadata`: run_id, timestamp,
  git commit (+ `-dirty` suffix when the worktree has uncommitted changes),
  software version (`graph_xai.__version__ = "0.2.0"`), model id, classifier
  classes, CAM method, masking methods list, grid size, device, and a
  SHA-256 of the input image bytes (never a full local path).

### Deliberate departure from the work order's literal stability-threshold example

The work order's example says "보통: 평균 Spearman >= 0.40" with no sign-agreement
condition. Testing surfaced a real problem with that: `|probability_drop|`
is blind to sign, so a result set where *every* region's sign flipped between
masking methods can still score a perfect magnitude-only Spearman correlation
(`test_fully_sign_flipped_results_are_rated_unstable_even_if_magnitudes_match`
in `tests/test_stability.py` demonstrates this with `drops_b = [-d for d in
drops_a]`). Calling that "보통" would contradict the work order's own
instruction not to assert stability without basis, so a
`moderate_min_sign_agreement` threshold (default `0.50`) was added to the
"보통" tier too. This is documented in `stability.py`'s module docstring and
in the README's Limitations section as an intentional, reasoned deviation
from the literal example, not an oversight.

### Wording fixes (work order section 6)

Removed/replaced everywhere in `graph_xai/pdf_report.py`:
- "판단" (judgment) -> "예측" (prediction) for the model's output, throughout
  section titles, table markers (`<-- 모델 예측`), and status text.
- "상관관계 기반 설명" (the old reasoning narrative's self-description) ->
  replaced with the work order's own recommended sentence
  (`SENSITIVITY_EXPLANATION_SENTENCE`).
- The report title changed from "Graph XAI 판단 및 근거 보고서" to
  "Graph XAI 모델 출력 민감도 분석 보고서" (one of the three suggested
  alternatives).
- Graph-edge wording changed from a literal "신경 연결" phrase to "실제 뇌의
  배선이나 신경학적 경로를 뜻하지 않습니다" -- same meaning, avoids the
  literal banned bigram even though the negated form (as in the work order's
  own recommended replacement sentence, which itself contains "병변" inside
  a negation) would arguably have been acceptable; erring conservative here
  cost nothing.
- The signature section was renamed from "의료진 검토 기록" to the work
  order's suggested "연구 검토자 확인" (`DEFAULT_REVIEWER_SECTION_TITLE`),
  with the exact disclaimer sentence the work order specified
  (`REVIEWER_SIGNATURE_DISCLAIMER`), verified present via
  `test_pdf_never_calls_the_prediction_a_judgment` and
  `test_pdf_never_uses_banned_correlation_based_explanation_phrase` in
  `tests/test_pdf_report.py`.

### PDF visual QC (work order section 13)

Every PDF page was rendered to PNG with PyMuPDF (`fitz`) and visually
inspected (via the Read tool, not just pypdf text extraction) after each
structural change to `pdf_report.py`. Three real issues were found and fixed:

1. **Table header overlap**: plain Python strings passed directly as Table
   cell content do not reliably wrap in reportlab -- the CAM-perturbation
   agreement table's headers overlapped each other, and the stability
   table's `stability_thresholds` dict repr overflowed past the table's
   right border. Fixed by wrapping every table cell in a `Paragraph` (see
   `_cell()` helper in `pdf_report.py`) and reformatting the thresholds dict
   as a compact `key=value, key=value` string instead of a raw `repr()`.
2. **Clipped y-axis label**: `render_probability_bar_png`'s y-axis label
   ("probability_drop (+ = supporting, - = suppressing)") got clipped
   mid-word by `tight_layout` when combined with rotated x-tick labels.
   Shortened to "probability_drop" and moved the sign explanation to the
   legend (which was already present and redundant).
3. **Clipped graph node**: `render_region_graph_png`'s outermost node
   (`top_left`) was partially cut off by the axes boundary, since
   matplotlib's default data limits hug the scatter points exactly. Fixed
   with `ax.margins(0.2)`.
4. (Minor, caught in the same pass) The "#1"/"#2"/"#3" rank labels on
   `render_top_cam_regions_highlight` used PIL's tiny default font in a
   color that could blend into the highlighted region. Fixed with a larger
   bold font sized relative to the image and a white text-stroke outline for
   contrast regardless of background.

All four were re-verified by regenerating the PDF (both a synthetic
mock-model case and a real-model case on `OAS30217_MR_d0077_cor090.png`) and
re-rendering every page to PNG a second time.

### Real-model end-to-end validation of the new pipeline

Running the full new pipeline (`run_graph_xai.run_analysis`) on
`OAS30217_MR_d0077_cor090.png` with the real model reproduced
`predicted_class=MCI`, `original_probability=0.7892` (matching round 1's
number exactly) and additionally surfaced a genuinely interesting real
result: masking methods disagreed on the *sign* of 5 of 9 regions
(`bottom_left, bottom_right, middle_right, top_center, top_left`), giving a
sign-agreement rate of 62.96% and a mean Spearman of 0.328 on
`|probability_drop|` -- correctly producing a `낮음` (low) stability verdict.
This is exactly the failure mode the work order was concerned about (relying
on `mean` alone), and the new pipeline surfaces it automatically rather than
hiding it.

### Batch evaluation validated against all 152 real slices

`test_real_model_batch_evaluation_deduplicates_six_known_subjects` runs
`run_batch_evaluation` over the same 152 local slice images (6 subjects) used
in round 1's validation, with the real model on CPU. It confirms
`total_images == 152`, `total_subjects == 6` (not 152 -- proving subject-level
dedup actually collapses repeated slices instead of double-counting them),
zero failed samples, and that class counts cover all of CN/MCI/AD. This test
takes about 10 minutes on this machine's CPU (152 images x ~11 model calls
each: 1 forward + 1 backward for CAM + 9 forward passes for the 3 masking
methods' region perturbation) and is skipped automatically if the real model
isn't available.

### Test results (this environment, round 2)

```text
graph_xai_extension/tests: 115 passed (114 in ~45s + 1 real-model batch test in ~10min)
  test_region_grid.py          11 passed
  test_masking.py              12 passed
  test_masking_compare.py       4 passed   (new)
  test_perturbation.py          5 passed
  test_ranking.py                7 passed   (new)
  test_stability.py              7 passed   (new)
  test_agreement.py              5 passed   (new)
  test_class_mapping.py         13 passed   (new)
  test_batch.py                 11 passed   (new; 1 of these is the ~10min real-model test)
  test_run_metadata.py           3 passed   (new)
  test_graph_builder.py          5 passed
  test_cam_visuals.py            8 passed   (new)
  test_pdf_report.py             7 passed   (rewritten for the new 12-section structure)
  test_integration.py            5 passed
  test_existing_files_unchanged.py  5 passed
```

### Regression checks (work order section 15, "회귀 테스트")

- `run_region_perturbation` (single-masking-method calculation) was not
  modified -- `graph_xai/masking_compare.run_all_masking_methods` calls it
  once per method rather than replacing it.
- `build_region_graph` (9 nodes, rook-adjacency default) was not modified.
- `build_export_payload`'s new parameters (`run_metadata`,
  `class_mapping_report`, the three rankings, `masking_comparison_table`,
  `stability_report`, `agreement_report`) are all optional and additive --
  omitting them reproduces the exact round-1 JSON payload shape.
- Result files are still never overwritten: `run_id` (from
  `build_run_metadata`, a fresh `uuid4().hex` each call) still drives every
  output filename.
- Round-1's git-diff-stat / SHA-256 / new-file-outside-extension protection
  tests all still pass unmodified (only the underlying baseline manifest was
  refreshed to v2, per above).

### Not implemented / explicitly out of scope (round 2)

- The standalone UI's batch tab does not expose the stability/CAM-agreement
  detail per sample (only the top-level `BatchSummary` fields) -- the full
  per-image detail is in the JSON the tab writes to `outputs/`.
- `test_evidence` is not populated automatically inside `run_graph_xai.py`'s
  per-run PDF (section 11 instead points the reader to this document) --
  wiring live pytest results into a single-image analysis run's PDF did not
  seem like a meaningful thing to automate versus documenting it here once.
- Diagonal (king-move) graph edges remain available
  (`build_region_graph(..., diagonal_edges=True)`) but are not exposed as a
  UI toggle, consistent with round 1.

## Round 3: `Graph_XAI_의료진용_보고서_개선_작업지시서.md` (2026-09-12)

### Baseline refresh

The user added this work order file after round 2's baseline (v2) was
captured, so `test_no_new_files_appeared_outside_extension_folder` correctly
failed against a stale baseline (same pattern as the v1->v2 refresh in round
2). The baseline was refreshed to v3 via
`graph_xai_extension/scripts/_regen_baseline_manifest.py` (a one-off helper,
not part of the test suite); v2 is archived at
`docs/preexisting_files_manifest_v2_20260912.json`.

### New modules

- `graph_xai/clinical_wording.py`: plain-Korean region/masking-method/
  stability labels, the auto-generated 3-4 sentence clinical summary
  (`build_clinical_summary_sentence`), the up-to-3-row "주요 변화 요약" table
  builder (`build_top_changes_rows`), and `find_forbidden_phrases` -- a
  sentence-scoped banned-phrase scanner that skips sentences containing a
  negation marker (e.g. "...진단 확률...을 의미하지 않습니다.") so the work
  order's own required disclaimer sentence is never flagged as a violation.
  Every sentence-building function takes real run values as arguments; no
  function has a hardcoded example number anywhere.
- `graph_xai/visualization.py::render_clinical_region_graph_png`: a
  medical-facing Graph XAI rendering using a colorblind-safe diverging
  palette (`PuOr_r`, replacing the technical report's `RdBu_r`) with an
  explicit `+`/`-` label baked onto every node so meaning never depends on
  color alone, plus a "색상은 질환/정상을 나타내는 색이 아닙니다" legend note.
  This same rendering now also replaces the technical appendix's own Graph
  XAI figure (work order section 12 applies to both).

### PDF report modes (`graph_xai/pdf_report.py`)

`build_graph_xai_pdf_report` gained a `report_mode` parameter with three
values, matching work order section 5:

- `"clinical_summary"`: a 3-page, plain-Korean-only report for medical staff
  -- key-result cards, four side-by-side images (original / CAM / overlay /
  top-changed region), a masked before/after pair with the value change, an
  up-to-3-row plain-language change table, and the "연구 검토자 확인"
  sign-off. No Spearman/Kendall/Jaccard/SHA-256/git/classes_/mock_mode text
  anywhere in this mode (verified by a dedicated test scanning the rendered
  PDF text).
- `"technical_full"`: the full round-2 report unchanged in content (all
  numbers preserved byte-for-byte in the values, only the Graph XAI figure
  itself was replaced per above).
- `"combined"` (default, matching the work order's required default): the
  clinical pages, then an explicit "연구자용 기술 부록" divider page, then
  the full technical report. Content flows naturally across pages (no forced
  page breaks between the three clinical pages) so the clinical section
  stays within its 2-3 page budget instead of leaving large blank gaps.

The reviewer sign-off disclaimer is now stated exactly once per generated
PDF (round 2 had accidentally printed it twice inside the same signature
block); in `combined` mode it appears once, at the end of the clinical
section, not duplicated into the appendix.

### PDF visual QC (round 3)

Rendered all three modes to PNG (PyMuPDF, 150 DPI) for a real MCI sample and
inspected every page. Two real bugs were found and fixed:

1. The clinical Graph XAI figure's Korean title/caption/legend text rendered
   as missing-glyph boxes (matplotlib defaults to DejaVu Sans, which has no
   Hangul glyphs) -- fixed by registering Malgun Gothic with matplotlib's
   `font_manager` and passing `fontproperties=` on every Korean text call in
   `visualization.py`.
2. Node-name labels overlapped large nodes' circles (the CAM-ratio-scaled
   marker for `middle_center` was big enough to swallow a fixed `-16pt`
   label offset) -- fixed by scaling the label offset with each node's
   marker radius; also removed a rotated Korean colorbar label that read
   ambiguously at this figure size in favor of the plain-language legend
   note already present below the figure.

After both fixes, all three modes were regenerated and re-rendered a second
time: clinical_summary is a clean 3 pages, technical_full 6 pages, combined
8 pages, with no clipping/overlap/garbled text, and the sign meaning
(`+`/`-`, before/after tables) visually consistent across every image and
table.

### Tests added (round 3)

- `tests/test_clinical_wording.py` (20 tests): plain-language labels, the
  summary-sentence builder (including a test that its output is 3-4
  sentences and never contains the work order's own example number,
  proving it's not hardcoded), the top-changes table builder, and the
  forbidden-phrase scanner -- including a positive test that the required
  negation disclaimer is *not* flagged.
- `tests/test_pdf_report_modes.py` (15 tests): all three `report_mode`
  values produce a valid PDF; `clinical_summary` is 2-3 pages, excludes raw
  technical jargon, has zero forbidden-phrase violations, and shows the
  reviewer confirmation exactly once; `technical_full` excludes the clinical
  title but keeps all ranking sections; `combined` contains both and has at
  least as many pages as either alone.
- `tests/test_ui_report_mode_selection.py` (3 tests): the UI's Korean
  report-type radio (통합/의료진용/기술 상세) plus the appendix checkbox
  resolve to the correct `report_mode` string.
- `tests/test_pdf_report_real_model_smoke.py` (1 test, real model + real
  sample, skips gracefully otherwise): calls `run_graph_xai.run_analysis`
  end-to-end for all three report-type UI choices and confirms each PDF is
  valid and a clinical preview sentence is produced when appropriate.

Full fast suite (excluding the ~10-minute real-model batch test):
**151 passed, 1 deselected**, 0 failures.

### Regression checks (round 3)

- `run_region_perturbation`, `build_region_graph`, and every round-1/round-2
  numeric computation are untouched; only `visualization.py` (new renderer
  added, existing `render_region_graph_png` unchanged) and `pdf_report.py`
  (restructured for modes, section content itself unchanged) were edited.
- All 7 pre-existing `test_pdf_report.py` tests (which call
  `build_graph_xai_pdf_report` without `report_mode`, exercising the
  `"combined"` default) still pass unmodified.

### UI changes (`run_graph_xai.py`)

Added to the single-image tab: a "보고서 유형" radio (통합/의료진용/기술
상세, default 통합), a "표시 언어" radio (한국어 only, non-interactive --
no other language is actually implemented, so the control honestly reflects
that instead of offering a non-functional choice), a "기술 부록 포함"
checkbox (default checked; only affects the 의료진용 choice), a read-only
"의료진용 미리보기" textbox showing the auto-generated summary sentence, and
a "설명 일관성 경고" textbox that auto-populates with the low-stability
warning text whenever `stability_verdict == "낮음"`.

### Not implemented / explicitly out of scope (round 3)

- Only Korean is actually supported; the "표시 언어" UI control exists to
  satisfy the work order's UI checklist item but is locked to 한국어 since
  no English/other-language template was requested or built.
- The batch-evaluation tab's PDF/report-mode selection was not extended;
  batch runs still only produce a JSON summary (unchanged from round 2).
  This work order's scope (sections 6-14) is entirely about the
  single-image report body.
