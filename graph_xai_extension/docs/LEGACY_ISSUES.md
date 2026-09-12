# Legacy Issues (found during read-only analysis, not fixed)

Per the work order's rule against modifying pre-existing files, everything
below was found while analyzing `clip_xai_app/` and
`clip_lr_grad_eclip_handoff_v1_20260812_105027/` and is reported here rather
than patched.

## 1. Classifier was pickled with a newer scikit-learn than is installed

- **File**: `clip_lr_grad_eclip_handoff_v1_20260812_105027/models/clip_lr_classifier_new_run.joblib`
- **Reproduce**: load the joblib file with `scikit-learn==1.5.2` installed
  (the version this environment has), e.g. run any of this extension's real
  model tests, or `clip_xai_app`'s own test suite.
- **Observed error/warning**:
  ```text
  sklearn\base.py:376: InconsistentVersionWarning: Trying to unpickle estimator
  LogisticRegression from version 1.6.1 when using version 1.5.2. This might
  lead to breaking code or invalid results.
  ```
- **Cause**: `metadata/model_config.json` records `"sklearn_version": "1.6.1"`,
  but the environment's installed `scikit-learn` is `1.5.2` (see
  `clip_xai_app/requirements.txt`, which pins no version at all for
  `scikit-learn`).
- **Recommended fix**: pin `scikit-learn==1.6.1` in
  `clip_xai_app/requirements.txt` (or re-export the classifier with the
  installed 1.5.2), so environment setup reproduces the exact version the
  model was fit with.
- **Impact observed so far**: predictions still match the project's own
  documented reference sample (`OAS30009_MR_d2457_cor094.png` -> CN=0.9180,
  reproduced bit-for-bit in this extension's real-model smoke test), so this
  is a latent-compatibility risk rather than a currently-wrong result.

## 2. CLIPProcessor's slow image processor is used implicitly

- **Files**: `clip_lr_grad_eclip_handoff_v1_20260812_105027/code/inference_clip_lr.py`
  and `grad_eclip.py`, both call `CLIPProcessor.from_pretrained(...)` without
  `use_fast`.
- **Reproduce**: import either module and construct `ClipLRPredictor` or
  `GradECLIPExplainer`.
- **Observed warning**:
  ```text
  Using a slow image processor as `use_fast` is unset and a slow processor was
  saved with this model. `use_fast=True` will be the default behavior in
  v4.52, even if the model was saved with a slow processor. This will result
  in minor differences in outputs.
  ```
- **Cause**: `transformers` is moving to a fast-by-default image processor;
  the saved processor config predates that default.
- **Recommended fix**: pass `use_fast=False` explicitly in both files to
  freeze current (validated) numeric behavior, or pin
  `transformers<4.52` in `clip_xai_app/requirements.txt`, or explicitly opt
  into `use_fast=True` and re-run the documented validation samples in
  `clip_xai_app/README.md` to confirm probabilities are still acceptable.
- **Impact**: none today (currently installed `transformers==4.56.2` still
  defaults to the slow processor and matches the documented reference
  probabilities), but an unpinned `pip install -r requirements.txt` on
  `transformers>=4.52` would silently start using the fast processor and
  could shift probabilities by an unverified amount.

## 3. Two independently-loaded `CLIPModel` instances with different attention implementations

- **Files**: `inference_clip_lr.ClipLRPredictor.__init__` loads
  `CLIPModel.from_pretrained(clip_model_dir)` with the library default
  attention implementation (SDPA on this environment's `torch`/`transformers`
  versions), while `grad_eclip.GradECLIPExplainer.__init__` loads a
  *second, separate* `CLIPModel.from_pretrained(..., attn_implementation="eager")`
  because Grad-ECLIP needs `output_attentions=True`, which SDPA does not
  support.
- **Observation, not a confirmed bug**: this means slice prediction
  (`predict_slices`) and CAM generation (`generate_xai`) run the vision
  encoder through two separately-instantiated models. SDPA and eager
  attention are mathematically equivalent and should produce matching
  results in exact arithmetic, but floating-point kernel differences mean
  the embeddings feeding the LR classifier are not guaranteed to be
  bit-identical between the two paths. This project's own reference
  validation (see `clip_xai_app/README.md`) only checks the
  `ClipLRPredictor` path's output against fixed reference numbers, not
  agreement between the two CLIP instances.
- **Recommended follow-up**: verify (e.g. extend
  `GradECLIPExplainer.verify_embedding_path_matches`, which already exists
  for a related check) that `ClipLRPredictor`'s and `GradECLIPExplainer`'s
  image embeddings agree to within an acceptable tolerance for the same
  input image, and consider sharing a single loaded `CLIPModel` between the
  two call sites to halve model-loading time/memory.

## 4. Handoff package's own `checksums.sha256` is known-stale (self-documented)

- **File**: `clip_lr_grad_eclip_handoff_v1_20260812_105027/checksums.sha256`
- Already acknowledged in the repo's own root `README.md`: after
  `inference_clip_lr.py` and `grad_eclip.py` had app-facing APIs added to
  them, their checksums no longer match the original handoff export. Models,
  processor, classifier, and data files were not changed. Not a new finding
  here, but re-noted because this extension's own baseline-protection tests
  (`tests/test_existing_files_unchanged.py`) intentionally hash the
  *current* state of those two files as the new baseline, not the original
  `checksums.sha256` values -- so this extension's protection tests will not
  re-flag this already-known, already-documented discrepancy.
