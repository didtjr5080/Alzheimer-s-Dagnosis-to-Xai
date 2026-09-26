"""Region-based masking/CAM sensitivity analysis for the merged
(OASIS-3+ADNI) model's spatial branches (CLIP 2D, 3D CNN 3D), shared with
GBM/ensemble simple probability display. Mirrors the methodology of
`graph_xai_extension` (3x3 grid, zero/mean/blur masking, supporting/
suppressing/absolute rankings, cross-method stability, CAM-perturbation
agreement, region-adjacency graph) but is an independent implementation --
this package does not import `graph_xai_extension` and `graph_xai_extension`
is not modified by it, keeping both extensions self-contained.
"""
