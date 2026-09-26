"""Ties `region_xai`'s generic 3x3(x3) masking-sensitivity analysis to the
real merged CLIP (2D) and 3D CNN (3D) branches for one scan, mirroring
`graph_xai_extension`'s methodology (independently reimplemented, see
`region_xai/__init__.py`)."""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from .adapters.merged_clip import CLASS_NAMES as CLIP_CLASS_NAMES
from .adapters.merged_clip import MergedClipBundle, predict_array as clip_predict_array
from .adapters.merged_cnn3d import CLASS_NAMES as CNN3D_CLASS_NAMES
from .adapters.merged_cnn3d import MergedCnn3dBundle
from .region_xai.agreement import compute_cam_perturbation_agreement
from .region_xai.graph_builder import build_region_graph_2d, build_region_graph_3d
from .region_xai.perturbation import run_all_masking_methods
from .region_xai.ranking import rank_by_absolute_sensitivity, rank_supporting_regions, rank_suppressing_regions
from .region_xai.region_grid_2d import split_into_regions_2d
from .region_xai.region_grid_3d import split_into_regions_3d
from .region_xai.schemas import CamAgreementReport, RankedRegion, RegionInfo, RegionPerturbationResult, StabilityReport
from .region_xai.stability import compare_masking_methods

PRIMARY_MASKING_METHOD = "mean"


@dataclass
class BranchRegionAnalysis:
    regions: list[RegionInfo]
    results_by_method: dict[str, list[RegionPerturbationResult]]
    primary_method: str
    supporting: list[RankedRegion]
    suppressing: list[RankedRegion]
    absolute_sensitivity: list[RankedRegion]
    stability: StabilityReport
    agreement_by_method: dict[str, CamAgreementReport]
    graph: nx.Graph


def _summarize(regions, results_by_method, primary_method) -> BranchRegionAnalysis:
    primary = results_by_method[primary_method]
    return BranchRegionAnalysis(
        regions=regions, results_by_method=results_by_method, primary_method=primary_method,
        supporting=rank_supporting_regions(primary), suppressing=rank_suppressing_regions(primary),
        absolute_sensitivity=rank_by_absolute_sensitivity(primary),
        stability=compare_masking_methods(results_by_method),
        agreement_by_method={m: compute_cam_perturbation_agreement(r, m) for m, r in results_by_method.items()},
        graph=build_region_graph_2d(primary) if len(regions) == 9 else build_region_graph_3d(primary),
    )


def analyze_clip_regions(
    bundle: MergedClipBundle, image_gray: np.ndarray, cam_224: np.ndarray, predicted_class: str,
    primary_method: str = PRIMARY_MASKING_METHOD,
) -> BranchRegionAnalysis:
    regions = split_into_regions_2d(cam_224, grid=3)
    predicted_idx = CLIP_CLASS_NAMES.index(predicted_class)
    results_by_method = run_all_masking_methods(
        image_gray, regions, lambda arr: clip_predict_array(bundle, arr), CLIP_CLASS_NAMES, predicted_idx,
    )
    return _summarize(regions, results_by_method, primary_method)


def analyze_cnn3d_regions(
    bundle: MergedCnn3dBundle, volume: np.ndarray, cam_volume: np.ndarray, predicted_class: str,
    primary_method: str = PRIMARY_MASKING_METHOD,
) -> BranchRegionAnalysis:
    regions = split_into_regions_3d(cam_volume, grid=3)
    predicted_idx = CNN3D_CLASS_NAMES.index(predicted_class)
    results_by_method = run_all_masking_methods(
        volume, regions, lambda arr: bundle.predict_array(arr), CNN3D_CLASS_NAMES, predicted_idx,
    )
    return _summarize(regions, results_by_method, primary_method)
