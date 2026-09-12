"""Class-order validation: never assumes CN=0, MCI=1, AD=2 (or any other fixed
index). Every place that turns a classifier's raw class indices into
human-readable names goes through `resolve_class_mapping`, driven only by the
model's own reported class order and its own index-to-name mapping. Every
probability vector shown to a user is checked with `validate_probabilities`
first. Anything this module cannot confidently interpret raises
`ClassMappingError` rather than guessing.
"""
from __future__ import annotations

import math

from .schemas import ClassMappingReport

EXPECTED_CLASS_SET = frozenset({"CN", "MCI", "AD"})


class ClassMappingError(ValueError):
    """Raised when a classifier's class order/labels cannot be trusted."""


def resolve_class_mapping(classes_idx, idx_to_name: dict) -> list[str]:
    """Turn raw classifier class indices (e.g. sklearn's `clf.classes_`, in
    whatever order the classifier itself reports) into class names using ONLY
    the model-provided index->name mapping. The returned list preserves the
    classifier's own order, so a probability array's i-th entry always lines
    up with the i-th name here -- regardless of which integer the training
    code happened to assign to which class.
    """
    names = []
    for idx in classes_idx:
        key = str(idx)
        if key not in idx_to_name:
            raise ClassMappingError(f"No name mapping for classifier class index {idx!r} (mapping={idx_to_name})")
        names.append(idx_to_name[key])
    if len(set(names)) != len(names):
        raise ClassMappingError(f"Duplicate class names after mapping: {names}")
    return names


def validate_class_mapping(class_names: list[str], *, allow_binary: bool = False) -> ClassMappingReport:
    """Checks `class_names` against the expected {CN, MCI, AD} set
    (order-independent -- this never checks or assumes *which* index each
    name has). Raises ClassMappingError for anything unexpected."""
    n = len(class_names)
    if n < 2:
        raise ClassMappingError(f"Need at least 2 classes, got {n}: {class_names}")

    is_binary = (n == 2)
    if is_binary and not allow_binary:
        raise ClassMappingError(
            f"Binary classifier detected ({class_names}) but this analysis was not told to "
            "accept binary models (allow_binary=True). The 3x3 region graph and its narrative "
            "text are written for the 3-class CN/MCI/AD setup; refusing to guess how a 2-class "
            "result should be described instead of silently proceeding."
        )

    unexpected = [name for name in class_names if name not in EXPECTED_CLASS_SET]
    if unexpected:
        raise ClassMappingError(
            f"Unexpected class name(s) not in {sorted(EXPECTED_CLASS_SET)}: {unexpected}. "
            "Refusing to guess a mapping for unknown classes -- stopping the analysis."
        )

    if not is_binary and n != 3:
        raise ClassMappingError(f"Expected 2 (binary) or 3 (CN/MCI/AD) classes, got {n}: {class_names}")

    return ClassMappingReport(
        classes_idx=list(range(n)),
        class_names_ordered=list(class_names),
        n_classes=n,
        is_binary=is_binary,
        probability_array_order_matches_class_names=True,
        probability_sum_ok=True,
        probability_sum_value=1.0,
        unexpected_classes=unexpected,
        ok=True,
        detail="class_names verified against the model's own reported classes; no fixed index assumed",
    )


def validate_probabilities(probs, class_names: list[str], tolerance: float = 1e-3) -> ClassMappingReport:
    """Checks one probability vector (length == len(class_names)) is finite
    and sums to 1 within `tolerance`, and that class_names itself passes
    `validate_class_mapping`. Raises ClassMappingError otherwise."""
    probs = [float(p) for p in probs]
    if len(probs) != len(class_names):
        raise ClassMappingError(f"probs length {len(probs)} != class_names length {len(class_names)}")
    if not all(math.isfinite(p) for p in probs):
        raise ClassMappingError(f"Non-finite probability value(s): {probs}")

    total = float(sum(probs))
    if abs(total - 1.0) > tolerance:
        raise ClassMappingError(f"Probabilities must sum to 1 within {tolerance}, got {total}")

    mapping = validate_class_mapping(class_names, allow_binary=(len(class_names) == 2))
    return ClassMappingReport(
        classes_idx=mapping.classes_idx,
        class_names_ordered=mapping.class_names_ordered,
        n_classes=mapping.n_classes,
        is_binary=mapping.is_binary,
        probability_array_order_matches_class_names=True,
        probability_sum_ok=True,
        probability_sum_value=total,
        unexpected_classes=mapping.unexpected_classes,
        ok=True,
        detail=f"probabilities sum to {total:.6f} (tolerance {tolerance})",
    )
