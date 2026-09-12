from __future__ import annotations

import pytest

from graph_xai.class_mapping import (
    ClassMappingError,
    resolve_class_mapping,
    validate_class_mapping,
    validate_probabilities,
)


def test_resolve_class_mapping_preserves_classifier_order_regardless_of_label_encoding():
    # classifier reports its OWN class order; CN is index 0 here but AD comes first
    classes_idx = [2, 0, 1]
    idx_to_name = {"0": "CN", "1": "MCI", "2": "AD"}
    names = resolve_class_mapping(classes_idx, idx_to_name)
    assert names == ["AD", "CN", "MCI"]

    # a probability array reported in that SAME order must map correctly
    probs = [0.7, 0.2, 0.1]
    by_name = dict(zip(names, probs))
    assert by_name["AD"] == 0.7
    assert by_name["CN"] == 0.2
    assert by_name["MCI"] == 0.1


def test_resolve_class_mapping_with_a_different_arbitrary_encoding_still_works():
    # a completely different (but internally consistent) label encoding
    classes_idx = [5, 9, 1]
    idx_to_name = {"5": "MCI", "9": "AD", "1": "CN"}
    names = resolve_class_mapping(classes_idx, idx_to_name)
    assert names == ["MCI", "AD", "CN"]


def test_resolve_class_mapping_rejects_missing_index():
    with pytest.raises(ClassMappingError):
        resolve_class_mapping([0, 1, 2], {"0": "CN", "1": "MCI"})


def test_resolve_class_mapping_rejects_duplicate_names():
    with pytest.raises(ClassMappingError):
        resolve_class_mapping([0, 1], {"0": "CN", "1": "CN"})


def test_original_predicted_class_index_and_name_agree():
    names = resolve_class_mapping([0, 1, 2], {"0": "CN", "1": "MCI", "2": "AD"})
    probs = [0.1, 0.7, 0.2]
    predicted_idx = probs.index(max(probs))
    predicted_name = names[predicted_idx]
    assert predicted_name == "MCI"
    assert names[predicted_idx] == names[probs.index(max(probs))]


def test_validate_probabilities_accepts_a_valid_vector():
    report = validate_probabilities([0.2, 0.5, 0.3], ["CN", "MCI", "AD"])
    assert report.ok
    assert report.probability_sum_ok
    assert report.n_classes == 3


def test_validate_probabilities_rejects_non_finite_values():
    with pytest.raises(ClassMappingError):
        validate_probabilities([0.5, float("nan"), 0.5], ["CN", "MCI", "AD"])
    with pytest.raises(ClassMappingError):
        validate_probabilities([0.5, float("inf"), 0.5], ["CN", "MCI", "AD"])


def test_validate_probabilities_rejects_sum_far_from_one():
    with pytest.raises(ClassMappingError):
        validate_probabilities([0.2, 0.2, 0.2], ["CN", "MCI", "AD"])


def test_validate_probabilities_allows_small_floating_point_tolerance():
    report = validate_probabilities([0.33333, 0.33333, 0.33334], ["CN", "MCI", "AD"])
    assert report.ok


def test_binary_classifier_rejected_by_default():
    with pytest.raises(ClassMappingError):
        validate_class_mapping(["CN", "AD"])


def test_binary_classifier_accepted_when_explicitly_allowed():
    report = validate_class_mapping(["CN", "AD"], allow_binary=True)
    assert report.is_binary
    assert report.n_classes == 2


def test_unexpected_class_name_stops_analysis():
    with pytest.raises(ClassMappingError):
        validate_class_mapping(["CN", "MCI", "Dementia_Other"])


def test_too_many_or_too_few_classes_rejected():
    with pytest.raises(ClassMappingError):
        validate_class_mapping(["CN"])
    with pytest.raises(ClassMappingError):
        validate_class_mapping(["CN", "MCI", "AD", "Other"])
