import numpy as np
import pytest
from journal2026 import (
    JOURNAL2026_D1_TINY, JOURNAL2026_D1_SMALL, JOURNAL2026_D1_MEDIUM,
    JOURNAL2026_D1_REGULAR, JOURNAL2026_D1_LARGE, JOURNAL2026_D1,
    JOURNAL2026_D1_PREVIEW,
    JOURNAL2026_D2_REGULAR, JOURNAL2026_D2_LARGE, JOURNAL2026_D2,
    JOURNAL2026_D2_PREVIEW,
    JOURNAL2026_D3_REGULAR, JOURNAL2026_D3, JOURNAL2026_D3_PREVIEW,
    JOURNAL2026_ESTIMATORS, JOURNAL2026_EST_NAMES,
    TIMING_ESTIMATORS, TIMING_EST_NAMES,
    JOURNAL2026_TRAIN_SIZES,
)


def test_d1_counts():
    assert len(JOURNAL2026_D1_TINY) == 4
    assert len(JOURNAL2026_D1_SMALL) == 6
    assert len(JOURNAL2026_D1_MEDIUM) == 7
    assert len(JOURNAL2026_D1_REGULAR) == 17
    assert len(JOURNAL2026_D1_LARGE) == 4
    assert len(JOURNAL2026_D1) == 21
    assert len(JOURNAL2026_D1_PREVIEW) == 9


def test_d1_regular_is_union_of_tiny_small_and_medium():
    assert JOURNAL2026_D1_REGULAR == JOURNAL2026_D1_TINY + JOURNAL2026_D1_SMALL + JOURNAL2026_D1_MEDIUM


def test_d1_tiny_datasets():
    datasets = {p.dataset for p in JOURNAL2026_D1_TINY}
    assert datasets == {'autompg', 'diabetes', 'real_estate', 'yacht'}


def test_d1_medium_datasets():
    datasets = {p.dataset for p in JOURNAL2026_D1_MEDIUM}
    assert datasets == {'abalone', 'crime', 'ribo', 'eye', 'naval_propulsion', 'parkinsons', 'student'}


def test_d2_counts():
    assert len(JOURNAL2026_D2_REGULAR) == 16
    assert len(JOURNAL2026_D2_LARGE) == 4
    assert len(JOURNAL2026_D2) == 20
    assert len(JOURNAL2026_D2_PREVIEW) == 9


def test_d3_counts():
    assert len(JOURNAL2026_D3_REGULAR) == 15
    assert len(JOURNAL2026_D3) == 15
    assert len(JOURNAL2026_D3_PREVIEW) == 8


def test_estimator_names():
    assert JOURNAL2026_EST_NAMES == ['EM', 'CV_fix', 'CV_glm']
    assert len(JOURNAL2026_ESTIMATORS) == 3
    assert TIMING_EST_NAMES == ['EM', 'CV_glm_11', 'CV_glm_101']
    assert len(TIMING_ESTIMATORS) == 3


def test_train_sizes_covers_all_d1_datasets():
    all_datasets = {p.dataset for p in JOURNAL2026_D1}
    assert all_datasets <= set(JOURNAL2026_TRAIN_SIZES)


def test_d3_preview_excludes_eye():
    datasets = {p.dataset for p in JOURNAL2026_D3_PREVIEW}
    assert 'eye' not in datasets


def test_d2_regular_excludes_ribo():
    datasets = {p.dataset for p in JOURNAL2026_D2_REGULAR}
    assert 'ribo' not in datasets
