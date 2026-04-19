import numpy as np
import pandas as pd
import pytest
from problems import PolynomialExpansion, OneHotEncodeCategories


def test_polynomial_expansion_rng_deterministic():
    X = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})
    small = PolynomialExpansion(2, max_entries=9)
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    assert list(small(X, rng1).columns) == list(small(X, rng2).columns)


def test_one_hot_encode_accepts_rng():
    X = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
    enc = OneHotEncodeCategories()
    result = enc(X, np.random.default_rng(0))
    assert result.equals(X)


from problems import EmpiricalDataProblem, PolynomialExpansion, NEURIPS2023, NEURIPS2023_D2, NEURIPS2023_D3


def test_zero_variance_filter_default_false():
    assert EmpiricalDataProblem('diabetes', 'target').zero_variance_filter is False


def test_zero_variance_filter_in_repr_only_when_true():
    assert 'zero_variance_filter' not in repr(EmpiricalDataProblem('diabetes', 'target'))
    assert repr(EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)) == \
        "EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)"


def test_zero_variance_filter_affects_equality():
    p_false = EmpiricalDataProblem('diabetes', 'target')
    p_true = EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)
    assert p_false == EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=False)
    assert p_false != p_true


def test_get_X_y_returns_four_tuple_with_correct_sizes():
    prob = EmpiricalDataProblem('diabetes', 'target')
    X_train, X_test, y_train, y_test = prob.get_X_y(300)
    assert X_train.shape[0] == 300
    assert X_train.shape[1] == 10
    assert len(y_train) == 300
    assert len(y_test) == X_test.shape[0]


def test_get_X_y_seeded_reproducible():
    prob = EmpiricalDataProblem('diabetes', 'target')
    Xtr1, _, _, _ = prob.get_X_y(300, rng=0)
    Xtr2, _, _, _ = prob.get_X_y(300, rng=0)
    assert list(Xtr1.index) == list(Xtr2.index)


def test_get_X_y_zero_variance_filter_removes_constant_columns():
    prob_no = EmpiricalDataProblem('naval_propulsion', 'GT_compressor_decay',
                                   drop=['GT_turbine_decay'])
    prob_filt = EmpiricalDataProblem('naval_propulsion', 'GT_compressor_decay',
                                     drop=['GT_turbine_decay'],
                                     zero_variance_filter=True)
    Xtr_no, Xte_no, _, _ = prob_no.get_X_y(50, rng=0)
    Xtr_f, Xte_f, _, _ = prob_filt.get_X_y(50, rng=0)
    std = Xtr_no.std()
    zero_var = std[std == 0].index.tolist()
    assert len(zero_var) > 0, "test requires at least one zero-variance train column"
    expected_cols = [c for c in Xtr_no.columns if c not in zero_var]
    assert list(Xtr_f.columns) == expected_cols
    assert list(Xte_f.columns) == expected_cols


def test_get_X_y_rng_threading_to_transforms():
    X_base = pd.DataFrame({'a': np.arange(100, dtype=float),
                           'b': np.arange(100, dtype=float)})
    pe = PolynomialExpansion(2, max_entries=200)
    prob = EmpiricalDataProblem.__new__(EmpiricalDataProblem)
    prob.dataset = 'diabetes'
    prob.target = 'target'
    prob.drop = ()
    prob.nan_policy = None
    prob.x_transforms = (PolynomialExpansion(2, max_entries=50),)
    prob.y_transforms = ()
    prob.zero_variance_filter = False
    prob._repr = ''
    Xtr1, _, _, _ = prob.get_X_y(50, rng=1)
    Xtr2, _, _, _ = prob.get_X_y(50, rng=1)
    assert list(Xtr1.columns) == list(Xtr2.columns)


def test_neurips2023_zero_variance_filter():
    assert all(p.zero_variance_filter for p in NEURIPS2023)
    assert all(p.zero_variance_filter for p in NEURIPS2023_D2)
    assert all(p.zero_variance_filter for p in NEURIPS2023_D3)
