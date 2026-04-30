import contextlib
import json
import os
import warnings
import numpy as np
import pytest
from util import to_json, from_json, save_json, load_json, route_warnings_to


# ── save_json / load_json ────────────────────────────────────────────────────

def test_save_load_roundtrip(tmp_path):
    path = str(tmp_path / 'test.json')
    data = {'a': 1, 'b': [2, 3]}
    save_json(path, data)
    assert load_json(path) == data


def test_load_json_missing_returns_default(tmp_path):
    assert load_json(str(tmp_path / 'missing.json'), default={'x': 1}) == {'x': 1}


def test_load_json_missing_returns_none(tmp_path):
    assert load_json(str(tmp_path / 'missing.json')) is None


def test_save_json_creates_parent_dirs(tmp_path):
    path = str(tmp_path / 'a' / 'b' / 'test.json')
    save_json(path, {'x': 1})
    assert os.path.exists(path)


def test_save_json_pretty_printed(tmp_path):
    path = str(tmp_path / 'pretty.json')
    save_json(path, {'a': 1}, indent=2)
    with open(path) as f:
        assert '\n' in f.read()


def test_save_json_compact(tmp_path):
    path = str(tmp_path / 'compact.json')
    save_json(path, {'a': 1}, indent=None)
    with open(path) as f:
        assert '\n' not in f.read()


# ── to_json dispatch ─────────────────────────────────────────────────────────

def test_to_json_primitives():
    assert to_json(None) is None
    assert to_json(True) is True
    assert to_json(42) == 42
    assert to_json(3.14) == 3.14
    assert to_json('hello') == 'hello'


def test_to_json_numpy_integer():
    result = to_json(np.int64(42))
    assert result == 42
    assert type(result) is int


def test_to_json_numpy_floating():
    result = to_json(np.float64(3.14))
    assert abs(result - 3.14) < 1e-10
    assert type(result) is float


def test_to_json_ndarray():
    result = to_json(np.array([[1, 2], [3, 4]]))
    assert result == [[1, 2], [3, 4]]
    assert isinstance(result, list)


def test_to_json_list():
    assert to_json([1, np.int64(2), 'a']) == [1, 2, 'a']


def test_to_json_tuple():
    assert to_json((1, 2)) == {'__tuple__': [1, 2]}


def test_to_json_dict():
    assert to_json({'a': np.int64(1), 'b': [np.float64(2.0)]}) == {'a': 1, 'b': [2.0]}


def test_to_json_named_import():
    assert to_json(np.log) == {'__import__': 'numpy.log'}


def test_to_json_transparent_class():
    from problems import EmpiricalDataProblem
    prob = EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=False)
    result = to_json(prob, include_defaults=False)
    assert result['__class__'] == 'problems.EmpiricalDataProblem'
    assert result['dataset'] == 'diabetes'
    assert 'zero_variance_filter' not in result
    assert 'x_transforms' not in result

def test_to_json_transparent_class_with_defaults():
    from problems import EmpiricalDataProblem
    prob = EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)
    result = to_json(prob, include_defaults=True)
    assert result['__class__'] == 'problems.EmpiricalDataProblem'
    assert result['dataset'] == 'diabetes'
    assert result['zero_variance_filter'] is True
    assert result['x_transforms'] == {'__tuple__': []}

def test_to_json_include_computed_false_excludes_underscore_attrs():
    from experiments import Metric
    m = Metric()
    m.computed_ = 99
    assert 'computed_' not in to_json(m, include_computed=False)


def test_to_json_include_computed_true_includes_all():
    from experiments import Metric
    m = Metric()
    m.a_ = 1
    m.b_ = 2
    result = to_json(m, include_computed=True)
    assert result['a_'] == 1
    assert result['b_'] == 2


def test_to_json_include_computed_list_selects_named():
    from experiments import Metric
    m = Metric()
    m.a_ = 1
    m.b_ = 2
    result = to_json(m, include_computed=['a_'])
    assert result['a_'] == 1
    assert 'b_' not in result


def test_to_json_warns_on_missing_attribute():
    class Leaky:
        def __init__(self, x):
            pass  # never stores x
    obj = Leaky.__new__(Leaky)
    with pytest.warns(UserWarning, match='x'):
        to_json(obj)


def test_to_json_does_not_propagate_include_computed():
    from fastridge import RidgeEM
    from experiments import Experiment, prediction_r2
    from problems import EmpiricalDataProblem
    prob = EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)
    exp = Experiment([prob], [RidgeEM()], reps=2, ns=[[309]], seed=1, verbose=False,
                     est_names=['EM'], stats=[prediction_r2])
    exp.run_id_ = 'test__20260425-000000-abcd'
    result = to_json(exp, include_computed=True)
    assert result['run_id_'] == 'test__20260425-000000-abcd'
    # nested estimator is serialised spec-only regardless of include_computed
    est_result = result['estimators'][0]
    assert not any(k.endswith('_') and not k.startswith('_') for k in est_result)


# ── from_json dispatch ───────────────────────────────────────────────────────

def test_from_json_primitives():
    assert from_json(None) is None
    assert from_json(True) is True
    assert from_json(42) == 42
    assert from_json(3.14) == pytest.approx(3.14)
    assert from_json('hello') == 'hello'


def test_from_json_list():
    assert from_json([1, 2, 3]) == [1, 2, 3]


def test_from_json_tuple():
    assert from_json({'__tuple__': [1, 2]}) == (1, 2)


def test_from_json_named_import():
    assert from_json({'__import__': 'numpy.log'}) is np.log


def test_from_json_plain_dict():
    assert from_json({'a': 1, 'b': [2, 3]}) == {'a': 1, 'b': [2, 3]}


def test_from_json_transparent_class():
    from problems import EmpiricalDataProblem
    data = {
        '__class__': 'problems.EmpiricalDataProblem',
        'dataset': 'diabetes',
        'target': 'target',
        'drop': {'__tuple__': []},
        'nan_policy': None,
        'x_transforms': {'__tuple__': []},
        'y_transforms': {'__tuple__': []},
        'zero_variance_filter': True,
    }
    obj = from_json(data)
    assert isinstance(obj, EmpiricalDataProblem)
    assert obj.dataset == 'diabetes'
    assert obj.zero_variance_filter is True
    assert obj.x_transforms == ()


def test_from_json_restores_computed_attrs():
    from experiments import Metric
    data = {'__class__': 'experiments.Metric', 'run_id_': 'test__20260425-000000-abcd'}
    obj = from_json(data)
    assert isinstance(obj, Metric)
    assert obj.run_id_ == 'test__20260425-000000-abcd'


# ── end-to-end experiment roundtrips ─────────────────────────────────────────

def test_experiment_roundtrip(tmp_path):
    from fastridge import RidgeEM, RidgeLOOCV
    from experiments import Experiment, prediction_r2, prediction_mean_squared_error
    from problems import EmpiricalDataProblem, OneHotEncodeCategories

    prob_plain = EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)
    prob_ohe = EmpiricalDataProblem(
        'abalone', 'Rings',
        x_transforms=(OneHotEncodeCategories(),),
        zero_variance_filter=True)
    estimators = [RidgeEM(), RidgeLOOCV(alphas=100)]
    exp = Experiment(
        [prob_plain, prob_ohe], estimators,
        reps=10, ns=[[309], [3177]], seed=1, verbose=False,
        est_names=['EM', 'CV'],
        stats=[prediction_r2, prediction_mean_squared_error])

    path = str(tmp_path / 'exp.json')
    save_json(path, to_json(exp))
    reconstructed = load_json(path)
    assert to_json(reconstructed) == to_json(exp)


def test_experiment_with_per_series_seeding_roundtrip(tmp_path):
    from fastridge import RidgeEM, RidgeLOOCV
    from neurips2023 import ExperimentWithPerSeriesSeeding
    from experiments import prediction_r2
    from problems import EmpiricalDataProblem
    from neurips2023 import NEURIPS2023_TRAIN_SIZES

    problems = [EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)]
    estimators = [RidgeEM(t2=False), RidgeLOOCV(alphas=np.logspace(-10, 10, 11, base=10))]
    exp = ExperimentWithPerSeriesSeeding(
        problems, estimators, reps=100,
        ns=[[NEURIPS2023_TRAIN_SIZES['diabetes']]], seed=1, verbose=False,
        est_names=['EM', 'CV_fix'],
        stats=[prediction_r2])

    path = str(tmp_path / 'exp_series.json')
    save_json(path, to_json(exp))
    reconstructed = load_json(path)
    assert to_json(reconstructed) == to_json(exp)


# ── route_warnings_to ────────────────────────────────────────────────────────

def test_route_warnings_to_redirects():
    received = []
    with route_warnings_to(received.append, propagate=False):
        warnings.warn('hello', UserWarning)
    assert received == ['UserWarning: hello']


def test_route_warnings_to_restores_on_exit():
    orig = warnings.showwarning
    with route_warnings_to(lambda s: None):
        pass
    assert warnings.showwarning is orig


def test_route_warnings_to_restores_on_exception():
    orig = warnings.showwarning
    with contextlib.suppress(ValueError):
        with route_warnings_to(lambda s: None):
            raise ValueError
    assert warnings.showwarning is orig


def test_route_warnings_to_propagate_true_chains():
    chained = []
    sentinel = lambda msg, cat, fn, ln, file=None, line=None: chained.append(str(msg))
    orig = warnings.showwarning
    warnings.showwarning = sentinel
    try:
        received = []
        with route_warnings_to(received.append, propagate=True):
            warnings.warn('hi', UserWarning)
        assert received == ['UserWarning: hi']
        assert chained == ['hi']
    finally:
        warnings.showwarning = orig
