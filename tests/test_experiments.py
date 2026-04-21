import os
import json
import numpy as np
import pytest
import experiments
from fastridge import RidgeEM
from problems import EmpiricalDataProblem, n_train_from_proportion
from experiments import (EmpiricalDataExperiment, Metric,
                         parameter_mean_squared_error, prediction_mean_squared_error,
                         regularization_parameter, number_of_iterations, variance_abs_error,
                         fitting_time, prediction_r2, number_of_features)


def _simple_exp(**kwargs):
    prob = EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)
    ns = n_train_from_proportion([prob])
    defaults = dict(seed=1, generator='MT19937', verbose=False)
    defaults.update(kwargs)
    return EmpiricalDataExperiment([prob], [RidgeEM()], reps=2, ns=ns, **defaults)


def test_result_shape():
    assert _simple_exp().run().prediction_r2_.shape == (2, 1, 1, 1)


def test_ns_shape():
    exp = _simple_exp()
    assert exp.ns.shape == (1, 1)
    assert int(exp.ns[0, 0]) == 309  # 442 * 0.7


def test_make_rng_fixed_progression_same_seed():
    exp = _simple_exp(seed_progression='fixed')
    r0 = exp._make_rng(unit_idx=0)
    r5 = exp._make_rng(unit_idx=5)
    assert r0.randint(10000) == r5.randint(10000)


def test_make_rng_sequential_progression_different_seeds():
    exp = _simple_exp(seed_progression='sequential')
    r0 = exp._make_rng(unit_idx=0)
    r1 = exp._make_rng(unit_idx=1)
    assert r0.randint(10000) != r1.randint(10000)


def test_series_scope_reproducible():
    exp1 = _simple_exp()
    exp2 = _simple_exp()
    exp1.run()
    exp2.run()
    np.testing.assert_array_equal(exp1.prediction_r2_, exp2.prediction_r2_)


def test_stat_instances_are_metric():
    for inst in [parameter_mean_squared_error, prediction_mean_squared_error,
                 regularization_parameter, number_of_iterations, variance_abs_error,
                 fitting_time, prediction_r2, number_of_features]:
        assert isinstance(inst, Metric)


def test_warn_recompute_returns_none_when_close():
    assert Metric().warn_recompute([0.85, 0.85, 0.85], 0.85) is None
    assert Metric().warn_recompute([0.85, 0.85], 0.85 + 1e-9) is None


def test_warn_recompute_returns_str_on_meaningful_change():
    assert isinstance(Metric().warn_recompute([0.85, 0.85, 0.85], 0.84), str)


def test_warn_recompute_series_element_wise():
    existing = [[0.85, 0.90], [0.85, 0.90]]
    assert Metric().warn_recompute(existing, [0.85, 0.10]) is not None
    assert Metric().warn_recompute(existing, [0.85, 0.90]) is None


def test_warn_retrieval_returns_none_when_consistent():
    assert Metric().warn_retrieval([{'value': 0.85, 'run_id': 'x'},
                                    {'value': 0.85, 'run_id': 'y'}]) is None


def test_warn_retrieval_returns_str_on_meaningful_variation():
    assert isinstance(Metric().warn_retrieval([{'value': 0.85, 'run_id': 'x'},
                                               {'value': 0.80, 'run_id': 'y'}]), str)


from experiments import (_cache_key, _load_metric_file, _save_metric_file,
                         _make_run_id, _write_run_file)


def test_cache_key_format():
    key = _cache_key(EmpiricalDataProblem('diabetes', 'target'))
    assert key.startswith('EmpiricalDataProblem__')
    assert len(key.split('__')[1]) > 0


# tmp_path: pytest built-in fixture providing an isolated temporary directory
# https://docs.pytest.org/en/stable/how-to/tmp_path.html
def test_load_metric_file_missing(tmp_path):
    data = _load_metric_file(str(tmp_path / 'missing.json'))
    assert data == {'computations': [], 'retrievals': []}


def test_save_load_metric_file_roundtrip(tmp_path):
    path = str(tmp_path / 'sub' / 'm.json')
    original = {'computations': [{'value': 0.85, 'run_id': 'abc'}], 'retrievals': []}
    _save_metric_file(path, original)
    assert _load_metric_file(path) == original


def test_make_run_id_format():
    parts = _make_run_id().split('-')
    assert len(parts) == 3
    assert len(parts[0]) == 8
    assert len(parts[1]) == 6
    assert len(parts[2]) == 4


# monkeypatch: pytest built-in fixture for temporarily patching module attributes
# https://docs.pytest.org/en/stable/how-to/monkeypatch.html
def test_write_run_file(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    run_id = _make_run_id()
    _write_run_file(run_id, {'problems': []},
                    {'trials_computed': 0, 'trials_retrieved': 0})
    path = os.path.join(str(tmp_path), 'runs', f'{run_id}.json')
    assert os.path.exists(path)
    with open(path) as f:
        data = json.load(f)
    assert data['run_id'] == run_id
    assert 'python' in data['environment']


def test_fitting_time_warn_recompute_tolerates_ci_variation():
    existing = [0.5, 0.52, 0.48]
    assert fitting_time.warn_recompute(existing, 0.51) is None


def test_fitting_time_warn_recompute_fires_outside_ci():
    existing = [0.5, 0.5, 0.5]
    assert isinstance(fitting_time.warn_recompute(existing, 5.0), str)


def test_fitting_time_warn_retrieval_single_computation():
    assert isinstance(fitting_time.warn_retrieval([{'value': 0.5, 'run_id': 'x'}]), str)


def test_fitting_time_warn_retrieval_narrow_ci():
    comps = [{'value': v, 'run_id': 'x'} for v in [0.5, 0.51, 0.49]]
    assert fitting_time.warn_retrieval(comps) is None


def test_fitting_time_warn_retrieval_wide_ci():
    comps = [{'value': v, 'run_id': 'x'} for v in [0.1, 10.0, 0.1]]
    assert fitting_time.warn_retrieval(comps) is not None


def test_pcg64_and_mt19937_differ():
    exp_mt = _simple_exp(generator='MT19937').run()
    exp_pc = _simple_exp(generator='PCG64').run()
    assert not np.array_equal(exp_mt.prediction_r2_, exp_pc.prediction_r2_)
