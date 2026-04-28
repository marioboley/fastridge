import os
import json
import numpy as np
import pytest
import experiments
from fastridge import RidgeEM
from problems import EmpiricalDataProblem, n_train_from_proportion
from neurips2023 import SyntheticDataExperiment
import neurips2023
from experiments import (Experiment, Metric,
                         parameter_mean_squared_error, prediction_mean_squared_error,
                         regularization_parameter, number_of_iterations, variance_abs_error,
                         fitting_time, prediction_r2, number_of_features)
from neurips2023 import ExperimentWithPerSeriesSeeding


def test_ns_shape():
    prob = EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)
    exp = Experiment([prob], [RidgeEM()], reps=2, ns=n_train_from_proportion([prob]))
    assert exp.ns.shape == (1, 1)
    assert int(exp.ns[0, 0]) == 309


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


from util import save_json, load_json
from experiments import make_run_id



# tmp_path: pytest built-in fixture providing an isolated temporary directory
# https://docs.pytest.org/en/stable/how-to/tmp_path.html
def test_load_metric_file_missing(tmp_path):
    data = load_json(str(tmp_path / 'missing.json'),
                     default={'computations': [], 'retrievals': []})
    assert data == {'computations': [], 'retrievals': []}


def test_save_load_metric_file_roundtrip(tmp_path):
    path = str(tmp_path / 'sub' / 'm.json')
    original = {'computations': [{'value': 0.85, 'run_id': 'abc'}], 'retrievals': []}
    save_json(path, original)
    assert load_json(path) == original


def testmake_run_id_format():
    run_id = make_run_id('Experiment')
    assert run_id.startswith('Experiment__')
    tail = run_id[len('Experiment__'):]
    parts = tail.split('-')
    assert len(parts) == 3
    assert len(parts[0]) == 8   # YYYYMMDD
    assert len(parts[1]) == 6   # HHMMSS
    assert len(parts[2]) == 4   # random suffix



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


def test_synthetic_experiment_importable():
    assert SyntheticDataExperiment is not None


def test_synthetic_experiment_not_in_experiments():
    assert not hasattr(experiments, 'SyntheticDataExperiment')


def _simple_new_exp(**kwargs):
    prob = EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)
    ns = n_train_from_proportion([prob])
    defaults = dict(seed=1, verbose=False)
    defaults.update(kwargs)
    return Experiment([prob], [RidgeEM()], reps=2, ns=ns, **defaults)


def test_new_experiment_result_shape():
    assert _simple_new_exp().run(ignore_cache=True).prediction_r2_.shape == (2, 1, 1, 1)


def test_new_experiment_trial_cache_hit(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    exp1 = _simple_new_exp().run()
    with pytest.warns(UserWarning, match='FittingTime'):
        exp2 = _simple_new_exp().run()
    np.testing.assert_array_equal(exp1.prediction_r2_, exp2.prediction_r2_)


def test_new_experiment_ignore_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    _simple_new_exp().run(ignore_cache=True)
    assert not os.path.exists(os.path.join(str(tmp_path), 'trial'))


def test_new_experiment_force_recompute(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    _simple_new_exp().run()
    with pytest.warns(UserWarning, match='FittingTime'):
        _simple_new_exp().run(force_recompute=True)
    trial_dir = os.path.join(str(tmp_path), 'trial')
    json_files = [os.path.join(r, f)
                  for r, _, fs in os.walk(trial_dir)
                  for f in fs if f.endswith('.json')]
    assert json_files
    with open(json_files[0]) as f:
        data = json.load(f)
    assert len(data['computations']) == 2


def test_new_experiment_run_file_written(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    _simple_new_exp().run()
    assert len(os.listdir(os.path.join(str(tmp_path), 'runs'))) == 1


def test_new_experiment_run_file_content(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    _simple_new_exp().run()
    runs_dir = os.path.join(str(tmp_path), 'runs')
    filenames = os.listdir(runs_dir)
    assert len(filenames) == 1
    assert filenames[0].startswith('Experiment__')
    with open(os.path.join(runs_dir, filenames[0])) as f:
        data = json.load(f)
    assert data['__class__'] == 'experiments.Experiment'
    assert data['run_id_'].startswith('Experiment__')
    assert data['timestamp_start_'] is not None
    assert data['timestamp_end_'] is not None
    assert 'python' in data['environment_']
    assert data['problems'][0]['__class__'] == 'problems.EmpiricalDataProblem'
    assert data['reps'] == 2


def _simple_series_exp(**kwargs):
    prob = EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)
    ns = n_train_from_proportion([prob])
    defaults = dict(seed=1, verbose=False)
    defaults.update(kwargs)
    return ExperimentWithPerSeriesSeeding([prob], [RidgeEM()], reps=2, ns=ns, **defaults)


def test_series_exp_result_shape():
    assert _simple_series_exp().run(ignore_cache=True).prediction_r2_.shape == (2, 1, 1, 1)


def test_series_exp_cache_hit(tmp_path, monkeypatch):
    monkeypatch.setattr(neurips2023, 'CACHE_DIR', str(tmp_path))
    exp1 = _simple_series_exp().run()
    with pytest.warns(UserWarning, match='FittingTime'):
        exp2 = _simple_series_exp().run()
    np.testing.assert_array_equal(exp1.prediction_r2_, exp2.prediction_r2_)


def test_series_exp_ignore_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(neurips2023, 'CACHE_DIR', str(tmp_path))
    _simple_series_exp().run(ignore_cache=True)
    assert not os.path.exists(os.path.join(str(tmp_path), 'series'))


def test_series_exp_reproducible():
    exp1 = _simple_series_exp().run(ignore_cache=True)
    exp2 = _simple_series_exp().run(ignore_cache=True)
    np.testing.assert_array_equal(exp1.prediction_r2_, exp2.prediction_r2_)


def test_new_experiment_overwrite_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    _simple_new_exp().run()
    _simple_new_exp().run(force_recompute=True)  # accumulates 2 computations
    _simple_new_exp().run(overwrite_cache=True)  # deletes per combo, rewrites: back to 1
    trial_dir = os.path.join(str(tmp_path), 'trial')
    json_files = [os.path.join(r, f)
                  for r, _, fs in os.walk(trial_dir)
                  for f in fs if f.endswith('.json')]
    with open(json_files[0]) as f:
        data = json.load(f)
    assert len(data['computations']) == 1


def test_series_exp_overwrite_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(neurips2023, 'CACHE_DIR', str(tmp_path))
    _simple_series_exp().run()
    _simple_series_exp().run(force_recompute=True)  # accumulates 2 computations
    _simple_series_exp().run(overwrite_cache=True)  # deletes per combo, rewrites: back to 1
    series_dir = os.path.join(str(tmp_path), 'series')
    json_files = [os.path.join(r, f)
                  for r, _, fs in os.walk(series_dir)
                  for f in fs if f.endswith('.json')]
    with open(json_files[0]) as f:
        data = json.load(f)
    assert len(data['computations']) == 1
