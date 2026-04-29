import os
import numpy as np
import pytest
import neurips2023
from fastridge import RidgeEM
from problems import EmpiricalDataProblem, n_train_from_proportion
from neurips2023 import ExperimentWithPerSeriesSeeding


@pytest.fixture
def cache_dir(tmp_path):
    """tmp_path with the runs/ subdirectory pre-created, as in a real checkout."""
    os.makedirs(tmp_path / 'runs')
    return tmp_path


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


def test_series_exp_ignore_cache(cache_dir, monkeypatch):
    monkeypatch.setattr(neurips2023, 'CACHE_DIR', str(cache_dir))
    _simple_series_exp().run(ignore_cache=True)
    assert not os.path.exists(os.path.join(str(cache_dir), 'series'))


def test_series_exp_reproducible():
    exp1 = _simple_series_exp().run(ignore_cache=True)
    exp2 = _simple_series_exp().run(ignore_cache=True)
    np.testing.assert_array_equal(exp1.prediction_r2_, exp2.prediction_r2_)


def test_series_exp_overwrite_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(neurips2023, 'CACHE_DIR', str(tmp_path))
    _simple_series_exp().run()
    with pytest.warns(UserWarning, match='FittingTime'):
        _simple_series_exp().run(force_recompute=True)  # accumulates 2 computations
    _simple_series_exp().run(overwrite_cache=True)  # deletes per combo, rewrites: back to 1
    series_dir = os.path.join(str(tmp_path), 'series')
    json_files = [os.path.join(r, f)
                  for r, _, fs in os.walk(series_dir)
                  for f in fs if f.endswith('.json')]
    with open(json_files[0]) as f:
        import json
        data = json.load(f)
    assert len(data['computations']) == 1


def test_series_exp_log_file_written(tmp_path, monkeypatch):
    monkeypatch.setattr(neurips2023, 'CACHE_DIR', str(tmp_path))
    _simple_series_exp().run()
    runs_dir = os.path.join(str(tmp_path), 'runs')
    log_files = [f for f in os.listdir(runs_dir) if f.endswith('.log')]
    assert len(log_files) == 1


def test_series_exp_get_x_y_called_once_per_rep(monkeypatch):
    prob = EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)
    ns = n_train_from_proportion([prob])
    call_count = {'n': 0}
    original = EmpiricalDataProblem.get_X_y
    def counting_get_X_y(self, *args, **kwargs):
        call_count['n'] += 1
        return original(self, *args, **kwargs)
    monkeypatch.setattr(EmpiricalDataProblem, 'get_X_y', counting_get_X_y)
    exp = ExperimentWithPerSeriesSeeding(
        [prob], [RidgeEM(), RidgeEM()], reps=3, ns=ns, seed=1, verbose=False)
    exp.run(ignore_cache=True)
    # 1 n_size * 3 reps — shared across both estimators
    assert call_count['n'] == 3
