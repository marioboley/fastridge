import numpy as np
import pytest
from fastridge import RidgeEM
from problems import EmpiricalDataProblem, n_train_from_proportion
from experiments import EmpiricalDataExperiment


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


def test_pcg64_and_mt19937_differ():
    exp_mt = _simple_exp(generator='MT19937').run()
    exp_pc = _simple_exp(generator='PCG64').run()
    assert not np.array_equal(exp_mt.prediction_r2_, exp_pc.prediction_r2_)
