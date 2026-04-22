import time
import numpy as np
from sklearn.base import clone
from fastprogress.fastprogress import progress_bar
from experiments import default_stats


class SyntheticDataExperiment:

    def __init__(self, problems, estimators, ns, reps, est_names=None, stats=default_stats,
                 keep_fits=True, verbose=0, seed=None):
        self.problems = problems
        self.estimators = estimators
        self.ns = np.atleast_2d(ns)
        self.ns = self.ns if len(self.ns) == len(self.problems) else self.ns.repeat(len(problems), axis=0)
        self.reps = reps
        self.verbose = verbose
        self.est_names = [str(est) for est in estimators] if est_names is None else est_names
        self.stats = stats
        self.keep_fits = keep_fits
        self.test_size = 10000
        self.rng = np.random.default_rng(seed)

    def run(self):
        if self.keep_fits:
            self.fits = {}
        for stat in self.stats:
            self.__dict__[str(stat) + '_'] = np.zeros(
                shape=(self.reps, len(self.problems), len(self.ns[0]), len(self.estimators)))
        for r in progress_bar(range(self.reps)):
            for i in range(len(self.problems)):
                x_test, y_test = self.problems[i].rvs(self.test_size, rng=self.rng)
                for n_idx, n in enumerate(self.ns[i]):
                    for j, est in enumerate(self.estimators):
                        x, y = self.problems[i].rvs(n, rng=self.rng)
                        _est = clone(est, safe=False)
                        fit_start_time = time.time()
                        _est.fit(x, y)
                        _est.fitting_time_ = time.time() - fit_start_time
                        if self.keep_fits:
                            self.fits[(r, i, n, j)] = _est
                        for stat in self.stats:
                            self.__dict__[str(stat) + '_'][r, i, n_idx, j] = stat(
                                _est, self.problems[i], x_test, y_test)
        return self
