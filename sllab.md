# Statistical Learning Lab

Part of the code for running statistical learning experiment in the subfolder `experiments` is supposed to form the core of an independent Python package `sllab`---short for Statistical Learning Lab.

## Goal

In one sentence, the goal of `sllab` is to provide an abstraction of statistical learning experiments that allows to conveniently evaluate estimators that follow the scikit-learn standard with minimal code and computational overhead and maximal reproducibility and standardisation.

## Assumptions and Constraints

### Unified abstractions of statistical experiment

For package maintainability as well as conceptual and API simplicity, the package contains only a single unified abstraction of a statistical learning experiment based on the notions of:

- **problem**, corresponding to statistical learning problems or, more precisely, families of problems that are parameterised by training (sample) size `n` and potentially by the number `p` and `q` of predictor (input) and target (output) variables, respectively (as well as by potentially further parameters such as training set diversity)

- **estimator**, a statistical learning approach, subsuming model and fitting algorithm, as captured in `scikit-learn` estimator objects. 

- **trial**, corresponding to a single repetition of fitting an estimator to a problem, where a repetition is characterised by all random choices in generating the training data for an estimator as well as potentially additional random choices made internally by the estimator

- **metric**, a numerical performance indicator that produces a single statistic (measurement) for a given trial that can depend on the problem, the fitted estimator, as well as the concrete training and test datasets associated with the trial.

In particular, an experiment consists of a set of problems, problem parameters, estimators, trials, and metrics.

### Block designs

While trials subsume the randomness of data generation and estimator decisions (on both fitting and prediction), experiments define block designs, where trials are ``blocked'' based on the data (problem) randomness only. That is, for each realisation of problem randomness, there is a block of trials that share that data randomness, one for each estimator. Note that his use of the terms "block" and "trial" differs slightly from their standard use in experiment design theory (where a trial would be understood to correspond to a block). Here, we prefer the modified block notion to retain a trial is an elementary cache unit. Two further notes regarding the nomenclature are in order:

1. Estimators can fail for individual trials, meaning that, despite the block design, there is not necessarily an equal number of numerical measurements for all estimators. If one considers failure as a possible outcome then the described block designs are ``complete''. If one does not, then they become incomplete. Both interpretations are supported.

2. Blocks can be repeated within an experiment leading to multiple trials with the same data randomness but potentially different estimator randomness. This is useful, e.g., to assess the effect of estimator randomness on performance stability.


### Large datasets that can be efficiently generated

Experiments can involve datasets of significant size relative to the amount of available application memory. This includes both train and test data. This assumption precludes caching more than one variant of training and test data at a given time during experiment execution. On the other hand, it is assumed that datasets can be generated efficiently, i.e., in linear time of their size.