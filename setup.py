from distutils.core import setup

setup(
    name = 'fastridge',
    py_modules = ['fastridge'],
    version = 'v1.0.0',  # Ideally should be same as your github release tag varsion
    description = 'Fast and robust approach to ridge regression with simultaneous estimation of model parameters and hyperparameter tuning within a Bayesian framework via expectation-maximization (EM). ',
    author = 'Mario Boley',
    author_email = 'mario.boley@monash.edu',
    url = 'https://github.com/marioboley/fastridge.git',
    keywords = ['Ridge regression', 'EM'],
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
