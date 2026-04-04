from setuptools import setup

setup(
    name = 'fastridge',
    py_modules = ['fastridge'],
    version = 'v1.1.0',
    description = 'Fast and robust approach to ridge regression with simultaneous estimation of model parameters and hyperparameter tuning within a Bayesian framework via expectation-maximization (EM). ',
    author = 'Mario Boley',
    author_email = 'mario.boley@monash.edu',
    url = 'https://github.com/marioboley/fastridge.git',
    install_requires = ['numpy>=1.21.5', 'scipy>=1.8.1'],
    keywords = ['Ridge regression', 'EM'],
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
