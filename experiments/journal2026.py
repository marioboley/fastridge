from fastridge import RidgeEM, RidgeLOOCV
from problems import EmpiricalDataProblem, PolynomialExpansion, onehot_non_numeric
from neurips2023 import NEURIPS2023_TRAIN_SIZES


JOURNAL2026_TRAIN_SIZES = NEURIPS2023_TRAIN_SIZES

# ── D1 (no polynomial expansion) ────────────────────────────────────────────

JOURNAL2026_D1_TINY = [
    EmpiricalDataProblem('autompg',          'mpg',
                         drop=('car_name',), nan_policy='drop_rows',
                         zero_variance_filter=True),
    EmpiricalDataProblem('diabetes',         'target',
                         zero_variance_filter=True),
    EmpiricalDataProblem('real_estate',      'Y house price of unit area',
                         zero_variance_filter=True),
    EmpiricalDataProblem('yacht',            'Residuary_resistance',
                         zero_variance_filter=True),
]

JOURNAL2026_D1_SMALL = [
    EmpiricalDataProblem('airfoil',          'scaled-sound-pressure',
                         zero_variance_filter=True),
    EmpiricalDataProblem('automobile',       'price',
                         nan_policy='drop_rows', x_transforms=onehot_non_numeric,
                         zero_variance_filter=True),
    EmpiricalDataProblem('boston',           'medv',
                         zero_variance_filter=True),
    EmpiricalDataProblem('concrete',         'Concrete compressive strength',
                         zero_variance_filter=True),
    EmpiricalDataProblem('facebook',         ('comment', 'like', 'share'),
                         drop=('Total Interactions',), nan_policy='drop_rows',
                         x_transforms=onehot_non_numeric, zero_variance_filter=True),
    EmpiricalDataProblem('forest',           'area',
                         x_transforms=onehot_non_numeric, zero_variance_filter=True),
]

JOURNAL2026_D1_MEDIUM = [
    EmpiricalDataProblem('abalone',          'Rings',
                         x_transforms=onehot_non_numeric, zero_variance_filter=True),
    EmpiricalDataProblem('crime',            'ViolentCrimesPerPop',
                         drop=('state', 'fold', 'communityname'), nan_policy='drop_cols',
                         zero_variance_filter=True),
    EmpiricalDataProblem('ribo',             'y',
                         zero_variance_filter=True),
    EmpiricalDataProblem('eye',              'y',
                         zero_variance_filter=True),
    EmpiricalDataProblem('naval_propulsion', ('GT_compressor_decay', 'GT_turbine_decay'),
                         zero_variance_filter=True),
    EmpiricalDataProblem('parkinsons',       ('motor_UPDRS', 'total_UPDRS'),
                         zero_variance_filter=True),
    EmpiricalDataProblem('student',          ('G1', 'G2', 'G3'),
                         x_transforms=onehot_non_numeric, zero_variance_filter=True),
]

JOURNAL2026_D1_LARGE = [
    EmpiricalDataProblem('twitter',   'V78',       zero_variance_filter=True),
    EmpiricalDataProblem('tomshw',    'V97',       zero_variance_filter=True),
    EmpiricalDataProblem('blog',      'V281',      zero_variance_filter=True),
    EmpiricalDataProblem('ct_slices', 'reference', zero_variance_filter=True),
]

JOURNAL2026_D1_REGULAR = JOURNAL2026_D1_TINY + JOURNAL2026_D1_SMALL + JOURNAL2026_D1_MEDIUM
JOURNAL2026_D1 = JOURNAL2026_D1_REGULAR + JOURNAL2026_D1_LARGE

# ── D2 (degree-2 polynomial expansion) ──────────────────────────────────────

JOURNAL2026_D2_TINY = [
    EmpiricalDataProblem('autompg',          'mpg',
                         drop=('car_name',), nan_policy='drop_rows',
                         x_transforms=(PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('diabetes',         'target',
                         x_transforms=(PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('real_estate',      'Y house price of unit area',
                         x_transforms=(PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('yacht',            'Residuary_resistance',
                         x_transforms=(PolynomialExpansion(2),),
                         zero_variance_filter=True),
]

JOURNAL2026_D2_SMALL = [
    EmpiricalDataProblem('airfoil',          'scaled-sound-pressure',
                         x_transforms=(PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('automobile',       'price',
                         nan_policy='drop_rows',
                         x_transforms=onehot_non_numeric + (PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('boston',           'medv',
                         x_transforms=(PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('concrete',         'Concrete compressive strength',
                         x_transforms=(PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('facebook',         ('comment', 'like', 'share'),
                         drop=('Total Interactions',), nan_policy='drop_rows',
                         x_transforms=onehot_non_numeric + (PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('forest',           'area',
                         x_transforms=onehot_non_numeric + (PolynomialExpansion(2),),
                         zero_variance_filter=True),
]

JOURNAL2026_D2_MEDIUM = [
    EmpiricalDataProblem('abalone',          'Rings',
                         x_transforms=onehot_non_numeric + (PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('crime',            'ViolentCrimesPerPop',
                         drop=('state', 'fold', 'communityname'), nan_policy='drop_cols',
                         x_transforms=(PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('eye',              'y',
                         x_transforms=(PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('naval_propulsion', ('GT_compressor_decay', 'GT_turbine_decay'),
                         x_transforms=(PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('parkinsons',       ('motor_UPDRS', 'total_UPDRS'),
                         x_transforms=(PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('student',          ('G1', 'G2', 'G3'),
                         x_transforms=onehot_non_numeric + (PolynomialExpansion(2),),
                         zero_variance_filter=True),
]

JOURNAL2026_D2_LARGE = [
    EmpiricalDataProblem('twitter',   'V78',
                         x_transforms=(PolynomialExpansion(2),), zero_variance_filter=True),
    EmpiricalDataProblem('tomshw',    'V97',
                         x_transforms=(PolynomialExpansion(2),), zero_variance_filter=True),
    EmpiricalDataProblem('blog',      'V281',
                         x_transforms=(PolynomialExpansion(2),), zero_variance_filter=True),
    EmpiricalDataProblem('ct_slices', 'reference',
                         x_transforms=(PolynomialExpansion(2),), zero_variance_filter=True),
]

JOURNAL2026_D2_REGULAR = JOURNAL2026_D2_TINY + JOURNAL2026_D2_SMALL + JOURNAL2026_D2_MEDIUM
JOURNAL2026_D2 = JOURNAL2026_D2_REGULAR + JOURNAL2026_D2_LARGE

# ── D3 (degree-3 polynomial expansion; no large datasets) ───────────────────

JOURNAL2026_D3_TINY = [
    EmpiricalDataProblem('autompg',          'mpg',
                         drop=('car_name',), nan_policy='drop_rows',
                         x_transforms=(PolynomialExpansion(3),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('diabetes',         'target',
                         x_transforms=(PolynomialExpansion(3),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('real_estate',      'Y house price of unit area',
                         x_transforms=(PolynomialExpansion(3),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('yacht',            'Residuary_resistance',
                         x_transforms=(PolynomialExpansion(3),),
                         zero_variance_filter=True),
]

JOURNAL2026_D3_SMALL = [
    EmpiricalDataProblem('airfoil',          'scaled-sound-pressure',
                         x_transforms=(PolynomialExpansion(3),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('automobile',       'price',
                         nan_policy='drop_rows',
                         x_transforms=onehot_non_numeric + (PolynomialExpansion(3),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('boston',           'medv',
                         x_transforms=(PolynomialExpansion(3),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('concrete',         'Concrete compressive strength',
                         x_transforms=(PolynomialExpansion(3),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('facebook',         ('comment', 'like', 'share'),
                         drop=('Total Interactions',), nan_policy='drop_rows',
                         x_transforms=onehot_non_numeric + (PolynomialExpansion(3),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('forest',           'area',
                         x_transforms=onehot_non_numeric + (PolynomialExpansion(3),),
                         zero_variance_filter=True),
]

JOURNAL2026_D3_MEDIUM = [
    EmpiricalDataProblem('abalone',          'Rings',
                         x_transforms=onehot_non_numeric + (PolynomialExpansion(3),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('crime',            'ViolentCrimesPerPop',
                         drop=('state', 'fold', 'communityname'), nan_policy='drop_cols',
                         x_transforms=(PolynomialExpansion(3),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('naval_propulsion', ('GT_compressor_decay', 'GT_turbine_decay'),
                         x_transforms=(PolynomialExpansion(3),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('parkinsons',       ('motor_UPDRS', 'total_UPDRS'),
                         x_transforms=(PolynomialExpansion(3),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('student',          ('G1', 'G2', 'G3'),
                         x_transforms=onehot_non_numeric + (PolynomialExpansion(3),),
                         zero_variance_filter=True),
]

JOURNAL2026_D3_REGULAR = JOURNAL2026_D3_TINY + JOURNAL2026_D3_SMALL + JOURNAL2026_D3_MEDIUM
JOURNAL2026_D3 = JOURNAL2026_D3_REGULAR

# ── Estimators ───────────────────────────────────────────────────────────────

JOURNAL2026_ESTIMATORS = [RidgeEM(), RidgeLOOCV(alphas=11), RidgeLOOCV(alphas=101)]
JOURNAL2026_EST_NAMES  = ['EM', 'CV_glm_11', 'CV_glm_101']
