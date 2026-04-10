"""
Dataset registry and retrieval for real data experiments.

>>> df = get_dataset('yacht')
>>> df.shape
(308, 7)
>>> df = get_dataset('diabetes')
>>> df.shape
(442, 11)
"""
import io
import urllib.request
import warnings
import zipfile
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).parent.parent / 'datasets'


def from_sklearn(loader_fn):
    """Return a source callable that loads a sklearn dataset as a full DataFrame.

    The returned callable takes no arguments and returns a pd.DataFrame
    containing all feature columns and the target column.

    >>> from sklearn.datasets import load_diabetes
    >>> src = from_sklearn(load_diabetes)
    >>> src().shape
    (442, 11)
    """
    def source():
        bunch = loader_fn(as_frame=True)
        return bunch.frame
    return source


def from_ucimlrepo(id):
    """Return a source callable that fetches a dataset from the UCI ML Repository.

    The returned callable takes no arguments and returns a pd.DataFrame
    with features and targets concatenated column-wise.
    """
    def source():
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=id)
        return pd.concat([ds.data.features, ds.data.targets], axis=1)
    return source


def from_url(url, **read_csv_kwargs):
    """Return a source callable that downloads a CSV from the given URL.

    Extra keyword arguments are forwarded to pd.read_csv.
    """
    def source():
        return pd.read_csv(url, **read_csv_kwargs)
    return source


def from_zip(url, entry, **read_csv_kwargs):
    """Return a source callable that downloads a ZIP and reads one entry as a DataFrame.

    Extra keyword arguments are forwarded to pd.read_csv.
    """
    def source():
        with urllib.request.urlopen(url) as resp:
            zf = zipfile.ZipFile(io.BytesIO(resp.read()))
        data = zf.read(entry).decode('latin-1')
        return pd.read_csv(io.StringIO(data), **read_csv_kwargs)
    return source


from sklearn.datasets import load_diabetes  # noqa: E402

DATASETS = {
    'abalone':          {'sources': [from_ucimlrepo(1)]},
    'autompg':          {'sources': [from_ucimlrepo(9)]},
    'automobile':       {'sources': [from_ucimlrepo(10)]},
    'airfoil':          {'sources': [from_ucimlrepo(291)]},
    'boston':           {'sources': [from_url('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')]},
    'crime':            {'sources': [from_ucimlrepo(183)]},
    'concrete':         {'sources': [from_ucimlrepo(165)]},
    'naval_propulsion': {'sources': [
        from_ucimlrepo(316),
        from_zip(
            'https://cdn.uci-ics-mlr-prod.aws.uci.edu/316/'
            'condition%2Bbased%2Bmaintenance%2Bof%2Bnaval%2Bpropulsion%2Bplants.zip',
            'UCI CBM Dataset/data.txt',
            sep=r'\s+', header=None,
            names=['lp', 'v', 'GTT', 'GTn', 'GGn', 'Ts', 'Tp', 'T48', 'T1', 'T2',
                   'P48', 'P1', 'P2', 'Pexh', 'TIC', 'mf',
                   'GT_compressor_decay', 'GT_turbine_decay'],
        ),
    ]},
    'diabetes':         {'sources': [from_sklearn(load_diabetes)]},
    'eye':              {'sources': []},  # URL TBD
    'facebook':         {'sources': [from_ucimlrepo(368)]},
    'forest':           {'sources': [from_ucimlrepo(162)]},
    'parkinsons':       {'sources': [from_ucimlrepo(189)]},
    'real_estate':      {'sources': [from_ucimlrepo(477)]},
    'student':          {'sources': [from_ucimlrepo(320)]},
    'yacht':            {'sources': [from_url('https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data',
                                              sep=r'\s+', header=None,
                                              names=['Longitudinal_position', 'Prismatic_coefficient',
                                                     'Length_displacement_ratio', 'Beam_draught_ratio',
                                                     'Length_beam_ratio', 'Froude_number',
                                                     'Residuary_resistance'])]},
    'ribo':             {'sources': []},  # URL TBD
    'crop':             {'sources': []},  # URL TBD
    'elec_devices':     {'sources': []},  # URL TBD
    'starlight':        {'sources': []},  # URL TBD
}


def get_dataset(name):
    """Retrieve a dataset by name, using the local cache if available.

    Checks datasets/<name>.csv first. If absent, tries each source callable
    in REGISTRY[name]['sources'] in order, persists the result to
    datasets/<name>.csv on success, and returns the DataFrame.

    Raises KeyError for unknown names.
    Raises RuntimeError if all sources fail and no cache file exists.

    >>> df = get_dataset('yacht')
    >>> df.shape
    (308, 7)
    >>> df = get_dataset('diabetes')
    >>> df.shape
    (442, 11)
    """
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset: '{name}'. Available: {list(DATASETS)}")

    cache_path = CACHE_DIR / f'{name}.csv'
    if cache_path.exists():
        return pd.read_csv(cache_path)

    sources = DATASETS[name]['sources']
    last_exc = None
    for source in sources:
        try:
            df = source()
            CACHE_DIR.mkdir(exist_ok=True)
            df.to_csv(cache_path, index=False)
            return df
        except Exception as exc:
            last_exc = exc

    raise RuntimeError(
        f"All sources failed for dataset '{name}'."
        + (f" Last error: {last_exc}" if last_exc else " No sources configured.")
    )
