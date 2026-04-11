"""
Dataset registry and retrieval for real data experiments.

>>> df = get_dataset('yacht')
>>> df.shape
(308, 7)
>>> df = get_dataset('diabetes')
>>> df.shape
(442, 11)
>>> from data import fetch_riboflavin
>>> df_ribo = fetch_riboflavin()
>>> df_ribo.shape
(71, 4089)
>>> list(df_ribo.columns[:3])
['y', 'AADK_at', 'AAPA_at']
>>> df_ribo['y'].dtype.kind
'f'
"""
import io
import tarfile
import tempfile
import urllib.request
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
# rdata is imported conditionally inside _fetch_cran_rdata and fetch_riboflavin

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


def _fetch_cran_rdata(package, version, rdata_path):
    """Download a CRAN package tarball and return the parsed R object at rdata_path.

    Fetches https://cran.r-project.org/src/contrib/{package}_{version}.tar.gz,
    extracts the file at rdata_path, parses it with rdata.parser.parse_file,
    and returns the parsed R object (not yet converted to a DataFrame).
    """
    import rdata
    url = f'https://cran.r-project.org/src/contrib/{package}_{version}.tar.gz'
    raw = urllib.request.urlopen(url).read()
    with tarfile.open(fileobj=io.BytesIO(raw)) as t:
        f = t.extractfile(rdata_path)
        tmp = tempfile.NamedTemporaryFile(suffix='.RData', delete=False)
        tmp.write(f.read())
        tmp.close()
    return rdata.parser.parse_file(tmp.name)


def from_cran(package, version, rdata_path):
    """Return a source callable that fetches an R dataset from a CRAN package.

    The RData file at rdata_path must contain a single flat R data.frame that
    rdata.conversion.convert() can map directly to a pandas DataFrame.
    For datasets with non-standard R structures (e.g. matrix-valued columns),
    write a dedicated source callable that calls _fetch_cran_rdata directly.
    """
    def source():
        import rdata
        parsed = _fetch_cran_rdata(package, version, rdata_path)
        converted = rdata.conversion.convert(parsed)
        return converted[list(converted.keys())[0]]
    return source


def fetch_riboflavin():
    """Fetch the riboflavin dataset from the hdi CRAN package (version 0.1-10).

    Returns a DataFrame with 71 rows and 4089 columns: 'y' (log riboflavin
    production rate) followed by 4088 gene expression columns (e.g. 'AADK_at').

    The hdi RData structure is a data.frame with a matrix-valued column x
    (71x4088) and a scalar column y — not convertible by rdata's standard
    converter, hence the custom unpacking below.
    """
    from rdata.parser._parser import RObjectType
    parsed = _fetch_cran_rdata('hdi', '0.1-10', 'hdi/data/riboflavin.RData')
    robj = parsed.object.value[0]
    y_robj = robj.value[0]   # REAL vector, length 71
    x_robj = robj.value[1]   # REAL vector, length 71*4088, with dim/dimnames attrs

    def _pairlist_to_dict(obj):
        result = {}
        cur = obj
        while cur is not None and cur.info.type not in (RObjectType.NILVALUE,):
            if cur.info.type == RObjectType.LIST:
                tag = cur.tag
                if tag is not None:
                    name = tag.value.value
                    if isinstance(name, bytes):
                        name = name.decode()
                    result[name] = cur.value[0]
                cur = cur.value[1] if len(cur.value) > 1 else None
            else:
                break
        return result

    def _extract_strvec(obj):
        if obj.info.type in (RObjectType.VEC, RObjectType.STR):
            return [v.value.decode() if isinstance(v.value, bytes) else str(v.value)
                    for v in obj.value]
        return []

    x_attrs = _pairlist_to_dict(x_robj.attributes)
    dim = [int(v) for v in x_attrs['dim'].value]
    dimnames = x_attrs['dimnames'].value
    rownames = _extract_strvec(dimnames[0])
    colnames = _extract_strvec(dimnames[1])

    x = pd.DataFrame(
        np.array(x_robj.value).reshape(dim, order='F'),
        index=rownames,
        columns=colnames,
    )
    y = pd.Series(np.array(y_robj.value), index=rownames, name='y')
    return pd.concat([y, x], axis=1).reset_index(drop=True)


from sklearn.datasets import load_diabetes  # noqa: E402

DATASETS = {
    'abalone':          {'sources': [from_ucimlrepo(1)]},
    'autompg':          {'sources': [
        from_ucimlrepo(9),
        from_zip(
            'https://cdn.uci-ics-mlr-prod.aws.uci.edu/9/auto%2Bmpg.zip',
            'auto-mpg.data',
            sep=r'\s+', header=None, na_values='?',
            names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                   'acceleration', 'model_year', 'origin', 'car_name'],
        ),
        # TODO: verify whether from_ucimlrepo includes car_name in features; if not,
        # sources are inconsistent and car_name should be dropped at source level.
        # Also: ucimlrepo has been unavailable frequently — review all ucimlrepo sources
        # and add CDN ZIP fallbacks where possible.
    ]},
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
    'student':          {'sources': [from_ucimlrepo(320)]},  # Portuguese only; see docs/superpowers/issues/2026-04-11-student-dataset-structure.md
    'yacht':            {'sources': [from_url('https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data',
                                              sep=r'\s+', header=None,
                                              names=['Longitudinal_position', 'Prismatic_coefficient',
                                                     'Length_displacement_ratio', 'Beam_draught_ratio',
                                                     'Length_beam_ratio', 'Froude_number',
                                                     'Residuary_resistance'])]},
    'ribo':             {'sources': [fetch_riboflavin]},
    'crop':             {'sources': []},  # URL TBD
    'elec_devices':     {'sources': []},  # URL TBD
    'starlight':        {'sources': []},  # URL TBD
}


# Additional missing value markers beyond pandas defaults:
# '', 'NA', 'N/A', '#N/A', '#N/A N/A', '#NA', 'NaN', '-NaN', '-nan', 'nan',
# 'None', '<NA>', 'NULL', 'null', 'n/a', '1.#IND', '-1.#IND', '1.#QNAN', '-1.#QNAN'
_EXTRA_NA_VALUES = ['?']


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
    if not cache_path.exists():
        sources = DATASETS[name]['sources']
        last_exc = None
        for source in sources:
            try:
                df = source()
                CACHE_DIR.mkdir(exist_ok=True)
                df.to_csv(cache_path, index=False)
                break
            except Exception as exc:
                last_exc = exc
        else:
            raise RuntimeError(
                f"All sources failed for dataset '{name}'."
                + (f" Last error: {last_exc}" if last_exc else " No sources configured.")
            )

    return pd.read_csv(cache_path, na_values=_EXTRA_NA_VALUES)
