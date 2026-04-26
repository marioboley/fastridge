import json
import os
import tempfile
import warnings
import inspect
import importlib
import sys
import platform
from typing import Any

import numpy as np


def to_json(obj, include_computed=False, include_defaults=False):
    """Serialise obj to a JSON-native Python object.

    include_computed controls which trailing-underscore attributes are appended
    to Transparent Class output: False (none), True (all in obj.__dict__), or
    list[str] (named subset). Does not propagate into recursive calls.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [to_json(item) for item in obj]
    if isinstance(obj, tuple):
        return {'__tuple__': [to_json(item) for item in obj]}
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items()}
    if is_importable(obj):
        return {'__import__': f'{obj.__module__}.{obj.__qualname__}'}
    cls = type(obj)
    cls_ref = f'{cls.__module__}.{cls.__qualname__}'
    params = {name: to_json(val) for name, val in init_params(obj, include_defaults).items()}
    computed = {k: to_json(getattr(obj, k)) for k in _computed_keys(obj, include_computed)}
    return {'__class__': cls_ref, **params, **computed}


def from_json(data):
    """Reconstruct a Python object from the output of json.loads().

    Input is always one of None, bool, int, float, str, list, or dict.
    """
    if data is None or isinstance(data, (bool, int, float, str)):
        return data
    if isinstance(data, list):
        return [from_json(item) for item in data]
    if isinstance(data, dict):
        if '__tuple__' in data:
            return tuple(from_json(item) for item in data['__tuple__'])
        if '__import__' in data:
            module, _, name = data['__import__'].rpartition('.')
            return getattr(importlib.import_module(module), name)
        if '__class__' in data:
            module, _, name = data['__class__'].rpartition('.')
            cls = getattr(importlib.import_module(module), name)
            init_kwargs = {k: from_json(v) for k, v in data.items()
                          if k != '__class__' and not k.endswith('_')}
            obj = cls(**init_kwargs)
            for k, v in data.items():
                if k.endswith('_') and not k.startswith('_'):
                    setattr(obj, k, from_json(v))
            return obj
        return {k: from_json(v) for k, v in data.items()}


def save_json(path, data, indent: int | str | None = 2):
    """Write data as JSON to path atomically via tempfile + os.replace.

    Creates parent directories as needed. indent=None produces compact
    single-line output. On write failure emits UserWarning rather than raising.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=indent)
        os.replace(tmp, path)
    except Exception as e:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        warnings.warn(f'JSON write failed for {path}: {e}')


def load_json(path, default=None) -> Any:
    """Read and deserialise a JSON file via from_json; return default if missing."""
    try:
        with open(path) as f:
            return from_json(json.load(f))
    except FileNotFoundError:
        return default


def is_importable(obj):
    """Return True if obj is accessible by name from its declared module and can be reconstructed via importlib.

    >>> import numpy as np
    >>> is_importable(np.log)
    True
    >>> is_importable(lambda x: x)
    False
    """
    try:
        return getattr(importlib.import_module(obj.__module__), obj.__qualname__) is obj
    except Exception:
        return False


def safe_equals(a, b):
    """Return True if a and b are equal, using np.array_equal (with equal_nan=True) for ndarray inputs.

    >>> import numpy as np
    >>> safe_equals(np.array([1, 2]), np.array([1, 2]))
    True
    >>> safe_equals(np.array([1, 2]), np.array([1, 2, 3]))
    False
    >>> safe_equals(np.array([np.nan]), np.array([np.nan]))
    True
    """
    try:
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return np.array_equal(a, b, equal_nan=True)
        else:
            return a == b
    except Exception:
        return False


def init_params(obj, include_defaults=True):
    """Return {name: value} for each __init__ parameter found as a same-named attribute on obj.

    >>> class Point:
    ...     def __init__(self, x, y=0): self.x = x; self.y = y
    >>> init_params(Point(3))
    {'x': 3, 'y': 0}
    >>> init_params(Point(3), include_defaults=False)
    {'x': 3}
    """
    cls = type(obj)
    result = {}
    VAR = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    for name, p in inspect.signature(cls.__init__).parameters.items():
        if name == 'self' or p.kind in VAR:
            continue
        try:
            value = getattr(obj, name)
            if include_defaults or p.default is inspect.Parameter.empty or not safe_equals(value, p.default):
                result[name] = value
        except AttributeError:
            warnings.warn(
                f'{cls.__name__}.{name}: init parameter not found as attribute; '
                'omitted from serialisation')
    return result


def _computed_keys(obj, include_computed):
    if include_computed is False:
        return []
    all_computed = [k for k in obj.__dict__ if k.endswith('_')]
    if include_computed is True:
        return all_computed
    return [k for k in all_computed if k in include_computed]


def environment():
    return {
                'python': sys.version.split()[0],
                'platform': platform.platform(),
                'numpy': np.version.version
            }