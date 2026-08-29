"""
Shared JSON-safety helpers.

H-5 (audit finding, part A): a NaN/Infinity-safe sanitizer previously
existed as two independently-maintained, near-duplicate copies -- one in
backend/api/routes/analyze.py, one as a local function inside
backend/services/sse_analyzer.py -- despite analyze.py's own copy
explicitly documenting that it already replaced two *other* inline
duplicates for this exact reason (F-4/F-8). The drift this was meant to
prevent had already happened again: the sse_analyzer.py copy was missing
the numpy-scalar-to-native-type conversion the analyze.py copy has, so a
raw numpy scalar reaching the SSE report would not have been converted,
unlike the same case on the regular /image endpoint.

This module is the single implementation both now import.
"""
import math
from typing import Any

import numpy as np


def sanitize(obj: Any) -> Any:
    """Recursively replace NaN/Infinity floats with 0.0, and convert
    numpy scalar types to native Python types.

    json.dumps() happily emits the literal tokens NaN/Infinity, which
    are not valid per RFC 8259 and many non-Python JSON parsers reject
    outright. Division-by-zero guards throughout the signal detectors
    (e.g. "+ 1e-8" denominators in several cosine-similarity calcs) mean
    a signal can legitimately produce a non-finite float. Numpy scalar
    types (e.g. np.float64) are also not natively JSON-serializable.

    A numpy generic is converted to its native type FIRST, without
    returning immediately -- it then falls through to the float check
    below. Returning right after the numpy conversion (the original
    behavior in both prior duplicate copies of this function) meant a
    numpy NaN/Infinity, e.g. np.float64('nan'), was converted to a
    native Python float but never actually checked for being non-finite,
    silently defeating the very sanitization this function exists to
    do. Found by direct testing while consolidating the two duplicates
    for H-5, not part of that finding's original scope, but a real bug
    in code both duplicates shared -- fixed here rather than carried
    forward into the new shared implementation.
    """
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        return 0.0 if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj
