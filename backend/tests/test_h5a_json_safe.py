"""
Regression tests for H-5, part A (audit finding): a NaN/Infinity-safe
sanitizer existed as two independently-maintained, near-duplicate copies
(backend/api/routes/analyze.py's module-level _sanitize(), and a local
function inside backend/services/sse_analyzer.py) despite analyze.py's
own docstring explicitly warning this project has been bitten by exactly
this kind of drift before. The two copies had already drifted: the
sse_analyzer.py copy was missing the numpy-scalar-to-native-type
conversion the analyze.py copy has.

Both now import backend.utils.json_safe.sanitize -- this tests that
shared implementation, and confirms both call sites actually use it
(not a second, still-separate copy that happens to look similar).
"""
import math

import numpy as np
import pytest

from backend.utils.json_safe import sanitize


class TestSanitizeBasics:
    def test_replaces_nan_with_zero(self):
        assert sanitize(float("nan")) == 0.0

    def test_replaces_positive_infinity_with_zero(self):
        assert sanitize(float("inf")) == 0.0

    def test_replaces_negative_infinity_with_zero(self):
        assert sanitize(float("-inf")) == 0.0

    def test_leaves_normal_float_unchanged(self):
        assert sanitize(0.7182) == 0.7182

    def test_leaves_int_unchanged(self):
        assert sanitize(42) == 42

    def test_leaves_string_unchanged(self):
        assert sanitize("classification") == "classification"

    def test_leaves_none_unchanged(self):
        assert sanitize(None) is None


class TestSanitizeRecursion:
    def test_sanitizes_nested_dict(self):
        obj = {"score": float("nan"), "nested": {"confidence": float("inf")}}
        result = sanitize(obj)
        assert result == {"score": 0.0, "nested": {"confidence": 0.0}}

    def test_sanitizes_list_of_dicts(self):
        obj = [{"score": float("nan")}, {"score": 0.5}]
        result = sanitize(obj)
        assert result == [{"score": 0.0}, {"score": 0.5}]

    def test_sanitizes_deeply_nested_mixed_structure(self):
        obj = {
            "signals": [
                {"score": float("nan"), "confidence": 0.8},
                {"score": 0.3, "confidence": float("-inf")},
            ],
            "summary": {"total": 2, "avg": float("inf")},
        }
        result = sanitize(obj)
        assert result["signals"][0]["score"] == 0.0
        assert result["signals"][1]["confidence"] == 0.0
        assert result["summary"]["avg"] == 0.0
        assert result["summary"]["total"] == 2


class TestSanitizeNumpyScalars:
    """This is the specific case the old sse_analyzer.py copy was
    missing -- confirmed by direct comparison with the original
    (pre-fix) local function, which only handled float/dict/list."""

    def test_converts_numpy_float64_to_native_float(self):
        result = sanitize(np.float64(0.42))
        assert result == 0.42
        assert isinstance(result, float)
        assert not isinstance(result, np.generic)

    def test_converts_numpy_nan_to_zero(self):
        result = sanitize(np.float64("nan"))
        assert result == 0.0

    def test_converts_numpy_int_to_native_int(self):
        result = sanitize(np.int64(7))
        assert result == 7
        assert not isinstance(result, np.generic)

    def test_converts_numpy_scalar_nested_in_dict(self):
        obj = {"score": np.float32(0.9), "count": np.int32(3)}
        result = sanitize(obj)
        assert result == {"score": pytest.approx(0.9, abs=1e-6), "count": 3}
        assert not isinstance(result["score"], np.generic)
        assert not isinstance(result["count"], np.generic)


class TestBothCallSitesUseTheSharedImplementation:
    """Confirms analyze.py and sse_analyzer.py aren't just two modules
    that both happen to define a matching function -- they import the
    literal same one."""

    def test_analyze_py_sanitize_is_the_shared_function(self):
        from backend.api.routes.analyze import _sanitize as analyze_sanitize
        from backend.utils.json_safe import sanitize as shared_sanitize

        assert analyze_sanitize is shared_sanitize

    def test_sse_analyzer_py_sanitize_is_the_shared_function(self):
        import backend.services.sse_analyzer as sse_module
        from backend.utils.json_safe import sanitize as shared_sanitize

        assert sse_module._sanitize is shared_sanitize
