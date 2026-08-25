"""
Regression tests for C-3 (audit finding, resource-exhaustion stopgap half):
seven of this router's nine file-accepting endpoints previously read the
entire request body via `await file.read()` before checking its size at
all. Only POST /image had a pre-read Content-Length guard. This adds the
same guard to the other eight, plus two endpoint-specific gaps the audit
flagged separately (/segment's wrong ceiling and missing file.close(),
/batch's total absence of any per-file size check in the route handler).

These tests prove ordering, not just behavior: each endpoint is called
directly (bypassing the ASGI/TestClient stack) with a real Starlette
Request carrying a deliberately oversized Content-Length header, and a
mock UploadFile whose .read() raises AssertionError if it is ever called.
A clean HTTP 413 with no AssertionError proves the size check fires
strictly BEFORE the body is read -- the actual resource-exhaustion
concern this finding raises.
"""
import asyncio

import pytest
from starlette.requests import Request
from fastapi import HTTPException

from backend.api.routes.analyze import (
    _reject_if_content_length_exceeds,
    analyze_image_heatmap,
    analyze_attribution,
    analyze_platform,
    analyze_c2pa,
    analyze_robustness,
    analyze_segment,
    analyze_image_stream,
    export_report,
    MAX_ANALYSIS_SIZE_BYTES,
)


_client_ip_counter = [0]


def _make_request(content_length: str | None) -> Request:
    # Each call gets its own fake client IP so slowapi's per-IP rate
    # limiter (keyed on remote address) doesn't let one test's calls
    # count against another, unrelated test's quota.
    _client_ip_counter[0] += 1
    fake_ip = f"10.0.0.{_client_ip_counter[0] % 250 + 1}"

    headers = []
    if content_length is not None:
        headers.append((b"content-length", content_length.encode()))
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/analyze/test",
        "headers": headers,
        "query_string": b"",
        "client": (fake_ip, 12345),
        "server": ("testserver", 80),
        "scheme": "http",
        "app": None,
    }
    return Request(scope)


class ExplodingUploadFile:
    """An UploadFile stand-in whose .read() fails the test if it is ever
    called -- used to prove a size guard fires strictly before any read."""

    filename = "big.jpg"
    content_type = "image/jpeg"

    async def read(self, *a, **k):
        raise AssertionError(
            "file.read() was called -- the pre-read Content-Length guard "
            "did not fire before the body was read."
        )

    async def close(self):
        pass


class TestHelperUnit:
    def test_no_content_length_header_passes(self):
        req = _make_request(None)
        _reject_if_content_length_exceeds(req, 1000)  # must not raise

    def test_under_limit_passes(self):
        req = _make_request("500")
        _reject_if_content_length_exceeds(req, 1000)  # must not raise

    def test_over_limit_raises_413(self):
        req = _make_request("999999999")
        with pytest.raises(HTTPException) as exc_info:
            _reject_if_content_length_exceeds(req, 1000)
        assert exc_info.value.status_code == 413

    def test_invalid_content_length_fails_open(self):
        """Matches the pre-existing /image behavior: an unparseable
        Content-Length header must not itself crash the request -- the
        post-read size check remains the backstop."""
        req = _make_request("not-a-number")
        _reject_if_content_length_exceeds(req, 1000)  # must not raise


@pytest.mark.parametrize(
    "endpoint_fn",
    [
        analyze_image_heatmap,
        analyze_attribution,
        analyze_platform,
        analyze_c2pa,
        analyze_robustness,
        analyze_image_stream,
    ],
)
def test_precheck_fires_before_read(endpoint_fn):
    """For each previously-unguarded endpoint: an oversized declared
    Content-Length must produce HTTP 413 WITHOUT ever calling
    file.read() -- proving the check is a genuine pre-read guard, not
    just a relabeled post-read check."""
    req = _make_request(str(MAX_ANALYSIS_SIZE_BYTES + 1))
    upload = ExplodingUploadFile()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(endpoint_fn(req, upload))

    assert exc_info.value.status_code == 413


def test_export_endpoint_precheck_fires_before_read():
    """export_report has an extra `fmt` path parameter, so it's tested
    separately from the parametrized group above."""
    req = _make_request(str(MAX_ANALYSIS_SIZE_BYTES + 1))
    upload = ExplodingUploadFile()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(export_report(req, "json", upload))

    assert exc_info.value.status_code == 413


def test_segment_precheck_fires_before_read():
    """/segment gets the same treatment, plus (separately, below) its
    ceiling is confirmed to be the intended 10MB, not the general 50MB
    validate_file() ceiling it previously fell back on."""
    req = _make_request(str(MAX_ANALYSIS_SIZE_BYTES + 1))
    upload = ExplodingUploadFile()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(analyze_segment(req, upload))

    assert exc_info.value.status_code == 413


def test_segment_post_read_ceiling_is_10mb_not_50mb():
    """Before this fix, /segment's only size guard was validate_file()'s
    general-purpose 50MB ceiling -- five times higher than every sibling
    CPU-heavy analysis endpoint's intended 10MB limit. This test builds a
    real UploadFile-like object just over 10MB (well under the old 50MB
    ceiling) with no Content-Length header (so the pre-read guard can't
    catch it either) and confirms the POST-read check now rejects it at
    the correct, intended ceiling."""

    class OversizedUpload:
        filename = "big.jpg"
        content_type = "image/jpeg"
        _data = b"\x00" * (MAX_ANALYSIS_SIZE_BYTES + 1024)

        async def read(self, *a, **k):
            return self._data

        async def close(self):
            pass

    req = _make_request(None)  # no Content-Length header -- forces the post-read check
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(analyze_segment(req, OversizedUpload()))

    assert exc_info.value.status_code == 413
    assert "10" in exc_info.value.detail or "MB" in exc_info.value.detail


class TestBatchEndpoint:
    def test_batch_rejects_oversized_total_content_length_before_reading(self):
        from backend.api.routes.analyze import analyze_batch
        from backend.services.batch_processor import MAX_BATCH_SIZE, MAX_IMAGE_BYTES

        req = _make_request(str(MAX_BATCH_SIZE * MAX_IMAGE_BYTES + 1))
        files = [ExplodingUploadFile()]

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(analyze_batch(req, files))

        assert exc_info.value.status_code == 413

    def test_batch_skips_individually_oversized_file_after_read(self):
        """Before this fix, /batch had NO per-file size check in the route
        handler at all -- the 5MB MAX_IMAGE_BYTES cap only existed inside
        batch_processor.process_batch(), reached only after every file in
        the batch was already fully buffered. This confirms an
        individually-oversized file is now skipped (not silently
        forwarded) immediately after its own read, before any other file
        in the batch is processed further."""
        from backend.services.batch_processor import MAX_IMAGE_BYTES

        class OversizedFile:
            filename = "toobig.jpg"
            content_type = "image/jpeg"
            _data = b"\x00" * (MAX_IMAGE_BYTES + 1024)

            async def read(self, *a, **k):
                return self._data

            async def close(self):
                pass

        from backend.api.routes.analyze import analyze_batch

        # A batch of only oversized files should hit the
        # "No valid image files in batch." 415, since every file gets
        # skipped rather than silently accepted.
        req = _make_request(None)  # no Content-Length header -- forces the per-file check
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(analyze_batch(req, [OversizedFile()]))
        assert exc_info.value.status_code == 415
