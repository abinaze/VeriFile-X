"""
Determinism and reproducibility tests.
"""
import pytest
import numpy as np


def test_detection_is_deterministic(sample_image_bytes):
    """Test that running detection twice gives identical results."""
    from backend.services.statistical_detector import StatisticalDetector
    
    # First run
    detector1 = StatisticalDetector(sample_image_bytes, "test.png")
    report1 = detector1.detect()
    
    # Second run
    detector2 = StatisticalDetector(sample_image_bytes, "test.png")
    report2 = detector2.detect()
    
    # Core results should be identical
    assert report1["ai_probability"] == report2["ai_probability"]
    assert report1["classification"] == report2["classification"]
    assert report1["total_signals"] == report2["total_signals"]
    
    # All signal scores should match
    for sig1, sig2 in zip(report1["all_signals"], report2["all_signals"]):
        assert sig1["signal_name"] == sig2["signal_name"]
        # Allow tiny floating-point differences
        assert abs(sig1["score"] - sig2["score"]) < 1e-6


def test_hash_generation_is_consistent(sample_image_bytes):
    """Test that hash generation is deterministic."""
    from backend.services.image_forensics import ImageForensics
    
    # Generate hashes twice
    forensics1 = ImageForensics(sample_image_bytes, "test.png")
    hashes1 = forensics1.generate_hashes()
    
    forensics2 = ImageForensics(sample_image_bytes, "test.png")
    hashes2 = forensics2.generate_hashes()
    
    # All hashes should be identical
    assert hashes1["sha256"] == hashes2["sha256"]
    assert hashes1["md5"] == hashes2["md5"]
    assert hashes1["perceptual_hash"] == hashes2["perceptual_hash"]


def test_forensic_report_stability(sample_image_bytes):
    """Test complete forensic report is stable."""
    from backend.services.image_forensics import ImageForensics
    
    forensics1 = ImageForensics(sample_image_bytes, "test.png")
    report1 = forensics1.generate_forensic_report()
    
    forensics2 = ImageForensics(sample_image_bytes, "test.png")
    report2 = forensics2.generate_forensic_report()
    
    # Core forensic data should match
    assert report1["summary"]["ai_probability"] == report2["summary"]["ai_probability"]
    assert report1["hashes"]["sha256"] == report2["hashes"]["sha256"]
    assert report1["file_info"]["width"] == report2["file_info"]["width"]


def test_cache_consistency(client, sample_image_bytes):
    """Test cache returns consistent results."""
    files = {"file": ("test.png", sample_image_bytes, "image/png")}
    
    # First request
    response1 = client.post("/api/v1/analyze/image", files=files)
    data1 = response1.json()
    
    # Second request (cached)
    response2 = client.post("/api/v1/analyze/image", files=files)
    data2 = response2.json()
    
    # Results should be identical
    assert data1["summary"]["ai_probability"] == data2["summary"]["ai_probability"]
    assert data1["hashes"]["sha256"] == data2["hashes"]["sha256"]


def test_signal_ordering_is_stable(sample_image_bytes):
    """Test that signal ordering is consistent across runs."""
    from backend.services.statistical_detector import StatisticalDetector
    
    detector1 = StatisticalDetector(sample_image_bytes, "test.png")
    report1 = detector1.detect()
    
    detector2 = StatisticalDetector(sample_image_bytes, "test.png")
    report2 = detector2.detect()
    
    # Signal names should appear in same order
    names1 = [s["signal_name"] for s in report1["all_signals"]]
    names2 = [s["signal_name"] for s in report2["all_signals"]]
    
    assert names1 == names2
