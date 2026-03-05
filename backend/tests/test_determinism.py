"""
Determinism and reproducibility tests.
"""
import pytest
import numpy as np


def test_detection_is_deterministic(sample_image_bytes):
    """Test that statistical detection is deterministic."""
    from backend.services.statistical_detector import StatisticalDetector
    
    # Statistical detector (without CLIP randomness)
    detector1 = StatisticalDetector(sample_image_bytes, "test.png")
    report1 = detector1.detect()
    
    detector2 = StatisticalDetector(sample_image_bytes, "test.png")
    report2 = detector2.detect()
    
    # Statistical results should be identical
    assert report1["ai_probability"] == report2["ai_probability"]
    assert report1["classification"] == report2["classification"]
    assert report1["total_signals"] == report2["total_signals"]


def test_hash_generation_is_consistent(sample_image_bytes):
    """Test that hash generation is deterministic."""
    from backend.services.image_forensics import ImageForensics
    
    forensics1 = ImageForensics(sample_image_bytes, "test.png")
    hashes1 = forensics1.generate_hashes()
    
    forensics2 = ImageForensics(sample_image_bytes, "test.png")
    hashes2 = forensics2.generate_hashes()
    
    assert hashes1["sha256"] == hashes2["sha256"]
    assert hashes1["md5"] == hashes2["md5"]
    assert hashes1["perceptual_hash"] == hashes2["perceptual_hash"]


def test_forensic_report_stability(sample_image_bytes):
    """Test forensic report stability (allowing CLIP variance)."""
    from backend.services.image_forensics import ImageForensics
    
    forensics1 = ImageForensics(sample_image_bytes, "test.png")
    report1 = forensics1.generate_forensic_report()
    
    forensics2 = ImageForensics(sample_image_bytes, "test.png")
    report2 = forensics2.generate_forensic_report()
    
    # Core forensic data should match
    assert report1["hashes"]["sha256"] == report2["hashes"]["sha256"]
    assert report1["file_info"]["width"] == report2["file_info"]["width"]
    
    # AI probability may vary due to CLIP random initialization
    # Allow 15% variance (CLIP contributes 25% to ensemble, so ~12% max variance expected)
    prob_diff = abs(report1["summary"]["ai_probability"] - report2["summary"]["ai_probability"])
    assert prob_diff < 0.15, f"AI probability variance too high: {prob_diff}"


def test_cache_consistency(client, sample_image_bytes):
    """Test cache returns consistent results."""
    files = {"file": ("test.png", sample_image_bytes, "image/png")}
    
    response1 = client.post("/api/v1/analyze/image", files=files)
    data1 = response1.json()
    
    response2 = client.post("/api/v1/analyze/image", files=files)
    data2 = response2.json()
    
    # Cached results should be identical
    assert data1["summary"]["ai_probability"] == data2["summary"]["ai_probability"]
    assert data1["hashes"]["sha256"] == data2["hashes"]["sha256"]


def test_signal_ordering_is_stable(sample_image_bytes):
    """Test that signal ordering is consistent."""
    from backend.services.advanced_ensemble_detector import AdvancedEnsembleDetector
    
    detector1 = AdvancedEnsembleDetector(sample_image_bytes, "test.png")
    report1 = detector1.detect()
    
    detector2 = AdvancedEnsembleDetector(sample_image_bytes, "test.png")
    report2 = detector2.detect()
    
    # Signal names should appear in same order
    names1 = [s["signal_name"] for s in report1["all_signals"]]
    names2 = [s["signal_name"] for s in report2["all_signals"]]
    
    assert names1 == names2
    assert len(names1) == 21  # Should have 21 signals
    
    detector1.cleanup()
    detector2.cleanup()
