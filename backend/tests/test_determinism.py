"""
Tests for system determinism and reproducibility.
"""
import pytest
from backend.services.image_forensics import ImageForensics


def test_detection_is_deterministic(sample_image_bytes):
    """Test that detection produces same results across runs."""
    forensics1 = ImageForensics(sample_image_bytes, "test.png")
    forensics2 = ImageForensics(sample_image_bytes, "test.png")
    
    result1 = forensics1._ai_detector.detect()
    result2 = forensics2._ai_detector.detect()
    
    # AI probability should be identical
    assert result1['ai_probability'] == result2['ai_probability']
    
    # Classification should be identical
    assert result1['classification'] == result2['classification']


def test_hash_generation_is_consistent(sample_image_bytes):
    """Test that hash generation is deterministic."""
    forensics = ImageForensics(sample_image_bytes, "test.png")
    
    hashes1 = forensics.generate_hashes()
    hashes2 = forensics.generate_hashes()
    
    # All hashes should match
    assert hashes1['md5'] == hashes2['md5']
    assert hashes1['sha256'] == hashes2['sha256']
    assert hashes1['perceptual'] == hashes2['perceptual']


def test_forensic_report_stability(sample_image_bytes):
    """Test that full forensic reports are stable across runs."""
    forensics1 = ImageForensics(sample_image_bytes, "test.png")
    forensics2 = ImageForensics(sample_image_bytes, "test.png")
    
    report1 = forensics1.generate_forensic_report()
    report2 = forensics2.generate_forensic_report()
    
    # Hashes should be identical
    assert report1["hashes"]["md5"] == report2["hashes"]["md5"]
    assert report1["hashes"]["sha256"] == report2["hashes"]["sha256"]
    
    # AI probability should be close (allow 20% variance for CLIP randomness)
    # CLIP uses random placeholder centroids until database is built
    ai_prob_1 = report1["summary"]["ai_probability"]
    ai_prob_2 = report2["summary"]["ai_probability"]
    variance = abs(ai_prob_1 - ai_prob_2) / max(ai_prob_1, ai_prob_2)
    
    assert variance < 0.20, f"AI probability variance too high: {variance:.3f} (prob1={ai_prob_1:.3f}, prob2={ai_prob_2:.3f})"
    
    # Signal counts should be identical
    assert report1["summary"]["total_detection_signals"] == report2["summary"]["total_detection_signals"]


def test_cache_consistency(sample_image_bytes):
    """Test that result cache produces consistent results."""
    from backend.core.cache import result_cache
    
    # Clear cache
    result_cache.clear()
    
    forensics = ImageForensics(sample_image_bytes, "test.png")
    
    # First call (cache miss)
    hashes1 = forensics.generate_hashes()
    
    # Second call (should hit cache)
    hashes2 = forensics.generate_hashes()
    
    # Should be identical
    assert hashes1 == hashes2


def test_signal_ordering_is_stable(sample_image_bytes):
    """Test that signals appear in consistent order."""
    forensics1 = ImageForensics(sample_image_bytes, "test.png")
    forensics2 = ImageForensics(sample_image_bytes, "test.png")
    
    report1 = forensics1.generate_forensic_report()
    report2 = forensics2.generate_forensic_report()
    
    signals1 = report1["ai_detection"]["signals"]
    signals2 = report2["ai_detection"]["signals"]
    
    # Signal names should appear in same order
    names1 = [s["name"] for s in signals1]
    names2 = [s["name"] for s in signals2]
    
    assert names1 == names2
