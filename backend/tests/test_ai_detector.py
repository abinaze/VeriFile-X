"""
Tests for AI-generated image detection.
"""
import pytest
from backend.services.ai_detector import AIDetector


def test_ai_detector_initialization(sample_image_bytes):
    """Test AI detector initializes correctly."""
    detector = AIDetector(sample_image_bytes, "test.png")
    assert detector.filename == "test.png"
    assert detector.cv_image is not None


def test_noise_analysis(sample_image_bytes):
    """Test noise pattern analysis."""
    detector = AIDetector(sample_image_bytes, "test.png")
    signals = detector.analyze_noise_patterns()
    
    assert "noise_variance" in signals
    assert "noise_consistency" in signals
    assert "suspicious" in signals
    assert isinstance(signals["suspicious"], bool)


def test_frequency_analysis(sample_image_bytes):
    """Test frequency domain analysis."""
    detector = AIDetector(sample_image_bytes, "test.png")
    signals = detector.analyze_frequency_domain()
    
    assert "frequency_ratio" in signals
    assert "spectral_entropy" in signals
    assert signals["spectral_entropy"] > 0


def test_jpeg_analysis(sample_image_bytes):
    """Test JPEG artifact analysis."""
    detector = AIDetector(sample_image_bytes, "test.png")
    signals = detector.analyze_jpeg_artifacts()
    
    assert "blockiness" in signals
    assert "edge_density" in signals
    assert signals["edge_density"] >= 0


def test_color_analysis(sample_image_bytes):
    """Test color distribution analysis."""
    detector = AIDetector(sample_image_bytes, "test.png")
    signals = detector.analyze_color_distribution()
    
    assert "color_entropy" in signals
    assert "mean_saturation" in signals
    assert signals["color_entropy"] > 0


def test_ai_probability_calculation(sample_image_bytes):
    """Test AI probability calculation."""
    detector = AIDetector(sample_image_bytes, "test.png")
    
    # Get all signals
    signals = {
        "noise": detector.analyze_noise_patterns(),
        "frequency": detector.analyze_frequency_domain(),
        "jpeg": detector.analyze_jpeg_artifacts(),
        "color": detector.analyze_color_distribution()
    }
    
    probability = detector.calculate_ai_probability(signals)
    
    assert 0 <= probability <= 1
    assert isinstance(probability, float)


def test_complete_detection(sample_image_bytes):
    """Test complete AI detection workflow."""
    detector = AIDetector(sample_image_bytes, "test.png")
    report = detector.detect()
    
    assert "ai_probability" in report
    assert "classification" in report
    assert "confidence" in report
    assert "detection_signals" in report
    
    # Check classification is valid
    valid_classifications = [
        "likely_ai_generated",
        "possibly_ai_generated",
        "likely_authentic"
    ]
    assert report["classification"] in valid_classifications


def test_forensics_integration(sample_image_bytes):
    """Test AI detection integration with forensics."""
    from backend.services.image_forensics import ImageForensics
    
    forensics = ImageForensics(sample_image_bytes, "test.png")
    ai_report = forensics.detect_ai_generation()
    
    assert "ai_probability" in ai_report
    assert "classification" in ai_report


def test_forensics_full_report_includes_ai(sample_image_bytes):
    """Test full forensic report includes AI detection."""
    from backend.services.image_forensics import ImageForensics
    
    forensics = ImageForensics(sample_image_bytes, "test.png")
    report = forensics.generate_forensic_report()
    
    assert "ai_detection" in report
    assert "ai_probability" in report["summary"]
    assert "ai_classification" in report["summary"]
