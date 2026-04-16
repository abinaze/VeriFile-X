"""
Batch Investigation Mode.

Processes multiple images in a single request, returning a unified
forensic report with cross-image analysis including:

- Per-image full forensic report
- Batch-level statistics (mean AI probability, class distribution)
- Clustering: which images are most similar (hash-based)
- Duplicate detection via perceptual hash comparison
- Highest-risk images ranked by AI probability
- Cross-image provenance consistency check

Limits:
  MAX_BATCH_SIZE = 10 images per request
  MAX_IMAGE_SIZE = 5MB per image in batch mode
  Processing is sequential to avoid OOM on GPU
"""
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

MAX_BATCH_SIZE    = 10
MAX_IMAGE_BYTES   = 5 * 1024 * 1024  # 5MB per image in batch


def _phash_distance(h1: str, h2: str) -> int:
    """Hamming distance between two perceptual hash hex strings."""
    try:
        i1 = int(h1, 16)
        i2 = int(h2, 16)
        xor = i1 ^ i2
        return bin(xor).count("1")
    except Exception:
        return 64  # Max distance on error


def _find_duplicates(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Identify duplicate or near-duplicate images using perceptual hash.
    Threshold: Hamming distance <= 8 (out of 64 bits).
    """
    pairs = []
    for i in range(len(reports)):
        for j in range(i + 1, len(reports)):
            h1 = reports[i].get("hashes", {}).get("perceptual_hash", "")
            h2 = reports[j].get("hashes", {}).get("perceptual_hash", "")
            if h1 and h2:
                dist = _phash_distance(h1, h2)
                if dist <= 8:
                    pairs.append({
                        "image_a":      reports[i]["file_info"]["filename"],
                        "image_b":      reports[j]["file_info"]["filename"],
                        "phash_distance": dist,
                        "similarity":   "identical" if dist == 0 else "near_duplicate",
                    })
    return pairs


def _rank_by_risk(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return images ranked by AI probability descending."""
    ranked = []
    for r in reports:
        ranked.append({
            "filename":         r["file_info"]["filename"],
            "ai_probability":   r["summary"]["ai_probability"],
            "classification":   r["summary"]["ai_classification"],
            "evidence_id":      r["evidence_id"],
            "predicted_generator": r.get("generator_attribution", {}).get("predicted_generator", "unknown"),
            "c2pa_status":      r.get("c2pa_provenance", {}).get("provenance_status", "none"),
        })
    return sorted(ranked, key=lambda x: x["ai_probability"], reverse=True)


def _batch_statistics(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate statistics across all images in the batch."""
    probs = [r["summary"]["ai_probability"] for r in reports]
    classes = [r["summary"]["ai_classification"] for r in reports]
    generators = [
        r.get("generator_attribution", {}).get("predicted_generator", "unknown")
        for r in reports
    ]
    c2pa_statuses = [
        r.get("c2pa_provenance", {}).get("provenance_status", "none")
        for r in reports
    ]

    class_counts: Dict[str, int] = {}
    for c in classes:
        class_counts[c] = class_counts.get(c, 0) + 1

    generator_counts: Dict[str, int] = {}
    for g in generators:
        generator_counts[g] = generator_counts.get(g, 0) + 1

    ai_count   = sum(1 for c in classes if "ai_generated" in c)
    real_count = sum(1 for c in classes if "authentic" in c)

    return {
        "total_images":          len(reports),
        "ai_detected_count":     ai_count,
        "authentic_count":       real_count,
        "uncertain_count":       len(reports) - ai_count - real_count,
        "mean_ai_probability":   round(sum(probs) / len(probs), 4) if probs else 0.0,
        "max_ai_probability":    round(max(probs), 4) if probs else 0.0,
        "min_ai_probability":    round(min(probs), 4) if probs else 0.0,
        "classification_breakdown": class_counts,
        "generator_breakdown":   generator_counts,
        "c2pa_has_credentials":  sum(1 for s in c2pa_statuses if s != "none"),
        "batch_verdict": (
            "high_risk"    if ai_count / max(len(reports), 1) >= 0.6 else
            "mixed"        if ai_count > 0 else
            "likely_authentic"
        ),
    }


def _provenance_consistency(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Check if C2PA provenance is consistent across images in the batch.
    A collection where some images have credentials and others do not
    may indicate selective credential stripping.
    """
    statuses = [
        r.get("c2pa_provenance", {}).get("provenance_status", "none")
        for r in reports
    ]
    has_credentials   = sum(1 for s in statuses if s != "none")
    lacks_credentials = sum(1 for s in statuses if s == "none")

    if has_credentials == 0:
        consistency = "consistent_no_credentials"
        note = "No images in batch have C2PA credentials. This is normal for most images."
    elif lacks_credentials == 0:
        consistency = "consistent_all_credentialed"
        note = "All images have C2PA credentials. Strong provenance signal."
    else:
        consistency = "inconsistent"
        note = (
            f"{has_credentials} of {len(reports)} images have C2PA credentials. "
            "Mixed provenance may indicate selective credential stripping."
        )

    return {
        "consistency":            consistency,
        "images_with_credentials": has_credentials,
        "images_without_credentials": lacks_credentials,
        "note":                   note,
    }


def process_batch(
    images: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Process a batch of images through the full forensic pipeline.

    Args:
        images: List of dicts with keys:
                  filename  - str
                  data      - bytes (image bytes)

    Returns:
        Batch forensic report with per-image results and aggregate analysis
    """
    from backend.services.image_forensics import ImageForensics

    if len(images) > MAX_BATCH_SIZE:
        return {
            "error":   f"Batch too large. Max {MAX_BATCH_SIZE} images per request.",
            "maximum": MAX_BATCH_SIZE,
            "received": len(images),
        }

    results     = []
    errors      = []
    reports     = []

    for item in images:
        filename = item.get("filename", "unknown")
        data     = item.get("data", b"")

        if len(data) == 0:
            errors.append({"filename": filename, "error": "Empty file"})
            continue

        if len(data) > MAX_IMAGE_BYTES:
            errors.append({
                "filename": filename,
                "error":    f"File too large for batch mode (max {MAX_IMAGE_BYTES // (1024*1024)}MB)"
            })
            continue

        try:
            forensics = ImageForensics(data, filename)
            report    = forensics.generate_forensic_report()
            reports.append(report)
            results.append({
                "filename":    filename,
                "status":      "success",
                "evidence_id": report["evidence_id"],
                "report":      report,
            })
            logger.info(
                f"Batch: processed {filename} "
                f"(ai={report['summary']['ai_probability']:.3f})"
            )
        except Exception as e:
            logger.warning(f"Batch: failed {filename}: {e}")
            errors.append({"filename": filename, "error": str(e)})

    if not reports:
        return {
            "status":         "failed",
            "processed":      0,
            "errors":         errors,
            "error":          "No images could be processed",
        }

    return {
        "status":                 "complete",
        "processed":              len(reports),
        "failed":                 len(errors),
        "errors":                 errors,
        "statistics":             _batch_statistics(reports),
        "risk_ranking":           _rank_by_risk(reports),
        "duplicate_pairs":        _find_duplicates(reports),
        "provenance_consistency": _provenance_consistency(reports),
        "individual_reports":     results,
    }
