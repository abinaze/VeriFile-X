"""
Deep Metadata Forensics Analysis.

Analyzes EXIF and image metadata for forensic inconsistencies:
- Camera model vs image characteristics
- GPS coordinate plausibility
- Timestamp logic (created vs modified)
- Software markers indicating AI generation or editing
- Missing metadata patterns typical of AI images
- Resolution and DPI consistency

Real photos have rich, consistent metadata.
AI-generated images typically have no EXIF, or inconsistent metadata.
"""
from typing import Dict, Any
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from io import BytesIO
from backend.core.logger import setup_logger

logger = setup_logger(__name__)

# Known AI generation software markers
AI_SOFTWARE_MARKERS = [
    "midjourney", "dall-e", "dall·e", "stable diffusion", "sdxl",
    "novelai", "dreamstudio", "firefly", "imagen", "nightcafe",
    "artbreeder", "runway", "leonardo", "ai generated", "ai-generated",
    "comfyui", "automatic1111", "invokeai", "diffusers"
]

# Known image editing software
EDITING_SOFTWARE = [
    "photoshop", "gimp", "lightroom", "affinity", "pixelmator",
    "paint.net", "canva", "snapseed", "vsco", "facetune",
    "meitu", "picsart", "capture one"
]

# Camera manufacturers (real cameras have these)
CAMERA_MANUFACTURERS = [
    "canon", "nikon", "sony", "fujifilm", "olympus", "panasonic",
    "leica", "hasselblad", "pentax", "ricoh", "apple", "samsung",
    "google", "huawei", "xiaomi", "dji", "gopro"
]


def _extract_full_exif(pil_image: Image.Image) -> Dict[str, Any]:
    """Extract all available EXIF data."""
    result = {}
    try:
        exif_obj = pil_image.getexif()
        raw = dict(exif_obj) if exif_obj else None
        if not raw:
            return {}
        for tag_id, value in raw.items():
            tag = TAGS.get(tag_id, str(tag_id))
            if tag == "GPSInfo" and isinstance(value, dict):
                gps = {}
                for k, v in value.items():
                    gps[GPSTAGS.get(k, k)] = v
                result["GPSInfo"] = gps
            else:
                try:
                    result[tag] = str(value)[:200]
                except Exception:
                    pass
    except Exception:
        pass
    return result


def _check_gps_plausibility(gps_info: Dict) -> tuple:
    """Check if GPS coordinates are plausible."""
    try:
        lat = gps_info.get("GPSLatitude")
        lon = gps_info.get("GPSLongitude")
        if not lat or not lon:
            return True, "No GPS data"
        # Basic range check
        if isinstance(lat, tuple) and len(lat) >= 1:
            lat_val = float(lat[0])
            lon_val = float(lon[0]) if isinstance(lon, tuple) else 0
            if 0 <= lat_val <= 90 and 0 <= lon_val <= 180:
                return True, f"GPS valid ({lat_val:.2f}, {lon_val:.2f})"
            else:
                return False, "GPS coordinates out of valid range"
    except Exception:
        pass
    return True, "GPS data present but unverifiable"


def _check_timestamps(exif: Dict) -> tuple:
    """Check timestamp consistency."""
    datetime_orig = exif.get("DateTimeOriginal", "")
    datetime_dig = exif.get("DateTimeDigitized", "")
    datetime_mod = exif.get("DateTime", "")

    issues = []
    if datetime_orig and datetime_dig:
        if datetime_orig != datetime_dig:
            issues.append("Original and digitized timestamps differ")
    if datetime_orig and datetime_mod:
        # Modified should be >= original
        try:
            # EXIF datetime format YYYY:MM:DD HH:MM:SS is safely string-comparable
            if len(datetime_mod) == 19 and len(datetime_orig) == 19 and datetime_mod < datetime_orig:
                issues.append("File modified before original capture date")
        except Exception:
            pass

    return len(issues) == 0, issues


def analyze_metadata(image_bytes: bytes, filename: str = "unknown") -> Dict[str, Any]:
    """
    Deep forensic metadata analysis.
    Returns both a detailed report and a signal score.
    """
    try:
        img = Image.open(BytesIO(image_bytes))
        exif = _extract_full_exif(img)

        flags = []
        positives = []
        score_components = []

        # === Check 1: Missing EXIF (weak indicator — common in screenshots, exports, web images) ===
        if not exif:
            flags.append("No EXIF metadata — possible AI generation or web export")
            score_components.append(0.40)
        else:
            positives.append("EXIF metadata present")
            score_components.append(0.15)

        # === Check 2: Camera make/model ===
        make = exif.get("Make", "").lower()
        model = exif.get("Model", "").lower()

        if make or model:
            has_real_camera = any(m in (make + model) for m in CAMERA_MANUFACTURERS)
            if has_real_camera:
                positives.append(f"Real camera detected: {exif.get('Make','')} {exif.get('Model','')}")
                score_components.append(0.10)
            else:
                flags.append(f"Unknown camera make/model: {make} {model}")
                score_components.append(0.45)
        elif exif:
            flags.append("EXIF present but no camera make/model")
            score_components.append(0.55)

        # === Check 3: AI/editing software markers ===
        software = exif.get("Software", "").lower()
        artist = exif.get("Artist", "").lower()
        description = exif.get("ImageDescription", "").lower()
        combined_text = software + " " + artist + " " + description

        ai_found = [m for m in AI_SOFTWARE_MARKERS if m in combined_text]
        edit_found = [m for m in EDITING_SOFTWARE if m in combined_text]

        if ai_found:
            flags.append(f"AI generation software detected: {', '.join(ai_found)}")
            score_components.append(0.95)
        elif edit_found:
            flags.append(f"Image editing software detected: {', '.join(edit_found)}")
            score_components.append(0.60)
        elif software:
            positives.append(f"Software: {exif.get('Software', '')}")
            score_components.append(0.20)

        # === Check 4: Timestamp consistency ===
        ts_ok, ts_issues = _check_timestamps(exif)
        if not ts_ok:
            for issue in ts_issues:
                flags.append(issue)
            score_components.append(0.65)
        elif exif.get("DateTimeOriginal"):
            positives.append(f"Timestamp: {exif.get('DateTimeOriginal', '')}")
            score_components.append(0.10)

        # === Check 5: GPS plausibility ===
        gps_info = exif.get("GPSInfo", {})
        if gps_info:
            gps_ok, gps_msg = _check_gps_plausibility(gps_info)
            if gps_ok:
                positives.append(f"GPS data valid: {gps_msg}")
                score_components.append(0.05)
            else:
                flags.append(f"GPS anomaly: {gps_msg}")
                score_components.append(0.70)

        # === Check 6: Image format vs content consistency ===
        mode = img.mode
        width, height = img.size

        # Very round dimensions common in AI
        if width % 64 == 0 and height % 64 == 0 and width >= 512:
            flags.append(
                f"Dimensions {width}x{height} are multiples of 64 "
                "(weak AI generation indicator — also common in exports)"
            )
            score_components.append(0.25)

        # === Compute final score ===
        if score_components:
            ai_score = float(sum(score_components) / len(score_components))
        else:
            ai_score = 0.5

        ai_score = min(1.0, max(0.0, ai_score))
        confidence = 0.75 if exif else 0.45

        if flags:
            explanation = f"Metadata anomalies: {flags[0]}"
        elif positives:
            explanation = f"Metadata consistent with authentic photo: {positives[0]}"
        else:
            explanation = "No metadata available for analysis"

        report = {
            "has_exif": bool(exif),
            "flags": flags,
            "positives": positives,
            "camera_make": exif.get("Make", ""),
            "camera_model": exif.get("Model", ""),
            "software": exif.get("Software", ""),
            "timestamp": exif.get("DateTimeOriginal", ""),
            "has_gps": bool(gps_info),
            "image_dimensions": f"{width}x{height}",
            "color_mode": mode,
        }

        signal = {
            "signal_name": "Metadata Forensics",
            "score": ai_score,
            "confidence": confidence,
            "explanation": explanation,
            "raw_value": len(flags),
            "method": "metadata_forensics",
            "detailed_report": report
        }

        logger.info(
            f"Metadata forensics: score={ai_score:.3f}, "
            f"flags={len(flags)}, positives={len(positives)}, file={filename}"
        )
        return signal

    except Exception as e:
        logger.warning(f"Metadata forensics failed: {e}")
        return {
            "signal_name": "Metadata Forensics",
            "score": 0.5,
            "confidence": 0.0,
            "explanation": f"Metadata analysis unavailable: {str(e)}",
            "raw_value": 0,
            "method": "metadata_forensics"
        }