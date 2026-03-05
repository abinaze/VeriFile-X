"""
Image forensics service with advanced ensemble AI detection.
"""
from typing import Dict, Any
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import hashlib
import imagehash
from io import BytesIO

from backend.core.logger import setup_logger
from backend.services.advanced_ensemble_detector import AdvancedEnsembleDetector

logger = setup_logger(__name__)


class ImageForensics:
    """Complete image forensics analysis pipeline with advanced detection."""
    
    def __init__(self, image_bytes: bytes, filename: str):
        """Initialize forensics analyzer."""
        self.image_bytes = image_bytes
        self.filename = filename
        self.pil_image = Image.open(BytesIO(image_bytes))
        logger.info(f"Initialized forensics for {filename}")
    
    def extract_exif(self) -> Dict[str, Any]:
        """Extract EXIF metadata."""
        exif_data = {}
        try:
            exif = self.pil_image._getexif()
            if not exif:
                logger.warning(f"No EXIF data found in {self.filename}")
                return {"has_exif": False}
            
            exif_data["has_exif"] = True
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == "GPSInfo":
                    gps_data = {}
                    for gps_tag_id in value:
                        gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                        gps_data[gps_tag] = value[gps_tag_id]
                    exif_data["gps"] = gps_data
                else:
                    exif_data[tag] = str(value)
            logger.info(f"Extracted EXIF: {len(exif_data)} fields")
        except (AttributeError, KeyError, IndexError) as e:
            logger.warning(f"Error extracting EXIF: {e}")
            exif_data["has_exif"] = False
        return exif_data
    
    def generate_hashes(self) -> Dict[str, str]:
        """Generate cryptographic and perceptual hashes."""
        sha256 = hashlib.sha256(self.image_bytes).hexdigest()
        md5 = hashlib.md5(self.image_bytes).hexdigest()
        phash = str(imagehash.phash(self.pil_image))
        ahash = str(imagehash.average_hash(self.pil_image))
        dhash = str(imagehash.dhash(self.pil_image))
        logger.info(f"Generated 5 hashes for {self.filename}")
        return {
            "sha256": sha256,
            "md5": md5,
            "perceptual_hash": phash,
            "average_hash": ahash,
            "difference_hash": dhash
        }
    
    def detect_tampering_indicators(self, exif_data: Dict) -> Dict[str, Any]:
        """Detect tampering indicators."""
        suspicious_flags = []
        if not exif_data.get("has_exif", False):
            suspicious_flags.append("Missing EXIF metadata")
        if exif_data.get("Software"):
            software = exif_data["Software"].lower()
            editing_tools = ["photoshop", "gimp", "paint.net", "pixlr", "canva"]
            if any(tool in software for tool in editing_tools):
                suspicious_flags.append(f"Editing software detected: {exif_data['Software']}")
        ai_keywords = ["midjourney", "dall-e", "stable diffusion", "ai generated"]
        for key, value in exif_data.items():
            if isinstance(value, str) and any(kw in value.lower() for kw in ai_keywords):
                suspicious_flags.append(f"AI generation marker in {key}")
        confidence = "high" if len(suspicious_flags) == 0 else "medium" if len(suspicious_flags) <= 2 else "low"
        logger.info(f"Tampering analysis complete: {len(suspicious_flags)} flags")
        return {"suspicious_flags": suspicious_flags, "confidence": confidence}
    
    def detect_ai_generation(self) -> Dict[str, Any]:
        """Run advanced ensemble AI detection."""
        logger.info(f"Running advanced ensemble AI detection for {self.filename}")
        detector = AdvancedEnsembleDetector(self.image_bytes, self.filename)
        result = detector.detect()
        
        # Clean up GPU resources
        detector.cleanup()
        
        return result
    
    def generate_forensic_report(self) -> Dict[str, Any]:
        """Generate complete forensic report."""
        logger.info(f"Generating forensic report for {self.filename}")
        exif_data = self.extract_exif()
        hashes = self.generate_hashes()
        tampering = self.detect_tampering_indicators(exif_data)
        ai_detection = self.detect_ai_generation()
        width, height = self.pil_image.size
        image_format = self.pil_image.format or "Unknown"
        mode = self.pil_image.mode
        image_info = {
            "filename": self.filename,
            "format": image_format,
            "mode": mode,
            "width": width,
            "height": height,
            "file_size_bytes": len(self.image_bytes)
        }
        report = {
            "metadata": {
                "analysis_timestamp": datetime.now().isoformat(),
                "analyzer_version": "6.0.0"
            },
            "file_info": image_info,
            "exif_data": exif_data,
            "hashes": hashes,
            "tampering_analysis": tampering,
            "ai_detection": ai_detection,
            "summary": {
                "has_metadata": exif_data.get("has_exif", False),
                "suspicious_flags_count": len(tampering["suspicious_flags"]),
                "authenticity_confidence": tampering["confidence"],
                "ai_probability": ai_detection["ai_probability"],
                "ai_classification": ai_detection["classification"],
                "total_detection_signals": ai_detection["total_signals"],
                "suspicious_detection_signals": ai_detection["suspicious_signals_count"]
            }
        }
        logger.info(f"Forensic report generated: {report['summary']}")
        return report
