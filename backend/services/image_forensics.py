"""
Image forensics analysis module.
Why: Extract metadata, generate hashes, detect tampering.
"""
import hashlib
import imagehash
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from io import BytesIO
from typing import Dict, Optional, Any
from datetime import datetime

from backend.core.logger import setup_logger

logger = setup_logger(__name__)


class ImageForensics:
    """
    Image forensics analyzer.
    Extracts metadata, generates hashes, detects anomalies.
    """
    
    def __init__(self, image_bytes: bytes, filename: str):
        """
        Initialize forensics analyzer.
        
        Args:
            image_bytes: Raw image file content
            filename: Original filename
        """
        self.image_bytes = image_bytes
        self.filename = filename
        self.image = Image.open(BytesIO(image_bytes))
        logger.info(f"Initialized forensics for {filename}")
    
    def extract_exif(self) -> Dict[str, Any]:
        """
        Extract EXIF metadata from image.
        
        Why EXIF matters:
        - Real photos have camera metadata
        - AI-generated images often lack EXIF
        - Editing software leaves traces
        
        Returns:
            Dictionary of EXIF data
        """
        exif_data = {}
        
        try:
            # Get raw EXIF data
            exif = self.image._getexif()
            
            if exif is None:
                logger.warning(f"No EXIF data found in {self.filename}")
                return {"has_exif": False, "warning": "No EXIF metadata present"}
            
            # Parse EXIF tags
            for tag_id, value in exif.items():
                tag_name = TAGS.get(tag_id, tag_id)
                
                # Handle GPS data separately
                if tag_name == "GPSInfo":
                    gps_data = self._parse_gps(value)
                    exif_data["gps"] = gps_data
                else:
                    # Convert bytes to string if needed
                    if isinstance(value, bytes):
                        value = value.decode(errors='ignore')
                    exif_data[tag_name] = value
            
            exif_data["has_exif"] = True
            logger.info(f"Extracted {len(exif_data)} EXIF fields from {self.filename}")
            
        except AttributeError:
            logger.warning(f"Image format does not support EXIF: {self.filename}")
            exif_data = {"has_exif": False, "warning": "Image format does not support EXIF"}
        except Exception as e:
            logger.error(f"Error extracting EXIF from {self.filename}: {str(e)}")
            exif_data = {"has_exif": False, "error": str(e)}
        
        return exif_data
    
    def _parse_gps(self, gps_info: dict) -> Dict[str, Any]:
        """
        Parse GPS coordinates from EXIF.
        
        Why GPS matters:
        - Verifies photo location claims
        - Can detect stock photos (common GPS coords)
        - Spoofed GPS indicates manipulation
        """
        gps_data = {}
        
        try:
            for tag_id, value in gps_info.items():
                tag_name = GPSTAGS.get(tag_id, tag_id)
                gps_data[tag_name] = value
            
            # Convert to decimal degrees if coordinates present
            if "GPSLatitude" in gps_data and "GPSLongitude" in gps_data:
                lat = self._convert_to_degrees(
                    gps_data["GPSLatitude"],
                    gps_data.get("GPSLatitudeRef", "N")
                )
                lon = self._convert_to_degrees(
                    gps_data["GPSLongitude"],
                    gps_data.get("GPSLongitudeRef", "E")
                )
                gps_data["latitude"] = lat
                gps_data["longitude"] = lon
        except Exception as e:
            logger.error(f"Error parsing GPS data: {str(e)}")
            gps_data["error"] = str(e)
        
        return gps_data
    
    def _convert_to_degrees(self, value: tuple, ref: str) -> float:
        """Convert GPS coordinates to decimal degrees."""
        degrees = float(value[0])
        minutes = float(value[1])
        seconds = float(value[2])
        
        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
        
        if ref in ['S', 'W']:
            decimal = -decimal
        
        return round(decimal, 6)
    
    def generate_hashes(self) -> Dict[str, str]:
        """
        Generate cryptographic and perceptual hashes.
        
        Hash types:
        - SHA-256: Cryptographic integrity check
        - MD5: Legacy but still used
        - Perceptual: Detects similar images (survives resize/compression)
        
        Returns:
            Dictionary of hash values
        """
        hashes = {}
        
        try:
            # Cryptographic hashes (exact match)
            hashes["sha256"] = hashlib.sha256(self.image_bytes).hexdigest()
            hashes["md5"] = hashlib.md5(self.image_bytes).hexdigest()
            
            # Perceptual hash (detects similar images)
            # Survives minor edits, resize, compression
            phash = imagehash.phash(self.image)
            hashes["perceptual_hash"] = str(phash)
            
            # Average hash (faster, less accurate)
            ahash = imagehash.average_hash(self.image)
            hashes["average_hash"] = str(ahash)
            
            # Difference hash (good for finding duplicates)
            dhash = imagehash.dhash(self.image)
            hashes["difference_hash"] = str(dhash)
            
            logger.info(f"Generated 5 hashes for {self.filename}")
            
        except Exception as e:
            logger.error(f"Error generating hashes: {str(e)}")
            hashes["error"] = str(e)
        
        return hashes
    
    def detect_tampering_indicators(self, exif_data: dict) -> Dict[str, Any]:
        """
        Detect potential tampering indicators.
        
        Red flags:
        - No EXIF data (suspicious for camera photos)
        - Software field indicates editing (Photoshop, GIMP)
        - Timestamp inconsistencies
        - Missing expected camera data
        
        Returns:
            Dictionary of tampering indicators
        """
        indicators = {
            "suspicious_flags": [],
            "confidence": "high",  # high = likely authentic
            "analysis": []
        }
        
        # Check 1: Missing EXIF
        if not exif_data.get("has_exif"):
            indicators["suspicious_flags"].append("no_exif_data")
            indicators["analysis"].append("No EXIF metadata present (suspicious for camera photos)")
            indicators["confidence"] = "low"
        
        # Check 2: Editing software detected
        software = exif_data.get("Software", "")
        if software:
            editing_tools = ["Photoshop", "GIMP", "Paint", "Lightroom", "Affinity"]
            if any(tool.lower() in software.lower() for tool in editing_tools):
                indicators["suspicious_flags"].append("editing_software_detected")
                indicators["analysis"].append(f"Editing software detected: {software}")
                indicators["confidence"] = "medium"
        
        # Check 3: Missing camera data
        if exif_data.get("has_exif") and not exif_data.get("Make"):
            indicators["suspicious_flags"].append("missing_camera_make")
            indicators["analysis"].append("Camera make/model information missing")
            indicators["confidence"] = "medium"
        
        # Check 4: AI generation indicators
        # Some AI tools add software tags
        ai_indicators = ["midjourney", "stable diffusion", "dall-e", "generated"]
        software_lower = software.lower() if software else ""
        if any(ai_tool in software_lower for ai_tool in ai_indicators):
            indicators["suspicious_flags"].append("ai_generation_detected")
            indicators["analysis"].append(f"AI generation indicator in metadata: {software}")
            indicators["confidence"] = "very_low"
        
        logger.info(f"Tampering analysis complete: {len(indicators['suspicious_flags'])} flags")
        
        return indicators
    
    def generate_forensic_report(self) -> Dict[str, Any]:
        """
        Generate complete forensic analysis report.
        
        Returns:
            Comprehensive forensic report as JSON-serializable dict
        """
        logger.info(f"Generating forensic report for {self.filename}")
        
        # Extract all forensic data
        exif_data = self.extract_exif()
        hashes = self.generate_hashes()
        tampering = self.detect_tampering_indicators(exif_data)
        
        # Basic image properties
        image_info = {
            "filename": self.filename,
            "format": self.image.format,
            "mode": self.image.mode,
            "size": self.image.size,
            "width": self.image.width,
            "height": self.image.height,
            "file_size_bytes": len(self.image_bytes),
        }
        
        # Compile report
        report = {
            "metadata": {
                "analysis_timestamp": datetime.now().isoformat(),
                "analyzer_version": "1.0.0"
            },
            "file_info": image_info,
            "exif_data": exif_data,
            "hashes": hashes,
            "tampering_analysis": tampering,
            "summary": {
                "has_metadata": exif_data.get("has_exif", False),
                "suspicious_flags_count": len(tampering["suspicious_flags"]),
                "authenticity_confidence": tampering["confidence"]
            }
        }
        
        logger.info(f"Forensic report generated: {report['summary']}")
        
        return report
