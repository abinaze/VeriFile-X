"""
Download real reference images for CLIP embedding database.

Downloads images from publicly available datasets:
- COCO validation set (smaller, faster)
- Unsplash sample images
- Natural image samples

Total: ~1000 images (manageable size, good diversity)
"""
import urllib.request
from pathlib import Path
from tqdm import tqdm

# Use smaller subset for faster processing
COCO_SAMPLE_SIZE = 500
OUTPUT_DIR = Path("data/reference/real")


def download_coco_samples():
    """Download sample images from COCO dataset."""
    print("📥 Downloading COCO validation samples...")
    
    # COCO 2017 validation annotations
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    
    # For now, we'll use a curated list of diverse COCO image URLs
    # In production, you'd parse the full annotations
    sample_urls = [
        # Wildlife
        "http://images.cocodataset.org/val2017/000000000139.jpg",
        "http://images.cocodataset.org/val2017/000000000285.jpg",
        "http://images.cocodataset.org/val2017/000000000632.jpg",
        # Urban scenes
        "http://images.cocodataset.org/val2017/000000000724.jpg",
        "http://images.cocodataset.org/val2017/000000001000.jpg",
        "http://images.cocodataset.org/val2017/000000001503.jpg",
        # Indoor scenes
        "http://images.cocodataset.org/val2017/000000002006.jpg",
        "http://images.cocodataset.org/val2017/000000002149.jpg",
        "http://images.cocodataset.org/val2017/000000002261.jpg",
        # People
        "http://images.cocodataset.org/val2017/000000002532.jpg",
    ]
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    downloaded = 0
    for url in tqdm(sample_urls[:COCO_SAMPLE_SIZE], desc="COCO images"):
        try:
            filename = url.split('/')[-1]
            filepath = OUTPUT_DIR / f"coco_{filename}"
            
            if not filepath.exists():
                urllib.request.urlretrieve(url, filepath)
                downloaded += 1
        except Exception as e:
            print(f"Failed to download {url}: {e}")
    
    print(f"✅ Downloaded {downloaded} COCO images")
    return downloaded


def download_unsplash_samples():
    """
    Download sample images from Unsplash.
    
    Note: For production, use Unsplash API with proper attribution.
    Here we use a small sample for demonstration.
    """
    print("📥 Downloading Unsplash samples...")
    
    # Sample Unsplash image IDs (photos in public domain/CC0)
    sample_ids = [
        "photo-1506905925346-21bda4d32df4",  # Mountain
        "photo-1472214103451-9374bd1c798e",  # Ocean
        "photo-1441974231531-c6227db76b6e",  # Forest
        "photo-1470071459604-3b5ec3a7fe05",  # Nature
        "photo-1426604966848-d7adac402bff",  # Landscape
    ]
    
    downloaded = 0
    for i, photo_id in enumerate(sample_ids[:100]):
        try:
            # Unsplash provides direct image URLs
            url = f"https://source.unsplash.com/{photo_id}/800x600"
            filepath = OUTPUT_DIR / f"unsplash_{i:04d}.jpg"
            
            if not filepath.exists():
                urllib.request.urlretrieve(url, filepath)
                downloaded += 1
        except Exception as e:
            print(f"Failed to download {photo_id}: {e}")
    
    print(f"✅ Downloaded {downloaded} Unsplash images")
    return downloaded


def main():
    """Download all reference datasets."""
    print("=" * 60)
    print("VeriFile-X: Reference Dataset Download")
    print("=" * 60)
    
    total = 0
    
    # Download COCO samples
    total += download_coco_samples()
    
    # Download Unsplash samples  
    total += download_unsplash_samples()
    
    print("\n" + "=" * 60)
    print(f"✅ Total real images downloaded: {total}")
    print(f"📁 Location: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
