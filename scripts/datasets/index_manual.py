"""
Index images you downloaded manually into manifest.csv.
Run: python scripts/datasets/index_manual.py --source raise_1k --label real
"""
import csv
import argparse
from pathlib import Path
from PIL import Image

ROOT     = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "manifest.csv"


def get_image_info(path):
    try:
        img  = Image.open(path)
        w, h = img.size
        exif = img._getexif() if hasattr(img, "_getexif") else None
        return w, h, exif is not None
    except Exception:
        return 0, 0, False


def assign_split(i, total):
    pct = i / total
    if pct < 0.80: return "train"
    if pct < 0.90: return "val"
    return "test"


def index_folder(source: str, label: str, generator: str):
    label_map = {"real": "data/real", "ai": "data/ai"}
    folder = ROOT / label_map[label] / source

    if not folder.exists():
        print(f"Folder not found: {folder}")
        return

    exts   = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
    images = [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in exts]
    print(f"Found {len(images)} images in {folder}")

    fieldnames = [
        "path", "label", "source", "generator",
        "split", "width", "height", "has_exif", "verified"
    ]
    write_header = not MANIFEST.exists() or MANIFEST.stat().st_size == 0

    rows = []
    for i, img_path in enumerate(images):
        w, h, has_exif = get_image_info(img_path)
        if w < 256 or h < 256:
            print(f"  Skipping (too small): {img_path.name}")
            continue
        rows.append({
            "path":      str(img_path.relative_to(ROOT)),
            "label":     label,
            "source":    source,
            "generator": generator,
            "split":     assign_split(i, len(images)),
            "width":     w,
            "height":    h,
            "has_exif":  has_exif,
            "verified":  True,
        })

    with open(MANIFEST, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerows(rows)

    print(f"Done. Added {len(rows)} images from {source} to manifest.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",    required=True, help="e.g. raise_1k")
    parser.add_argument("--label",     required=True, choices=["real", "ai"])
    parser.add_argument("--generator", default="none", help="e.g. stylegan2")
    args = parser.parse_args()
    index_folder(args.source, args.label, args.generator)
