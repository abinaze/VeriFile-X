"""
Download real photo datasets.
Run from repo root: python scripts/datasets/download_real.py --dataset coco
"""
import csv
import argparse
import zipfile
import tarfile
import requests
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Repo root = where you run this from
ROOT      = Path(__file__).resolve().parents[2]
DATA_REAL = ROOT / "data" / "real"
MANIFEST  = ROOT / "data" / "manifest.csv"


# -- Helpers ----------------------------------------------------------------

def download_file(url: str, dest: Path, desc: str = "") -> Path:
    """Download with progress bar. Resumes if file partially downloaded."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    headers  = {"Range": f"bytes={existing}-"} if existing else {}

    r = requests.get(url, headers=headers, stream=True, timeout=60)
    total = int(r.headers.get("content-length", 0)) + existing

    mode = "ab" if existing else "wb"
    with open(dest, mode) as f, tqdm(
        total=total, initial=existing, unit="B",
        unit_scale=True, desc=desc or dest.name
    ) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    return dest


def extract(archive: Path, dest: Path):
    """Extract zip or tar archive."""
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive.name} -> {dest}")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    elif archive.name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive) as t:
            t.extractall(dest)
    print("Done extracting.")


def add_to_manifest(rows: list[dict]):
    """Append rows to manifest.csv."""
    fieldnames = [
        "path", "label", "source", "generator",
        "split", "width", "height", "has_exif", "verified"
    ]
    write_header = not MANIFEST.exists() or MANIFEST.stat().st_size == 0
    with open(MANIFEST, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"  Added {len(rows)} rows to manifest.csv")


def get_image_info(path: Path) -> dict:
    """Get width, height, has_exif from image file."""
    try:
        img  = Image.open(path)
        w, h = img.size
        exif = img._getexif() if hasattr(img, "_getexif") else None
        return {"width": w, "height": h, "has_exif": exif is not None}
    except Exception:
        return {"width": 0, "height": 0, "has_exif": False}


def assign_split(index: int, total: int) -> str:
    """80% train, 10% val, 10% test."""
    pct = index / total
    if pct < 0.80: return "train"
    if pct < 0.90: return "val"
    return "test"


# -- Dataset downloaders ----------------------------------------------------

def download_coco():
    """COCO 2017 validation set — 5,000 real photos, 1GB, CC BY 4.0."""
    print("\n-- COCO 2017 val --------------------------------------")
    dest_dir = DATA_REAL / "coco_val"
    archive  = dest_dir / "val2017.zip"

    if not archive.exists():
        download_file(
            "http://images.cocodataset.org/zips/val2017.zip",
            archive,
            "COCO 2017 val"
        )
    else:
        print("Archive already downloaded, skipping.")

    extract(archive, dest_dir)

    images = list((dest_dir / "val2017").glob("*.jpg"))
    print(f"Found {len(images)} images")

    rows = []
    for i, img_path in enumerate(sorted(images)):
        info = get_image_info(img_path)
        if info["width"] < 256 or info["height"] < 256:
            continue  # skip tiny images
        rows.append({
            "path":      str(img_path.relative_to(ROOT)),
            "label":     "real",
            "source":    "coco_val",
            "generator": "none",
            "split":     assign_split(i, len(images)),
            "width":     info["width"],
            "height":    info["height"],
            "has_exif":  info["has_exif"],
            "verified":  True,
        })

    add_to_manifest(rows)
    print(f"COCO done: {len(rows)} images added.")


def download_div2k():
    """DIV2K — 1,000 high-resolution real photos, CC Free research."""
    print("\n-- DIV2K ----------------------------------------------")
    dest_dir = DATA_REAL / "div2k"
    dest_dir.mkdir(parents=True, exist_ok=True)

    urls = [
        ("https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip", "DIV2K train HR"),
        ("https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip", "DIV2K valid HR"),
    ]

    for url, desc in urls:
        archive = dest_dir / Path(url).name
        if not archive.exists():
            download_file(url, archive, desc)
        else:
            print(f"{archive.name} already downloaded.")
        extract(archive, dest_dir)

    images = list(dest_dir.rglob("*.png")) + list(dest_dir.rglob("*.jpg"))
    print(f"Found {len(images)} images")

    rows = []
    for i, img_path in enumerate(sorted(images)):
        info = get_image_info(img_path)
        rows.append({
            "path":      str(img_path.relative_to(ROOT)),
            "label":     "real",
            "source":    "div2k",
            "generator": "none",
            "split":     assign_split(i, len(images)),
            "width":     info["width"],
            "height":    info["height"],
            "has_exif":  info["has_exif"],
            "verified":  True,
        })

    add_to_manifest(rows)
    print(f"DIV2K done: {len(rows)} images added.")


def download_raise1k():
    """
    RAISE-1k — 1,000 raw uncompressed DSLR photos.
    Direct download requires registration at http://loki.disi.unitn.it/RAISE/
    This script prints the instructions since it requires a form submission.
    """
    print("\n-- RAISE-1k -------------------------------------------")
    print("RAISE requires a registration form — cannot be automated.")
    print("Steps:")
    print("  1. Go to: http://loki.disi.unitn.it/RAISE/")
    print("  2. Fill the form to request download links")
    print("  3. Download the ZIP files you receive by email")
    print("  4. Extract to: data/real/raise_1k/")
    print("  5. Run: python scripts/datasets/index_manual.py --source raise_1k --label real")
    print("Skipping automated download.")


def download_unsplash():
    """
    Unsplash Lite — 25,000 professional photos.
    Uses the Unsplash dataset GitHub release.
    """
    print("\n-- Unsplash Lite --------------------------------------")
    dest_dir = DATA_REAL / "unsplash"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Unsplash Lite dataset CSV (metadata only — images need individual fetch)
    csv_url = (
        "https://github.com/unsplash/datasets/releases/download/"
        "v1.2.0/lite-00000-of-00001.tsv"
    )
    csv_dest = dest_dir / "unsplash_lite.tsv"

    if not csv_dest.exists():
        download_file(csv_url, csv_dest, "Unsplash Lite metadata")
    else:
        print("Metadata already downloaded.")

    # Parse and download images (first 5000 for starter set)
    print("Downloading images from Unsplash CDN (first 5,000)...")
    rows = []
    downloaded = 0
    limit = 5000

    with open(csv_dest, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if downloaded >= limit:
                break
            photo_id = row.get("photo_id", "")
            if not photo_id:
                continue

            img_url  = f"https://images.unsplash.com/photo-{photo_id}?w=1024&q=90"
            img_path = dest_dir / f"{photo_id}.jpg"

            if img_path.exists():
                downloaded += 1
                continue

            try:
                r = requests.get(img_url, timeout=15)
                if r.status_code == 200:
                    img_path.write_bytes(r.content)
                    info = get_image_info(img_path)
                    if info["width"] < 256:
                        img_path.unlink()
                        continue
                    rows.append({
                        "path":      str(img_path.relative_to(ROOT)),
                        "label":     "real",
                        "source":    "unsplash",
                        "generator": "none",
                        "split":     assign_split(downloaded, limit),
                        "width":     info["width"],
                        "height":    info["height"],
                        "has_exif":  info["has_exif"],
                        "verified":  True,
                    })
                    downloaded += 1
                    if downloaded % 100 == 0:
                        print(f"  Downloaded {downloaded}/{limit}")
            except Exception as e:
                print(f"  Failed {photo_id}: {e}")
                continue

    add_to_manifest(rows)
    print(f"Unsplash done: {len(rows)} images added.")


# -- Entry point ------------------------------------------------------------

DATASETS = {
    "coco":     download_coco,
    "div2k":    download_div2k,
    "raise1k":  download_raise1k,
    "unsplash": download_unsplash,
    "all":      None,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download real photo datasets")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        required=True,
        help="Which dataset to download"
    )
    args = parser.parse_args()

    if args.dataset == "all":
        download_coco()
        download_div2k()
        download_raise1k()
        download_unsplash()
    else:
        DATASETS[args.dataset]()

    print("\nDone. Check data/manifest.csv for all added images.")
