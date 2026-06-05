"""
Download real photo datasets.
Run from repo root: python scripts/datasets/download_real.py --dataset coco
"""
import csv
import argparse
import zipfile
import tarfile
import time
import requests
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Repo root = where you run this from
ROOT      = Path(__file__).resolve().parents[2]
DATA_REAL = ROOT / "data" / "real"
MANIFEST  = ROOT / "data" / "manifest.csv"


# -- Helpers ----------------------------------------------------------------

def _is_valid_zip(path: Path) -> bool:
    """Return True only if the file exists AND is a valid (complete) zip."""
    if not path.exists() or path.stat().st_size < 22:   # 22 = min zip size
        return False
    try:
        with zipfile.ZipFile(path, "r") as z:
            bad = z.testzip()   # None means all good; returns first bad file name otherwise
            return bad is None
    except (zipfile.BadZipFile, OSError):
        return False


def download_file(url: str, dest: Path, desc: str = "",
                  timeout: int = 120, max_retries: int = 5) -> Path:
    """
    Download with progress bar.
    - Resumes where it left off using HTTP Range requests.
    - Retries up to max_retries times with exponential back-off.
    - Deletes the partial file if the server doesn't support resuming
      (i.e. responds 200 instead of 206 when we send a Range header).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            existing = dest.stat().st_size if dest.exists() else 0
            headers  = {"Range": f"bytes={existing}-"} if existing else {}

            r = requests.get(url, headers=headers, stream=True,
                             timeout=timeout)
            r.raise_for_status()

            # If server ignored our Range header and sent the full file again,
            # start fresh to avoid corrupted concatenation.
            if existing and r.status_code == 200:
                print(f"  Server does not support resume — restarting download.")
                dest.unlink(missing_ok=True)
                existing = 0

            content_length = int(r.headers.get("content-length", 0))
            total = content_length + existing

            mode = "ab" if existing else "wb"
            with open(dest, mode) as f, tqdm(
                total=total, initial=existing, unit="B",
                unit_scale=True, desc=desc or dest.name
            ) as bar:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            return dest   # success

        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as exc:
            wait = 2 ** attempt
            print(f"\n  Network error (attempt {attempt}/{max_retries}): {exc}")
            if attempt < max_retries:
                print(f"  Retrying in {wait}s …")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Download failed after {max_retries} attempts: {url}"
                ) from exc

    return dest   # unreachable, but satisfies type checkers


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
    """COCO 2017 validation set -- 5,000 real photos, ~800 MB, CC BY 4.0."""
    print("\n-- COCO 2017 val --------------------------------------")
    dest_dir = DATA_REAL / "coco_val"
    archive  = dest_dir / "val2017.zip"

    # Always validate the zip before trusting it.
    if _is_valid_zip(archive):
        print(f"Archive already downloaded and valid ({archive.stat().st_size // 1024**2} MB), skipping download.")
    else:
        if archive.exists():
            size_mb = archive.stat().st_size // 1024**2
            print(f"Existing archive is corrupt or incomplete ({size_mb} MB) — re-downloading.")
            archive.unlink()

        print("Downloading COCO 2017 val (~800 MB) — this takes several minutes.")
        print("Will retry automatically on timeout.")
        download_file(
            "http://images.cocodataset.org/zips/val2017.zip",
            archive,
            "COCO 2017 val",
            timeout=120,
            max_retries=5,
        )

        # Validate after download
        if not _is_valid_zip(archive):
            archive.unlink(missing_ok=True)
            raise RuntimeError(
                "val2017.zip failed zip validation after download. "
                "Try again — the COCO server may have dropped the connection."
            )

    # Only extract if the target folder doesn't already have images
    val_dir = dest_dir / "val2017"
    if val_dir.exists() and len(list(val_dir.glob("*.jpg"))) > 4000:
        print(f"Already extracted ({len(list(val_dir.glob('*.jpg')))} images found).")
    else:
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
    """DIV2K -- 1,000 high-resolution real photos, CC Free research."""
    print("\n-- DIV2K ----------------------------------------------")
    dest_dir = DATA_REAL / "div2k"
    dest_dir.mkdir(parents=True, exist_ok=True)

    urls = [
        ("https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip", "DIV2K train HR"),
        ("https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip", "DIV2K valid HR"),
    ]

    for url, desc in urls:
        archive = dest_dir / Path(url).name
        if _is_valid_zip(archive):
            print(f"{archive.name} already downloaded and valid, skipping.")
        else:
            if archive.exists():
                print(f"{archive.name} is corrupt — re-downloading.")
                archive.unlink()
            download_file(url, archive, desc, timeout=120, max_retries=5)
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
    RAISE-1k -- 1,000 raw uncompressed DSLR photos.
    Direct download requires registration at http://loki.disi.unitn.it/RAISE/
    This script prints the instructions since it requires a form submission.
    """
    print("\n-- RAISE-1k -------------------------------------------")
    print("RAISE requires a registration form -- cannot be automated.")
    print("Steps:")
    print("  1. Go to: http://loki.disi.unitn.it/RAISE/")
    print("  2. Fill the form to request download links")
    print("  3. Download the ZIP files you receive by email")
    print("  4. Extract to: data/real/raise_1k/")
    print("  5. Run: python scripts/datasets/index_manual.py --source raise_1k --label real")
    print("Skipping automated download.")


def download_unsplash():
    """
    Unsplash Lite -- 25,000 professional photos.
    Uses the Unsplash dataset GitHub release.
    """
    print("\n-- Unsplash Lite --------------------------------------")
    dest_dir = DATA_REAL / "unsplash"
    dest_dir.mkdir(parents=True, exist_ok=True)

    csv_url = (
        "https://github.com/unsplash/datasets/releases/download/"
        "v1.2.0/lite-00000-of-00001.tsv"
    )
    csv_dest = dest_dir / "unsplash_lite.tsv"

    if not csv_dest.exists():
        download_file(csv_url, csv_dest, "Unsplash Lite metadata")
    else:
        print("Metadata already downloaded.")

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
