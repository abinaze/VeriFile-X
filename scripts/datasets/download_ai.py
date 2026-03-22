"""
Download AI-generated datasets.
Run from repo root: python scripts/datasets/download_ai.py --dataset cifake

Requires for Kaggle datasets:
  1. Go to kaggle.com → Account → Create API Token
  2. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json
  3. chmod 600 ~/.kaggle/kaggle.json
"""
import os
import sys
import csv
import argparse
import zipfile
import shutil
import requests
from pathlib import Path
from tqdm import tqdm
from PIL import Image

ROOT    = Path(__file__).resolve().parents[2]
DATA_AI = ROOT / "data" / "ai"
MANIFEST = ROOT / "data" / "manifest.csv"


# ── Shared helpers (same as download_real.py) ──────────────────────────────

def download_file(url: str, dest: Path, desc: str = "") -> Path:
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


def add_to_manifest(rows: list[dict]):
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
    try:
        img  = Image.open(path)
        w, h = img.size
        exif = img._getexif() if hasattr(img, "_getexif") else None
        return {"width": w, "height": h, "has_exif": exif is not None}
    except Exception:
        return {"width": 0, "height": 0, "has_exif": False}


def assign_split(index: int, total: int) -> str:
    pct = index / total
    if pct < 0.80: return "train"
    if pct < 0.90: return "val"
    return "test"


def kaggle_download(dataset_slug: str, dest_dir: Path):
    """Download a Kaggle dataset. Requires ~/.kaggle/kaggle.json"""
    try:
        import kaggle
    except ImportError:
        print("Install kaggle: pip install kaggle")
        sys.exit(1)

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from Kaggle: {dataset_slug}")
    os.system(
        f"kaggle datasets download -d {dataset_slug} "
        f"--path {dest_dir} --unzip"
    )


# ── Dataset downloaders ────────────────────────────────────────────────────

def download_cifake():
    """
    CIFAKE — 120,000 images (60K real CIFAR-10, 60K SD 1.4 generated).
    We only use the FAKE split (60K AI images).
    License: CC BY 4.0
    """
    print("\n── CIFAKE ─────────────────────────────────────────────")
    dest_dir = DATA_AI / "cifake"

    kaggle_download(
        "birdy654/cifake-real-and-ai-generated-synthetic-images",
        dest_dir
    )

    # CIFAKE structure: test/FAKE/, train/FAKE/
    rows = []
    for split_folder in ["train", "test"]:
        fake_dir = dest_dir / split_folder / "FAKE"
        if not fake_dir.exists():
            print(f"  {fake_dir} not found, skipping")
            continue

        images = list(fake_dir.glob("*.jpg")) + list(fake_dir.glob("*.png"))
        print(f"  {split_folder}/FAKE: {len(images)} images")

        for i, img_path in enumerate(sorted(images)):
            info = get_image_info(img_path)
            rows.append({
                "path":      str(img_path.relative_to(ROOT)),
                "label":     "ai",
                "source":    "cifake",
                "generator": "stable_diffusion_1.4",
                "split":     split_folder if split_folder == "test" else assign_split(i, len(images)),
                "width":     info["width"],
                "height":    info["height"],
                "has_exif":  False,
                "verified":  True,
            })

    add_to_manifest(rows)
    print(f"CIFAKE done: {len(rows)} AI images added.")


def download_fake_real_faces():
    """
    140K Real and Fake Faces — StyleGAN + ProGAN generated.
    License: CC0
    """
    print("\n── Fake vs Real Faces ─────────────────────────────────")
    dest_dir = DATA_AI / "stylegan_faces"

    kaggle_download(
        "xhlulu/140k-real-and-fake-faces",
        dest_dir
    )

    # Only take the FAKE images
    fake_dirs = [
        dest_dir / "real_vs_fake" / "test" / "fake",
        dest_dir / "real_vs_fake" / "train" / "fake",
        dest_dir / "real_vs_fake" / "valid" / "fake",
    ]

    rows = []
    for fake_dir in fake_dirs:
        if not fake_dir.exists():
            continue
        images = list(fake_dir.glob("*.jpg")) + list(fake_dir.glob("*.png"))
        split  = fake_dir.parts[-2]  # train / test / valid
        if split == "valid":
            split = "val"

        for img_path in sorted(images):
            info = get_image_info(img_path)
            if info["width"] < 256:
                continue
            rows.append({
                "path":      str(img_path.relative_to(ROOT)),
                "label":     "ai",
                "source":    "stylegan_faces",
                "generator": "stylegan2",
                "split":     split,
                "width":     info["width"],
                "height":    info["height"],
                "has_exif":  False,
                "verified":  True,
            })

    add_to_manifest(rows)
    print(f"Fake faces done: {len(rows)} AI images added.")


def download_diffusiondb():
    """
    DiffusionDB — Stable Diffusion 1.4/2.0 images from HuggingFace.
    Streams 10,000 images without downloading the full 1.6TB dataset.
    License: CC BY 4.0
    """
    print("\n── DiffusionDB (streaming 10K) ────────────────────────")
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: pip install datasets")
        sys.exit(1)

    dest_dir = DATA_AI / "stable_diffusion"
    dest_dir.mkdir(parents=True, exist_ok=True)

    LIMIT = 10_000
    print(f"Streaming {LIMIT} images from HuggingFace (no full download)...")

    ds = load_dataset(
        "poloclub/diffusiondb",
        "2m_first_1k",   # smallest subset to start
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    rows = []
    for i, item in enumerate(tqdm(ds, total=LIMIT, desc="DiffusionDB")):
        if i >= LIMIT:
            break
        try:
            img      = item["image"]  # PIL Image
            img_path = dest_dir / f"diffdb_{i:06d}.jpg"
            img.convert("RGB").save(img_path, quality=90)
            w, h = img.size
            rows.append({
                "path":      str(img_path.relative_to(ROOT)),
                "label":     "ai",
                "source":    "diffusiondb",
                "generator": "stable_diffusion_1.4",
                "split":     assign_split(i, LIMIT),
                "width":     w,
                "height":    h,
                "has_exif":  False,
                "verified":  True,
            })
        except Exception as e:
            print(f"  Skipped item {i}: {e}")
            continue

    add_to_manifest(rows)
    print(f"DiffusionDB done: {len(rows)} images added.")


def download_genimage_subset():
    """
    GenImage — 8 generators including MJ, SDXL, DALL-E.
    Streams a 10K subset from HuggingFace.
    License: CC BY-NC
    """
    print("\n── GenImage subset (10K) ──────────────────────────────")
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: pip install datasets")
        sys.exit(1)

    LIMIT = 10_000
    dest_dir = DATA_AI / "stable_diffusion"
    dest_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "ImageNet-GenImage/GenImage",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    rows = []
    for i, item in enumerate(tqdm(ds, total=LIMIT, desc="GenImage")):
        if i >= LIMIT:
            break
        if item.get("label") != 0:   # 0 = AI, 1 = real in GenImage
            continue
        try:
            img       = item["image"]
            generator = item.get("source", "unknown").lower().replace(" ", "_")
            img_path  = dest_dir / f"genimage_{i:06d}.jpg"
            img.convert("RGB").save(img_path, quality=90)
            w, h = img.size
            rows.append({
                "path":      str(img_path.relative_to(ROOT)),
                "label":     "ai",
                "source":    "genimage",
                "generator": generator,
                "split":     assign_split(i, LIMIT),
                "width":     w,
                "height":    h,
                "has_exif":  False,
                "verified":  True,
            })
        except Exception as e:
            print(f"  Skipped {i}: {e}")

    add_to_manifest(rows)
    print(f"GenImage done: {len(rows)} AI images added.")


# ── Entry point ────────────────────────────────────────────────────────────

DATASETS = {
    "cifake":       download_cifake,
    "faces":        download_fake_real_faces,
    "diffusiondb":  download_diffusiondb,
    "genimage":     download_genimage_subset,
    "all":          None,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download AI-generated datasets")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        required=True,
        help="Which dataset to download"
    )
    args = parser.parse_args()

    if args.dataset == "all":
        download_cifake()
        download_fake_real_faces()
        download_diffusiondb()
        download_genimage_subset()
    else:
        DATASETS[args.dataset]()

    print("\nDone. Check data/manifest.csv")
