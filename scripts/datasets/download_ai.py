"""
Download AI-generated datasets.
Run from repo root: python scripts/datasets/download_ai.py --dataset cifake

Requires for Kaggle datasets:
  1. Go to kaggle.com -> Account -> Create API Token
  2. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json
  3. chmod 600 ~/.kaggle/kaggle.json
"""
import os
import sys
import csv
import argparse
import requests
from pathlib import Path
from tqdm import tqdm
from PIL import Image

ROOT    = Path(__file__).resolve().parents[2]
DATA_AI = ROOT / "data" / "ai"
MANIFEST = ROOT / "data" / "manifest.csv"


# -- Shared helpers (same as download_real.py) ------------------------------

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
        import importlib
        if importlib.util.find_spec("kaggle") is None:
            print("Install kaggle: pip install kaggle")
            import sys; sys.exit(1)
    except Exception:
        pass

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from Kaggle: {dataset_slug}")
    os.system(
        f"kaggle datasets download -d {dataset_slug} "
        f"--path {dest_dir} --unzip"
    )


# -- Dataset downloaders ----------------------------------------------------

def download_cifake():
    """
    CIFAKE — 120,000 images (60K real CIFAR-10, 60K SD 1.4 generated).
    We only use the FAKE split (60K AI images).
    License: CC BY 4.0
    """
    print("\n-- CIFAKE ---------------------------------------------")
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
                "path":      img_path.relative_to(ROOT).as_posix(),
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
    print("\n-- Fake vs Real Faces ---------------------------------")
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
                "path":      img_path.relative_to(ROOT).as_posix(),
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
    DiffusionDB -- downloads a 1K zip directly from HuggingFace file storage.
    Uses the part-000001.zip which contains 1000 SD 1.4 images at full resolution.
    No loading script needed -- direct HTTP download.
    License: CC0 1.0
    """
    print("\n-- DiffusionDB (direct zip download) -----------------")
    import zipfile, io, requests as _req

    dest_dir = DATA_AI / "stable_diffusion"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Each zip has 1000 images. Download a few zips to get ~5000 images.
    base_url = "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/images"
    zip_ids  = [1, 2, 3, 4, 5]   # part-000001.zip ... part-000005.zip

    rows  = []
    saved = 0
    for zip_id in zip_ids:
        fname  = f"part-{zip_id:06d}.zip"
        url    = f"{base_url}/{fname}"
        print(f"  Downloading {fname}...")
        try:
            r = _req.get(url, timeout=120, stream=True)
            r.raise_for_status()
            data = b"".join(r.iter_content(65536))
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for name in z.namelist():
                    if not name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                        continue
                    try:
                        from PIL import Image as _PIL
                        img_bytes = z.read(name)
                        img       = _PIL.open(io.BytesIO(img_bytes)).convert("RGB")
                        w, h      = img.size
                        if w < 256 or h < 256:
                            continue
                        out_path = dest_dir / f"diffdb_{saved:06d}.jpg"
                        img.save(out_path, quality=90)
                        rows.append({
                            "path":      out_path.relative_to(ROOT).as_posix(),
                            "label":     "ai",
                            "source":    "diffusiondb",
                            "generator": "stable_diffusion_1.4",
                            "split":     assign_split(saved, 5000),
                            "width":     w,
                            "height":    h,
                            "has_exif":  False,
                            "verified":  True,
                        })
                        saved += 1
                    except Exception as e:
                        print(f"    Skipped {name}: {e}")
            print(f"    {saved} images so far")
        except Exception as e:
            print(f"  Failed {fname}: {e} -- skipping")

    add_to_manifest(rows)
    print(f"DiffusionDB done: {len(rows)} images added.")


def download_genimage_subset():
    """
    Defactify MS-COCOAI dataset -- real + AI images at matched resolutions.
    Uses SD3, SD2.1, SDXL, DALL-E 3, MidJourney v6 generators.
    Available on HuggingFace as Parquet files (no loading script).
    License: research use
    """
    print("\n-- Defactify MS-COCOAI (AI images, full resolution) --")
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: pip install datasets")
        sys.exit(1)

    LIMIT    = 5000
    dest_dir = DATA_AI / "defactify"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Try Defactify / MS-COCOAI first, then CortexLM/midjourney-v6 as fallback
    candidates = [
        ("Rajarshi-Roy-research/Defactify_Image_Dataset", {"split": "train", "streaming": True}),
        ("CortexLM/midjourney-v6",                        {"split": "train", "streaming": True}),
    ]

    ds = None
    for repo, kwargs in candidates:
        try:
            ds = load_dataset(repo, **kwargs)
            print(f"  Loaded from {repo}")
            break
        except Exception as e:
            print(f"  {repo} failed: {e}")

    if ds is None:
        print("  All sources failed. Skipping genimage/defactify step.")
        return

    rows  = []
    saved = 0
    from tqdm import tqdm as _tqdm
    for i, item in enumerate(_tqdm(ds, total=LIMIT, desc="Defactify")):
        if saved >= LIMIT:
            break
        try:
            # Defactify labels: 'real' or 'fake'/'ai'. Keep only AI images.
            lbl = str(item.get("label", item.get("is_ai", "ai"))).lower()
            if lbl in ("real", "1", "human"):
                continue
            img = item.get("image") or item.get("jpg") or item.get("img")
            if img is None:
                continue
            if not hasattr(img, "size"):
                from PIL import Image as _PIL
                import io
                img = _PIL.open(io.BytesIO(img))
            generator = str(item.get("generator", item.get("source", "unknown"))).lower().replace(" ", "_")
            img_path  = dest_dir / f"defactify_{saved:06d}.jpg"
            img.convert("RGB").save(img_path, quality=90)
            w, h = img.size
            if w < 256 or h < 256:
                img_path.unlink(missing_ok=True)
                continue
            rows.append({
                "path":      img_path.relative_to(ROOT).as_posix(),
                "label":     "ai",
                "source":    "defactify",
                "generator": generator,
                "split":     assign_split(saved, LIMIT),
                "width":     w,
                "height":    h,
                "has_exif":  False,
                "verified":  True,
            })
            saved += 1
        except Exception as e:
            print(f"  Skipped {i}: {e}")

    add_to_manifest(rows)
    print(f"Defactify done: {len(rows)} AI images added.")

DATASETS = {
    "cifake":       download_cifake,
    "faces":        download_fake_real_faces,
    "diffusiondb":  download_diffusiondb,
    "genimage":     download_genimage_subset,
    "all":          None,
}

if __name__ == "__main__":
    import argparse
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
