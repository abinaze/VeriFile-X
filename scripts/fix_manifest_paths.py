"""
One-time utility: normalise all backslash paths in manifest.csv to forward
slashes. Run this once after any download on Windows, then delete this file.

Usage:
    python scripts/fix_manifest_paths.py
"""
import csv
import sys
from pathlib import Path

ROOT     = Path(__file__).parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"

def main():
    if not MANIFEST.exists():
        print(f"manifest.csv not found at {MANIFEST}")
        sys.exit(1)

    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []

    fixed = 0
    for row in rows:
        p = row.get("path", "")
        clean = p.replace("\\", "/")
        if clean != p:
            row["path"] = clean
            fixed += 1

    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Done. Fixed {fixed} backslash paths out of {len(rows)} total rows.")
    print(f"Manifest: {MANIFEST}")

if __name__ == "__main__":
    main()
