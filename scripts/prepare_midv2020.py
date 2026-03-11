"""
MIDV-2020 Dataset Preparation Script
======================================

Downloads and prepares MIDV-2020 for training the document fraud detector.

What this script downloads
--------------------------
  MIDV-2020  — Photo/ and Scan_upright/ subsets only (~2–4 GB, not the full 124 GB)
               Contains 1,000 real mock identity documents (passports, ID cards, etc.)
               with artificially generated faces — safe for research use.

  COCO val2017 — ~1 GB subset of random non-document photos used as the
                 not_document training class.

How to get MIDV-2020 access
----------------------------
  1. Open:  https://l3i-share.univ-lr.fr/MIDV2020/midv2020.html
  2. Fill in the access request form and accept the license
  3. You will receive sFTP credentials by email
  4. Run this script with those credentials:

       venv/bin/python scripts/prepare_midv2020.py \\
           --sftp_host  smartengines.com \\
           --sftp_user  YOUR_USERNAME    \\
           --sftp_pass  YOUR_PASSWORD    \\
           --output     data

  If you already have the dataset downloaded locally, use --local_midv instead:

       venv/bin/python scripts/prepare_midv2020.py \\
           --local_midv /path/to/midv2020 \\
           --output     data

Output
------
  data/
  ├── real_documents/   ← MIDV-2020 Photos + Scans resized to 800px wide
  └── random_photos/    ← COCO val2017 sample (5,000 images)
"""

import argparse
import io
import os
import random
import shutil
import tarfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import urllib.request

# ── constants ────────────────────────────────────────────────────────────────

COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_N_SAMPLES = 5000      # how many COCO images to keep
MAX_LONG_EDGE = 800        # resize documents to this width (saves disk + speeds training)

# MIDV-2020 sFTP paths (relative to the server root)
MIDV_PHOTO_PATH = "/midv-2020/Photo"
MIDV_SCAN_PATH  = "/midv-2020/Scan_upright"

# ── helpers ──────────────────────────────────────────────────────────────────


def _resize_keep_aspect(image: np.ndarray, max_long_edge: int) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= max_long_edge:
        return image
    scale = max_long_edge / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _save_image(image: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])


def _progress(desc: str, current: int, total: int) -> None:
    pct = int(current / max(total, 1) * 100)
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    print(f"\r  {desc}  [{bar}] {pct}%  ({current}/{total})", end="", flush=True)


# ── MIDV-2020 via sFTP ────────────────────────────────────────────────────────


def _download_midv_sftp(host: str, user: str, password: str, output_dir: str) -> int:
    """Download MIDV-2020 Photo + Scan_upright via sFTP using paramiko."""
    try:
        import paramiko
    except ImportError:
        raise SystemExit(
            "\n[ERROR] paramiko is required for sFTP download.\n"
            "  Install it with:  venv/bin/pip install paramiko\n"
            "  Then re-run this script."
        )

    real_dir = os.path.join(output_dir, "real_documents")
    os.makedirs(real_dir, exist_ok=True)

    transport = paramiko.Transport((host, 22))
    transport.connect(username=user, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)

    count = 0
    for remote_base in [MIDV_PHOTO_PATH, MIDV_SCAN_PATH]:
        label = "Photo" if "Photo" in remote_base else "Scan"
        print(f"\nDownloading MIDV-2020 {label}/ …")
        try:
            entries = sftp.listdir(remote_base)
        except Exception as e:
            print(f"  [WARN] Could not list {remote_base}: {e}")
            continue

        for entry in entries:
            remote_path = f"{remote_base}/{entry}"
            local_path = os.path.join(real_dir, f"midv_{label}_{entry}")
            try:
                sftp.get(remote_path, local_path)
                # If it's a tar/zip, extract and remove archive
                if local_path.endswith(".tar") or local_path.endswith(".tar.gz"):
                    with tarfile.open(local_path) as tf:
                        tf.extractall(real_dir)
                    os.remove(local_path)
                count += 1
                _progress(f"MIDV {label}", count, len(entries))
            except Exception as e:
                print(f"\n  [WARN] Skipping {entry}: {e}")

    sftp.close()
    transport.close()
    print(f"\nDownloaded {count} MIDV-2020 files")
    return count


# ── MIDV-2020 from local path ─────────────────────────────────────────────────


def _process_local_midv(local_path: str, output_dir: str) -> int:
    """Process a locally downloaded MIDV-2020 folder into real_documents/."""
    real_dir = os.path.join(output_dir, "real_documents")
    os.makedirs(real_dir, exist_ok=True)

    # Collect all image files (TIFF, JPG, PNG)
    img_exts = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"}
    all_files = []
    for root, _, files in os.walk(local_path):
        for f in files:
            if Path(f).suffix.lower() in img_exts:
                all_files.append(os.path.join(root, f))

    if not all_files:
        raise SystemExit(f"[ERROR] No image files found in {local_path}")

    print(f"Processing {len(all_files)} MIDV-2020 images …")
    saved = 0
    for i, src in enumerate(all_files):
        img = cv2.imread(src, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = _resize_keep_aspect(img, MAX_LONG_EDGE)
        out = os.path.join(real_dir, f"midv_{i:06d}.jpg")
        _save_image(img, out)
        saved += 1
        if i % 50 == 0:
            _progress("Processing", i + 1, len(all_files))

    print(f"\nSaved {saved} document images → {real_dir}")
    return saved


# ── COCO random photos ────────────────────────────────────────────────────────


def _download_coco(output_dir: str, n_samples: int) -> int:
    """Download COCO val2017, sample n_samples images into random_photos/."""
    photo_dir = os.path.join(output_dir, "random_photos")
    os.makedirs(photo_dir, exist_ok=True)

    # Check how many we already have
    existing = list(Path(photo_dir).glob("coco_*.jpg"))
    if len(existing) >= n_samples:
        print(f"random_photos/ already has {len(existing)} images — skipping COCO download.")
        return len(existing)

    zip_path = os.path.join(output_dir, "_coco_val2017.zip")

    if not os.path.exists(zip_path):
        print(f"Downloading COCO val2017 (~1 GB) …")

        def _report(block, block_size, total):
            downloaded = block * block_size
            if total > 0:
                _progress("COCO", min(downloaded, total), total)

        urllib.request.urlretrieve(COCO_VAL_URL, zip_path, _report)
        print()

    print(f"Extracting COCO images (sampling {n_samples}) …")
    saved = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".jpg")]
        random.shuffle(members)
        members = members[:n_samples]

        for i, member in enumerate(members):
            data = zf.read(member)
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = _resize_keep_aspect(img, MAX_LONG_EDGE)
            out = os.path.join(photo_dir, f"coco_{i:05d}.jpg")
            _save_image(img, out)
            saved += 1
            if i % 100 == 0:
                _progress("Extracting COCO", i + 1, len(members))

    print(f"\nSaved {saved} COCO images → {photo_dir}")

    # Clean up zip to save disk space
    os.remove(zip_path)
    print("Removed COCO zip archive.")
    return saved


# ── summary ───────────────────────────────────────────────────────────────────


def _print_summary(output_dir: str) -> None:
    real_dir = os.path.join(output_dir, "real_documents")
    photo_dir = os.path.join(output_dir, "random_photos")

    n_real = len(list(Path(real_dir).glob("*"))) if os.path.isdir(real_dir) else 0
    n_photo = len(list(Path(photo_dir).glob("*"))) if os.path.isdir(photo_dir) else 0

    print("\n" + "═" * 55)
    print("  Dataset ready!")
    print(f"  real_documents : {n_real:,} images  →  {real_dir}")
    print(f"  random_photos  : {n_photo:,} images  →  {photo_dir}")
    print("═" * 55)
    print("\nNext step — train the fraud detector:")
    print(
        f"  venv/bin/python scripts/train_doc_fraud_detector.py \\\n"
        f"      --real_docs {real_dir} \\\n"
        f"      --photos    {photo_dir} \\\n"
        f"      --output    models/doc_fraud \\\n"
        f"      --epochs    30\n"
    )


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MIDV-2020 + COCO data for document fraud detector training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = parser.add_mutually_exclusive_group()
    src.add_argument("--sftp_host", help="MIDV-2020 sFTP host (e.g. smartengines.com)")
    src.add_argument(
        "--local_midv",
        help="Path to already-downloaded MIDV-2020 folder (skip sFTP)",
    )

    parser.add_argument("--sftp_user", help="sFTP username")
    parser.add_argument("--sftp_pass", help="sFTP password")
    parser.add_argument("--output", default="data", help="Output base directory (default: data)")
    parser.add_argument(
        "--coco_samples",
        type=int,
        default=COCO_N_SAMPLES,
        help=f"Number of COCO images to download (default: {COCO_N_SAMPLES})",
    )
    parser.add_argument(
        "--skip_coco",
        action="store_true",
        help="Skip COCO download (if you already have random_photos/)",
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ── 1. MIDV-2020 documents ──
    if args.local_midv:
        _process_local_midv(args.local_midv, args.output)
    elif args.sftp_host:
        if not args.sftp_user or not args.sftp_pass:
            raise SystemExit("[ERROR] --sftp_user and --sftp_pass are required with --sftp_host")
        _download_midv_sftp(args.sftp_host, args.sftp_user, args.sftp_pass, args.output)
    else:
        print(
            "\n[INFO] No MIDV-2020 source specified.\n"
            "\nTo get access to MIDV-2020:\n"
            "  1. Go to:  https://l3i-share.univ-lr.fr/MIDV2020/midv2020.html\n"
            "  2. Fill in the access request form and accept the license\n"
            "  3. You will receive sFTP credentials by email\n"
            "  4. Re-run with:\n\n"
            "     venv/bin/python scripts/prepare_midv2020.py \\\n"
            "         --sftp_host smartengines.com \\\n"
            "         --sftp_user YOUR_USERNAME    \\\n"
            "         --sftp_pass YOUR_PASSWORD    \\\n"
            "         --output    data\n\n"
            "  OR if you already downloaded MIDV-2020 locally:\n\n"
            "     venv/bin/python scripts/prepare_midv2020.py \\\n"
            "         --local_midv /path/to/midv2020 \\\n"
            "         --output     data\n"
        )
        print("\nContinuing with COCO download only (random_photos class) …\n")

    # ── 2. COCO random photos ──
    if not args.skip_coco:
        _download_coco(args.output, args.coco_samples)

    _print_summary(args.output)


if __name__ == "__main__":
    main()
