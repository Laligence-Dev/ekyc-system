"""
Document Fraud Detector — Training Script
==========================================

Trains an EfficientNet-B0 classifier to detect:
  Class 0 — real_document  : genuine passport / government ID
  Class 1 — not_document   : regular photos, selfies, random images
  Class 2 — fake_document  : edited, spliced, or digitally altered documents

Fake document samples are generated AUTOMATICALLY from your real document images
by simulating the most common eKYC forgery techniques:
  • Face photo region replacement   (swap face crop with one from a random photo)
  • Clone stamp / copy-paste        (copy region within the same document)
  • Region splice                   (paste a patch from another image)
  • JPEG recompression artifacts    (simulate post-edit re-save)

Usage
-----
    python scripts/train_doc_fraud_detector.py \
        --real_docs  data/real_documents \
        --photos     data/random_photos  \
        --output     models/doc_fraud    \
        --epochs     30

Data layout expected
--------------------
    data/
    ├── real_documents/   ← scans / photos of real passports or ID cards
    └── random_photos/    ← any non-document images (selfies, landscapes, …)

Output
------
    models/doc_fraud/
    ├── doc_fraud_detector.onnx   ← loaded by the API at runtime
    └── class_names.txt
"""

import argparse
import io
import json
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

# ── constants ────────────────────────────────────────────────────────────────

IMG_SIZE = 224
CLASSES = ["real_document", "not_document", "fake_document"]
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── fake-document generation ─────────────────────────────────────────────────


def _load_rgb(path: str) -> np.ndarray | None:
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _save_rgb(image: np.ndarray, path: str, quality: int = 92) -> None:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])


def _jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    """Re-encode as JPEG at given quality and decode — creates compression seams."""
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def fake_clone_stamp(image: np.ndarray) -> np.ndarray:
    """Copy a random patch from one location and paste it somewhere else.
    Simulates the clone-stamp / healing-brush used to hide text or stamps."""
    h, w = image.shape[:2]
    ph = random.randint(h // 8, h // 3)
    pw = random.randint(w // 8, w // 3)

    # Source patch
    sy = random.randint(0, h - ph)
    sx = random.randint(0, w - pw)
    patch = image[sy : sy + ph, sx : sx + pw].copy()

    # Destination — must differ from source
    for _ in range(20):
        dy = random.randint(0, h - ph)
        dx = random.randint(0, w - pw)
        if abs(dy - sy) > ph // 2 or abs(dx - sx) > pw // 2:
            break

    fake = image.copy()
    fake[dy : dy + ph, dx : dx + pw] = patch

    # Feather edges slightly so the seam is subtler (but still ELA-visible)
    mask = np.zeros((h, w), dtype=np.float32)
    mask[dy : dy + ph, dx : dx + pw] = 1.0
    kernel = max(3, min(ph, pw) // 6) | 1  # odd number
    mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)
    mask = mask[:, :, None]
    fake = (fake * mask + image * (1 - mask)).astype(np.uint8)

    # Re-encode at varied quality to bake in compression inconsistency
    return _jpeg_roundtrip(fake, random.randint(70, 90))


def fake_region_splice(image: np.ndarray, donor_images: list[np.ndarray]) -> np.ndarray | None:
    """Paste a region from a completely different image onto the document.
    Simulates photo-substitution or text-field replacement."""
    if not donor_images:
        return None

    donor = random.choice(donor_images)
    h, w = image.shape[:2]
    dh, dw = donor.shape[:2]

    ph = random.randint(h // 6, h // 2)
    pw = random.randint(w // 6, w // 2)

    if dh < ph or dw < pw:
        return None

    # Crop patch from donor
    sy = random.randint(0, dh - ph)
    sx = random.randint(0, dw - pw)
    patch = donor[sy : sy + ph, sx : sx + pw].copy()
    patch = cv2.resize(patch, (pw, ph))

    # Destination on document
    dy = random.randint(0, h - ph)
    dx = random.randint(0, w - pw)

    fake = image.copy()
    fake[dy : dy + ph, dx : dx + pw] = patch

    return _jpeg_roundtrip(fake, random.randint(65, 85))


def fake_face_swap(doc_image: np.ndarray, donor_images: list[np.ndarray]) -> np.ndarray | None:
    """Detect the face region in the document and replace it with a cropped face
    from a random donor photo — the most common eKYC attack."""
    try:
        import face_recognition  # optional — skip if unavailable
    except ImportError:
        return None

    if not donor_images:
        return None

    face_locs = face_recognition.face_locations(doc_image, model="hog")
    if not face_locs:
        return None

    top, right, bottom, left = face_locs[0]
    doc_face_h = bottom - top
    doc_face_w = right - left
    if doc_face_h <= 0 or doc_face_w <= 0:
        return None

    # Find a donor image that has a detectable face
    random.shuffle(donor_images)
    for donor in donor_images[:10]:
        donor_locs = face_recognition.face_locations(donor, model="hog")
        if not donor_locs:
            continue
        dt, dr, db, dl = donor_locs[0]
        face_crop = donor[dt:db, dl:dr]
        if face_crop.size == 0:
            continue

        face_resized = cv2.resize(face_crop, (doc_face_w, doc_face_h))
        fake = doc_image.copy()
        fake[top:bottom, left:right] = face_resized
        return _jpeg_roundtrip(fake, random.randint(75, 92))

    return None


def generate_fake_documents(
    real_paths: list[str],
    donor_paths: list[str],
    output_dir: str,
    n_samples: int,
) -> int:
    """Generate `n_samples` fake document images into `output_dir`.
    Returns the actual number generated."""
    os.makedirs(output_dir, exist_ok=True)
    donor_images = [img for p in donor_paths if (img := _load_rgb(p)) is not None]

    techniques = [fake_clone_stamp, fake_region_splice, fake_face_swap]
    generated = 0
    attempts = 0
    max_attempts = n_samples * 5

    while generated < n_samples and attempts < max_attempts:
        attempts += 1
        src_path = random.choice(real_paths)
        image = _load_rgb(src_path)
        if image is None:
            continue

        technique = random.choice(techniques)
        if technique is fake_clone_stamp:
            result = technique(image)
        else:
            result = technique(image, donor_images)

        if result is None:
            continue

        out_path = os.path.join(output_dir, f"fake_{generated:05d}.jpg")
        _save_rgb(result, out_path)
        generated += 1

    return generated


# ── dataset ──────────────────────────────────────────────────────────────────

_train_tf = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

_val_tf = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class DocDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def _collect_image_paths(folder: str) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = []
    for f in Path(folder).rglob("*"):
        if f.suffix.lower() in exts:
            paths.append(str(f))
    return paths


def build_datasets(
    real_dir: str,
    photo_dir: str,
    fake_dir: str,
    val_split: float = 0.15,
) -> tuple[DocDataset, DocDataset]:
    real_paths = [(p, 0) for p in _collect_image_paths(real_dir)]
    photo_paths = [(p, 1) for p in _collect_image_paths(photo_dir)]
    fake_paths = [(p, 2) for p in _collect_image_paths(fake_dir)]

    all_samples = real_paths + photo_paths + fake_paths
    random.shuffle(all_samples)

    split = int(len(all_samples) * (1 - val_split))
    train_samples = all_samples[:split]
    val_samples = all_samples[split:]

    print(
        f"Dataset — train: {len(train_samples)}  val: {len(val_samples)}\n"
        f"  real_document: {len(real_paths)}  not_document: {len(photo_paths)}  fake_document: {len(fake_paths)}"
    )
    return DocDataset(train_samples, _train_tf), DocDataset(val_samples, _val_tf)


# ── model ─────────────────────────────────────────────────────────────────────


def build_model(num_classes: int = 3, dropout: float = 0.4) -> nn.Module:
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


# ── training loop ─────────────────────────────────────────────────────────────


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    output_dir: str,
) -> nn.Module:
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        train_loss = train_correct = train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += images.size(0)

        scheduler.step()

        # ── validate ──
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        avg_loss = train_loss / max(train_total, 1)

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"loss={avg_loss:.4f}  "
            f"train_acc={train_acc:.3f}  "
            f"val_acc={val_acc:.3f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nBest val_acc: {best_val_acc:.4f}")
    model.load_state_dict(best_state)
    return model


# ── ONNX export ───────────────────────────────────────────────────────────────


def export_onnx(model: nn.Module, output_dir: str, device: torch.device) -> str:
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    onnx_path = os.path.join(output_dir, "doc_fraud_detector.onnx")

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported ONNX model → {onnx_path}")
    return onnx_path


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train eKYC document fraud detector")
    parser.add_argument("--real_docs", required=True, help="Folder of real document images")
    parser.add_argument("--photos", required=True, help="Folder of non-document photos")
    parser.add_argument("--output", default="models/doc_fraud", help="Output directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--fake_ratio",
        type=float,
        default=1.0,
        help="Ratio of fake samples relative to real (default 1.0 = same count)",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ── 1. generate fake documents ──
    real_paths = _collect_image_paths(args.real_docs)
    donor_paths = _collect_image_paths(args.photos)
    if not real_paths:
        raise SystemExit(f"No images found in --real_docs: {args.real_docs}")
    if not donor_paths:
        raise SystemExit(f"No images found in --photos: {args.photos}")

    n_fake = max(1, int(len(real_paths) * args.fake_ratio))
    fake_dir = os.path.join(args.output, "generated_fakes")
    print(f"Generating {n_fake} fake document samples …")
    actual = generate_fake_documents(real_paths, donor_paths, fake_dir, n_fake)
    print(f"Generated {actual} fake samples → {fake_dir}")

    # ── 2. build datasets ──
    train_ds, val_ds = build_datasets(args.real_docs, args.photos, fake_dir)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    # ── 3. build & train model ──
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(num_classes=len(CLASSES)).to(device)
    model = train(model, train_loader, val_loader, args.epochs, args.lr, device, args.output)

    # ── 4. export ──
    export_onnx(model, args.output, device)

    # Save class names so the inference service can read them
    with open(os.path.join(args.output, "class_names.txt"), "w") as f:
        f.write("\n".join(CLASSES))

    print("\nDone! Run the API and the document fraud detector will load automatically.")


if __name__ == "__main__":
    main()
