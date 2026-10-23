"""Torch-free cache helpers shared by gate_dataset.py and build_gate_cache.py."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from moe.data import project_root

CACHE_SIDE = 320          # short side of cached / routed gate images
CACHE_QUALITY = 95
CACHE_DIR = "gate_cache"


def downscale(image: Image.Image, side: int = CACHE_SIDE) -> Image.Image:
    """Resize so the short side equals ``side``. Uses JPEG draft mode for speed."""
    width, height = image.size
    if image.format == "JPEG" and min(width, height) > 2 * side:
        scale = (2 * side) / float(min(width, height))
        image.draft("RGB", (int(width * scale), int(height * scale)))
    image = image.convert("RGB")
    width, height = image.size
    scale = side / float(min(width, height))
    size = (max(side, round(width * scale)), max(side, round(height * scale)))
    return image.resize(size, Image.LANCZOS, reducing_gap=3.0)


def jpeg_roundtrip(image: Image.Image, quality: int = CACHE_QUALITY) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def prepare_gate_image(image: Image.Image) -> Image.Image:
    """Exactly what the cache stores. Used at routing time for train/infer parity."""
    return jpeg_roundtrip(downscale(image))


def cache_path_for(path: Path) -> Path:
    root = project_root()
    try:
        relative = Path(path).resolve().relative_to(root)
    except ValueError:
        relative = Path(*Path(path).resolve().parts[1:])
    return root / CACHE_DIR / relative.with_suffix(".jpg")


def load_gate_image(path: Path) -> Image.Image:
    cached = cache_path_for(path)
    if cached.is_file():
        with Image.open(cached) as image:
            return image.convert("RGB")
    with Image.open(path) as image:
        return prepare_gate_image(image)
