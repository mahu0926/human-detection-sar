"""On-the-fly gate samples. Original images on disk are never modified.

Every gate input goes through the same geometry: downscale so the short side is
CACHE_SIDE, then crop a square (random for train, center otherwise). There is
no letterbox padding, because padding bars reveal the aspect ratio and Heridal
is the only 4:3 source, so the gate could learn "padding = vegetation".

Training also randomises JPEG quality, resolution and blur so the gate cannot
use sharpness/compression differences between 4000px and 1920px sources.
The experts always receive the original full-resolution file.

Decoding 4K JPEGs every epoch is slow, so ``build_gate_cache.py`` writes a
short-side-320 copy of every image to ``gate_cache/`` once. The dataset uses the
cache when present and falls back to the original file otherwise.
"""

from __future__ import annotations

import random
from collections import Counter

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

from moe.cache import jpeg_roundtrip, load_gate_image
from moe.data import CLASS_NAMES, ImageRecord, build_index

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RandomJPEG:
    def __init__(self, quality=(30, 95), p: float = 0.3):
        self.quality, self.p = quality, p

    def __call__(self, image):
        if random.random() >= self.p:
            return image
        return jpeg_roundtrip(image, random.randint(*self.quality))


class RandomDownUp:
    """Downsample then upsample back, so native sharpness is not a usable cue."""

    def __init__(self, scale=(0.35, 0.9), p: float = 0.3):
        self.scale, self.p = scale, p

    def __call__(self, image):
        if random.random() >= self.p:
            return image
        factor = random.uniform(*self.scale)
        width, height = image.size
        small = image.resize((max(8, int(width * factor)), max(8, int(height * factor))), Image.BILINEAR)
        return small.resize((width, height), Image.BILINEAR)


def build_transforms(input_size: int, train: bool) -> T.Compose:
    normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    resize_side = int(round(input_size * 256 / 224))
    if train:
        return T.Compose(
            [
                RandomDownUp(),
                RandomJPEG(),
                T.RandomResizedCrop(input_size, scale=(0.25, 1.0), ratio=(0.75, 1.333)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.03),
                T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.2),
                T.ToTensor(),
                normalize,
            ]
        )
    return T.Compose(
        [
            T.Resize(resize_side),
            T.CenterCrop(input_size),
            T.ToTensor(),
            normalize,
        ]
    )


def build_tta_transform(input_size: int):
    """Five crops (4 corners + center) stacked into one batch."""
    normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    resize_side = int(round(input_size * 256 / 224))
    to_tensor = T.Compose([T.ToTensor(), normalize])
    return T.Compose(
        [
            T.Resize(resize_side),
            T.FiveCrop(input_size),
            T.Lambda(lambda crops: torch.stack([to_tensor(crop) for crop in crops])),
        ]
    )


class GateDataset(Dataset):
    """Unified 3-class dataset. ``label`` is the source dataset, not a box class."""

    def __init__(
        self,
        records: list[ImageRecord],
        transform,
        return_path: bool = False,
    ):
        self.records = list(records)
        self.transform = transform
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        try:
            image = load_gate_image(record.path)
        except OSError as exc:
            raise OSError(f"Unreadable image: {record.path}") from exc
        tensor = self.transform(image)
        if self.return_path:
            return tensor, record.class_id, str(record.path)
        return tensor, record.class_id


def class_counts(records: list[ImageRecord]) -> dict[str, Counter]:
    """Per-split counts keyed by class name."""
    table: dict[str, Counter] = {}
    for record in records:
        bucket = table.setdefault(record.split, Counter())
        bucket[record.class_name] += 1
    return table


def format_counts(records: list[ImageRecord]) -> str:
    lines = []
    counts = class_counts(records)
    header = f"{'split':<8}" + "".join(f"{name:>12}" for name in CLASS_NAMES) + f"{'total':>10}"
    lines.append(header)
    for split in ("train", "val", "test"):
        bucket = counts.get(split, Counter())
        row = f"{split:<8}"
        total = 0
        for name in CLASS_NAMES:
            value = bucket.get(name, 0)
            total += value
            row += f"{value:>12}"
        row += f"{total:>10}"
        lines.append(row)
    return "\n".join(lines)


def make_weighted_sampler(records: list[ImageRecord], generator=None) -> WeightedRandomSampler:
    """Inverse-frequency sampler so a large maritime set cannot dominate a step."""
    counts = Counter(record.class_id for record in records)
    missing = [class_id for class_id in range(len(CLASS_NAMES)) if counts[class_id] == 0]
    if missing:
        names = [CLASS_NAMES[class_id] for class_id in missing]
        raise RuntimeError(f"Training split has no images for: {', '.join(names)}")
    weights = [1.0 / counts[record.class_id] for record in records]
    return WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def split_records(config: dict) -> dict[str, list[ImageRecord]]:
    grouped = {"train": [], "val": [], "test": []}
    records, _excluded = build_index(config)
    for record in records:
        if record.split not in grouped:
            raise RuntimeError(f"Unexpected split {record.split!r} on {record.path}")
        grouped[record.split].append(record)
    return grouped


def make_datasets(config: dict, return_paths: bool = False) -> dict[str, GateDataset]:
    size = int(config["input_size"])
    grouped = split_records(config)
    return {
        "train": GateDataset(grouped["train"], build_transforms(size, train=True)),
        "val": GateDataset(
            grouped["val"],
            build_transforms(size, train=False),
            return_path=return_paths,
        ),
        "test": GateDataset(
            grouped["test"],
            build_transforms(size, train=False),
            return_path=return_paths,
        ),
    }


def make_loader(
    dataset: GateDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    sampler: WeightedRandomSampler | None = None,
    pin_memory: bool = False,
) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)
