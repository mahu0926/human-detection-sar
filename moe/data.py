"""Discover gate-classification samples from the three detection datasets.

The gate label is which dataset an image came from (maritime / urban /
vegetation). Bounding-box files are located so they can be checked, but they
are not classification targets.

Official train/val/test folders are kept. A seeded 70/20/10 split is used only
when a dataset has no split folders at all. SeaDronesSee is marked
``preserve_splits``: frames come from video, so a missing split is an error
rather than a random re-split.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, replace
from pathlib import Path

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".gif",
    ".heic",
    ".ppm",
}
SKIP_DIR_NAMES = {".cache", ".git", "__pycache__", ".huggingface"}

SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}

CLASS_NAMES = ["maritime", "urban", "vegetation"]


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    dataset: str
    class_id: int
    class_name: str
    official_split: str
    split: str
    split_source: str
    label_path: Path | None


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def canonical_split(part: str) -> str | None:
    token = part.lower()
    if token in SPLIT_ALIASES:
        return SPLIT_ALIASES[token]
    for alias, canon in SPLIT_ALIASES.items():
        if token.endswith("_" + alias) or token.endswith("-" + alias):
            return canon
    return None


def split_from_relative(rel: Path) -> str:
    for part in rel.parts:
        found = canonical_split(part)
        if found:
            return found
    return "unsplit"


def mirrored_label_path(image_path: Path) -> Path:
    """Map ``.../images/.../name.jpg`` to ``.../labels/.../name.txt``."""
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def find_label(image_path: Path) -> Path | None:
    """Prefer a parallel ``labels`` file. Fall back to a ``.txt`` beside the image."""
    mirrored = mirrored_label_path(image_path)
    if mirrored.is_file():
        return mirrored
    sibling = image_path.with_suffix(".txt")
    if sibling != mirrored and sibling.is_file():
        return sibling
    return None


def iter_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in SKIP_DIR_NAMES and not name.startswith(".")
        ]
        for name in filenames:
            if name.startswith("."):
                continue
            yield Path(dirpath) / name


def _resolve_root(base: Path, spec_root: str) -> Path:
    path = Path(spec_root)
    if not path.is_absolute():
        path = base / path
    return path


def discover_images(root: Path, spec: dict) -> list[ImageRecord]:
    """Walk one dataset root and pair each image with its label file, if any."""
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    name = spec["name"]
    class_id = int(spec["class_id"])
    class_name = spec["class_name"]
    records: list[ImageRecord] = []

    for path in iter_files(root):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(root)
        official = split_from_relative(relative)
        records.append(
            ImageRecord(
                path=path,
                dataset=name,
                class_id=class_id,
                class_name=class_name,
                official_split=official,
                split=official,
                split_source="official" if official != "unsplit" else "unsplit",
                label_path=find_label(path),
            )
        )

    records.sort(key=lambda record: str(record.path))
    return records


def _assign_fractions(records: list[ImageRecord], seed: int) -> list[ImageRecord]:
    """Seeded 70/20/10. Paths are sorted before the shuffle so the split is stable."""
    ordered = sorted(records, key=lambda record: str(record.path))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    count = len(ordered)
    if count == 0:
        return []
    if count == 1:
        cuts = {"train": ordered, "val": [], "test": []}
    elif count == 2:
        cuts = {"train": ordered[:1], "val": ordered[1:], "test": []}
    else:
        n_train = max(1, int(round(count * 0.70)))
        n_val = int(round(count * 0.20))
        if n_train + n_val >= count:
            n_val = max(0, count - n_train - 1)
        n_test = count - n_train - n_val
        if n_val == 0 and n_test > 1:
            n_val = 1
            n_test -= 1
        cuts = {
            "train": ordered[:n_train],
            "val": ordered[n_train : n_train + n_val],
            "test": ordered[n_train + n_val : n_train + n_val + n_test],
        }

    assigned: list[ImageRecord] = []
    for split_name, group in cuts.items():
        for record in group:
            assigned.append(
                replace(record, split=split_name, split_source="seeded_70_20_10")
            )
    return assigned


def _id_key(record: ImageRecord):
    stem = record.path.stem
    return (0, int(stem), "") if stem.isdigit() else (1, 0, stem)


def carve_test_from_val(records: list[ImageRecord]) -> list[ImageRecord]:
    """Split val into two contiguous halves by frame ID (no shuffling).

    Lower IDs stay ``val`` (early stopping); upper IDs become ``test``. Keeping
    contiguous ID blocks together limits neighbouring-frame leakage between them.
    """
    val = sorted((r for r in records if r.split == "val"), key=_id_key)
    if len(val) < 2:
        return records
    upper = {r.path for r in val[len(val) // 2 :]}
    out = []
    for record in records:
        if record.path in upper:
            out.append(replace(record, split="test", split_source="val_upper_half_by_id"))
        else:
            out.append(record)
    return out


def finalize_splits(
    records: list[ImageRecord], spec: dict, seed: int
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Keep official splits. Randomly divide a dataset only when it has none.

    A dataset with train and val but no test keeps that hole. Inventing a test
    set from the other folders would reshuffle video frames. Images that sit
    outside a dataset which already has split folders (the SeaDronesSee README
    banner, for example) are excluded rather than randomly assigned.
    ``preserve_splits`` also refuses the no-split fallback.
    """
    official = [record for record in records if record.official_split != "unsplit"]
    loose = [record for record in records if record.official_split == "unsplit"]
    preserve = bool(spec.get("preserve_splits", False))

    if official:
        excluded = [
            replace(record, split="excluded", split_source="outside_official_splits")
            for record in loose
        ]
        if spec.get("carve_test_from_val") and not any(r.split == "test" for r in official):
            official = carve_test_from_val(official)
        return list(official), excluded

    if preserve:
        raise RuntimeError(
            f"{spec['name']} has no official train/val/test folders and "
            "preserve_splits is set. Refusing a random 70/20/10 split because "
            "frames from the same video would land in different splits."
        )
    return _assign_fractions(records, seed), []


def dataset_specs(config: dict | None = None) -> list[dict]:
    if config is None:
        raise ValueError("config is required")
    specs = config.get("datasets")
    if not specs:
        raise ValueError("config.yaml has no datasets")
    names = list(config.get("class_names", CLASS_NAMES))
    for spec in specs:
        class_id = int(spec["class_id"])
        if class_id < 0 or class_id >= len(names):
            raise ValueError(f"class_id {class_id} out of range for {spec['name']}")
        if names[class_id] != spec["class_name"]:
            raise ValueError(
                f"{spec['name']} class_name {spec['class_name']!r} does not match "
                f"class_names[{class_id}] = {names[class_id]!r}"
            )
    return specs


def build_index(config: dict, base: Path | None = None) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Return ``(kept, excluded)``.

    ``kept`` is what the gate trains on. ``excluded`` holds images that were
    found on disk but sit outside a dataset that already has official splits.
    """
    root = base or project_root()
    seed = int(config.get("seed", 42))
    records: list[ImageRecord] = []
    excluded: list[ImageRecord] = []
    for spec in dataset_specs(config):
        dataset_root = _resolve_root(root, spec["root"])
        found = discover_images(dataset_root, spec)
        # Distinct seed per dataset so adding a dataset cannot reshuffle another.
        dataset_seed = seed + int(spec["class_id"]) * 1009
        kept, dropped = finalize_splits(found, spec, dataset_seed)
        records.extend(kept)
        excluded.extend(dropped)
    return records, excluded


def label_files(root: Path) -> list[Path]:
    """YOLO label files that live under a ``labels`` directory."""
    found: list[Path] = []
    if not root.is_dir():
        return found
    for path in iter_files(root):
        if path.suffix.lower() != ".txt":
            continue
        if any(part.lower() == "labels" for part in path.parts):
            found.append(path)
    return found


def format_tree(root: Path, max_depth: int = 5) -> str:
    """Directory tree with file counts instead of every image name."""
    if not root.exists():
        return f"{root}  (missing)"

    lines = [f"{root.name}/"]

    def file_summary(files: list[Path]) -> str:
        counts: dict[str, int] = {}
        for path in files:
            suffix = path.suffix.lower() or "(no extension)"
            counts[suffix] = counts.get(suffix, 0) + 1
        parts = [f"{count} {suffix}" for suffix, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
        return ", ".join(parts)

    def visible(path: Path) -> list[Path]:
        entries = []
        for child in path.iterdir():
            if child.name.startswith(".") or child.name in SKIP_DIR_NAMES:
                continue
            entries.append(child)
        entries.sort(key=lambda child: (not child.is_dir(), child.name.lower()))
        return entries

    def walk(path: Path, prefix: str, depth: int) -> None:
        entries = visible(path)
        dirs = [entry for entry in entries if entry.is_dir()]
        files = [entry for entry in entries if entry.is_file()]
        nodes: list[tuple[str, object]] = []
        if files:
            nodes.append(("__files__", files))
        for directory in dirs:
            nodes.append((directory.name, directory))
        for index, (name, obj) in enumerate(nodes):
            last = index == len(nodes) - 1
            branch = "└── " if last else "├── "
            if name == "__files__":
                lines.append(prefix + branch + file_summary(obj))  # type: ignore[arg-type]
                continue
            lines.append(prefix + branch + f"{name}/")
            extension = prefix + ("    " if last else "│   ")
            if depth + 1 >= max_depth:
                lines.append(extension + "└── …")
            else:
                walk(obj, extension, depth + 1)  # type: ignore[arg-type]

    walk(root, "", 0)
    return "\n".join(lines)
