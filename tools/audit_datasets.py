"""Audit the three source datasets before any gate training.

Discovers each dataset's real folder layout, opens every image with PIL, and
checks that a YOLO label file exists for the experts. Writes ``audit_report.md``.

Usage:
    python audit_datasets.py
    python tools/audit_datasets.py --workers 8
"""

from __future__ import annotations

import argparse
import statistics
import sys
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moe.data import (
    CLASS_NAMES,
    IMAGE_EXTENSIONS,
    ImageRecord,
    build_index,
    dataset_specs,
    format_tree,
    iter_files,
    label_files,
    project_root,
)
from moe.utils import load_config  # noqa: E402

# Local drone imagery is trusted. A few SeaDronesSee frames are 4K.
Image.MAX_IMAGE_PIXELS = 500_000_000

PREFERRED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
GRAYSCALE_MODES = {"L", "LA", "1"}
ALPHA_MODES = {"RGBA", "LA", "PA", "RGBa"}
RGB_MODES = {"RGB"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit MoE gate datasets")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="docs/audit_report.md")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def inspect_image(path_str: str) -> dict:
    """Open and fully decode one image. Runs in a worker process."""
    path = Path(path_str)
    record = {
        "path": path_str,
        "suffix": path.suffix.lower(),
        "ok": False,
        "error": "",
        "width": 0,
        "height": 0,
        "mode": "",
        "format": "",
    }
    try:
        with Image.open(path) as image:
            image.load()
            record["width"], record["height"] = image.size
            record["mode"] = image.mode or ""
            record["format"] = image.format or ""
            record["ok"] = True
    except Exception as exc:  # noqa: BLE001 - the audit must record every failure mode
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def inspect_label(path: Path) -> dict:
    empty = True
    boxes = 0
    bad = 0
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"empty": False, "boxes": 0, "bad": 1, "error": str(exc)}
    for line in text.splitlines():
        if not line.strip():
            continue
        empty = False
        parts = line.split()
        if len(parts) != 5:
            bad += 1
            continue
        try:
            float(parts[0])
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            bad += 1
            continue
        if any(value < 0 or value > 1 for value in coords):
            bad += 1
            continue
        boxes += 1
    return {"empty": empty, "boxes": boxes, "bad": bad, "error": ""}


def other_files(root: Path) -> Counter:
    counts: Counter = Counter()
    if not root.is_dir():
        return counts
    for path in iter_files(root):
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            continue
        if suffix == ".txt" and any(part.lower() == "labels" for part in path.parts):
            continue
        counts[suffix or "(no extension)"] += 1
    return counts


def _pct(part: int, whole: int) -> str:
    if whole <= 0:
        return "0%"
    return f"{100.0 * part / whole:.1f}%"


def _aspect_bucket(ratio: float) -> str:
    if ratio < 0.9:
        return "portrait"
    if ratio < 1.15:
        return "square"
    if ratio < 1.5:
        return "4:3-ish"
    if ratio < 2.0:
        return "16:9-ish"
    return "wider"


def aggregate(records: list[ImageRecord], inspections: dict[str, dict], label_stats: dict[str, dict]) -> dict:
    by_key: dict[tuple[str, str], list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_key[(record.dataset, record.split)].append(record)

    groups = []
    for (dataset, split), group in sorted(by_key.items(), key=lambda item: (item[0][0], item[0][1])):
        readable = []
        corrupt = []
        grayscale = 0
        alpha = 0
        other_mode = Counter()
        non_preferred = Counter()
        missing_label = []
        empty_label = 0
        bad_lines = 0
        boxes = 0
        modes = Counter()
        sizes = Counter()
        aspects = []
        extensions = Counter()
        for record in group:
            extensions[record.path.suffix.lower()] += 1
            info = inspections[str(record.path)]
            if not info["ok"]:
                corrupt.append(info)
                continue
            readable.append(info)
            mode = info["mode"]
            modes[mode] += 1
            if mode in GRAYSCALE_MODES:
                grayscale += 1
            if mode in ALPHA_MODES:
                alpha += 1
            if mode not in RGB_MODES and mode not in GRAYSCALE_MODES and mode not in ALPHA_MODES:
                other_mode[mode] += 1
            if info["suffix"] not in PREFERRED_EXTENSIONS:
                non_preferred[info["suffix"] or "(none)"] += 1
            sizes[(info["width"], info["height"])] += 1
            if info["height"]:
                aspects.append(info["width"] / info["height"])
            if record.label_path is None:
                missing_label.append(record)
            else:
                stats = label_stats[str(record.label_path.resolve())]
                boxes += stats["boxes"]
                bad_lines += stats["bad"]
                if stats["empty"]:
                    empty_label += 1
        min_size = min(sizes, key=lambda wh: wh[0] * wh[1]) if sizes else (0, 0)
        max_size = max(sizes, key=lambda wh: wh[0] * wh[1]) if sizes else (0, 0)
        common = sizes.most_common(8)
        groups.append(
            {
                "dataset": dataset,
                "class_name": group[0].class_name,
                "class_id": group[0].class_id,
                "split": split,
                "split_source": group[0].split_source,
                "images": len(group),
                "readable": len(readable),
                "corrupt": corrupt,
                "grayscale": grayscale,
                "alpha": alpha,
                "other_mode": other_mode,
                "non_preferred": non_preferred,
                "missing_label": missing_label,
                "empty_label": empty_label,
                "bad_lines": bad_lines,
                "boxes": boxes,
                "modes": modes,
                "sizes": sizes,
                "common_sizes": common,
                "min_size": min_size,
                "max_size": max_size,
                "aspects": aspects,
                "extensions": extensions,
            }
        )
    return {"groups": groups}


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _size(pair: tuple[int, int]) -> str:
    return f"{pair[0]}×{pair[1]}"


def render_report(
    config: dict,
    records: list[ImageRecord],
    stats: dict,
    trees: dict[str, str],
    sidecars: dict[str, Counter],
    orphans: dict[str, list[Path]],
    roots: dict[str, Path],
    excluded: list[ImageRecord],
    inspections: dict[str, dict],
) -> str:
    generated = datetime.now().isoformat(timespec="seconds")
    groups = stats["groups"]
    lines = [
        "# Dataset audit",
        "",
        f"Generated: {generated}",
        "",
        "The gate's label is the source dataset, not a bounding-box class.",
        "Original images were only read. Nothing on disk was resized or copied.",
        "",
        "## Summary",
        "",
    ]

    summary_rows = []
    for group in groups:
        non_pref = sum(group["non_preferred"].values())
        other_modes = sum(group["other_mode"].values())
        summary_rows.append(
            [
                group["dataset"],
                f"{group['class_id']} {group['class_name']}",
                group["split"],
                str(group["images"]),
                str(group["readable"]),
                str(len(group["corrupt"])),
                str(group["grayscale"]),
                str(group["alpha"]),
                str(other_modes),
                str(non_pref),
                str(len(group["missing_label"])),
                str(group["empty_label"]),
                str(group["bad_lines"]),
            ]
        )
    lines.append(
        _md_table(
            [
                "Dataset",
                "Class",
                "Split",
                "Images",
                "Readable",
                "Corrupt",
                "Grayscale",
                "Alpha",
                "Other mode",
                "Non-JPG/PNG",
                "Missing label",
                "Empty label",
                "Bad label lines",
            ],
            summary_rows,
        )
    )
    lines.append("")

    geo_rows = []
    for group in groups:
        aspects = group["aspects"]
        if aspects:
            aspect_txt = (
                f"{min(aspects):.3f} / {statistics.median(aspects):.3f} / {max(aspects):.3f}"
            )
        else:
            aspect_txt = "—"
        common = group["common_sizes"]
        common_txt = ", ".join(f"{_size(wh)} ({count})" for wh, count in common[:3]) or "—"
        geo_rows.append(
            [
                group["dataset"],
                group["split"],
                _size(group["min_size"]),
                _size(group["max_size"]),
                common_txt,
                aspect_txt,
            ]
        )
    lines.extend(
        [
            "Resolution is the decoded PIL size. Min and max are by pixel area.",
            "Aspect is width/height, reported as min / median / max.",
            "",
            _md_table(
                ["Dataset", "Split", "Min W×H", "Max W×H", "Most common W×H", "Aspect min / median / max"],
                geo_rows,
            ),
            "",
            "## Gate label counts",
            "",
            "These counts are what the gate trains on. They are not object-instance counts.",
            "",
        ]
    )

    balance_rows = []
    totals = Counter()
    for name in CLASS_NAMES:
        row = [name]
        class_total = 0
        for split in ("train", "val", "test"):
            count = sum(group["images"] for group in groups if group["class_name"] == name and group["split"] == split)
            class_total += count
            row.append(str(count))
        row.append(str(class_total))
        totals[name] = class_total
        balance_rows.append(row)
    grand = sum(totals.values()) or 1
    for row, name in zip(balance_rows, CLASS_NAMES):
        row.append(_pct(totals[name], grand))
    lines.append(_md_table(["Class", "Train", "Val", "Test", "Total", "Share"], balance_rows))
    lines.append("")

    largest = max(totals.values()) if totals else 0
    smallest = min(totals.values()) if totals else 1
    lines.extend(
        [
            "## Notes for training",
            "",
            f"- Class imbalance is large: the biggest class has {largest} images and the smallest has {smallest} "
            f"({largest / max(smallest, 1):.1f}×). `train_gate.py` uses a `WeightedRandomSampler` with inverse-frequency weights.",
            "- Resolutions are not shared across datasets (see the geometry table). A gate that sees the raw "
            "file can classify by image size or camera color instead of terrain. Training therefore random-crops, "
            "flips, color-jitters, and slightly blurs. Validation and routing letterbox to the configured input size "
            "without rewriting the file. Experts still open the original full-resolution image.",
            "- Official split folders are kept. A seeded 70/20/10 split is applied only when a dataset has no split "
            "folders. SeaDronesSee is `preserve_splits: true` because neighboring frames come from the same video.",
        ]
    )
    sds_test = sum(group["images"] for group in groups if group["dataset"] == "seadronessee" and group["split"] == "test")
    if sds_test == 0:
        lines.append(
            "- SeaDronesSee has no test split in this mirror (the official test set has no public labels). "
            "It was not invented by reshuffling train/val. Test-set gate metrics will not include maritime "
            "until a video-safe holdout exists. Score maritime with `python eval_gate.py --split val`."
        )
    corrupt_total = sum(len(group["corrupt"]) for group in groups)
    missing_total = sum(len(group["missing_label"]) for group in groups)
    if corrupt_total == 0 and missing_total == 0:
        lines.append("- Every indexed image decoded with PIL and has a matching label file.")
    lines.append("")

    for spec in dataset_specs(config):
        name = spec["name"]
        root = roots[name]
        lines.extend(
            [
                f"## {name} — class {spec['class_id']} {spec['class_name']}",
                "",
                f"Root: `{root}`",
                "",
                "Split policy: "
                + (
                    "official folders only (`preserve_splits`). Random assignment is refused."
                    if spec.get("preserve_splits")
                    else "official folders when present; otherwise a seeded 70/20/10 split."
                ),
                "",
                "### Folder structure",
                "",
                "```text",
                trees[name],
                "```",
                "",
            ]
        )
        sidecar = sidecars.get(name, Counter())
        if sidecar:
            bits = ", ".join(f"{count} `{suffix}`" for suffix, count in sidecar.most_common())
            lines.append(f"Non-image files skipped by the gate index: {bits}.")
            lines.append("")
        dataset_groups = [group for group in groups if group["dataset"] == name]
        if not dataset_groups:
            lines.append("No images found.")
            lines.append("")
            continue
        for group in dataset_groups:
            aspects = group["aspects"]
            buckets = Counter(_aspect_bucket(ratio) for ratio in aspects)
            bucket_txt = ", ".join(f"{key} {value}" for key, value in buckets.most_common()) or "—"
            ext_txt = ", ".join(f"{count} {suffix}" for suffix, count in group["extensions"].most_common())
            mode_txt = ", ".join(f"{count} {mode}" for mode, count in group["modes"].most_common()) or "—"
            lines.extend(
                [
                    f"### {group['split']} ({group['split_source']})",
                    "",
                    f"- Images: {group['images']} ({ext_txt})",
                    f"- Readable: {group['readable']}. Corrupt: {len(group['corrupt'])}.",
                    f"- Modes: {mode_txt}. Grayscale: {group['grayscale']}. Alpha: {group['alpha']}.",
                    f"- Labels: {group['images'] - len(group['missing_label'])} matched, "
                    f"{len(group['missing_label'])} missing, {group['empty_label']} empty, "
                    f"{group['boxes']} boxes, {group['bad_lines']} malformed lines.",
                    "- Most common sizes: "
                    + (
                        ", ".join(f"{_size(wh)} × {count}" for wh, count in group["common_sizes"][:5])
                        or "—"
                    ),
                    f"- Aspect buckets: {bucket_txt}.",
                    "",
                ]
            )
            if group["non_preferred"]:
                lines.append(
                    "- Non-JPG/PNG: "
                    + ", ".join(f"{count} {suffix}" for suffix, count in group["non_preferred"].most_common())
                )
                lines.append("")
            if group["other_mode"]:
                lines.append(
                    "- Other non-RGB modes: "
                    + ", ".join(f"{count} {mode}" for mode, count in group["other_mode"].most_common())
                )
                lines.append("")
            if group["corrupt"]:
                lines.append("Corrupt or unreadable files:")
                lines.append("")
                for info in group["corrupt"][:40]:
                    lines.append(f"- `{info['path']}` — {info['error']}")
                extra = len(group["corrupt"]) - 40
                if extra > 0:
                    lines.append(f"- … and {extra} more")
                lines.append("")
            if group["missing_label"]:
                lines.append("Images with no label file (first 40):")
                lines.append("")
                for record in group["missing_label"][:40]:
                    lines.append(f"- `{record.path}`")
                extra = len(group["missing_label"]) - 40
                if extra > 0:
                    lines.append(f"- … and {extra} more")
                lines.append("")
        orphan_list = orphans.get(name, [])
        lines.append(f"Orphan label files (no image): {len(orphan_list)}.")
        if orphan_list:
            lines.append("")
            for path in orphan_list[:40]:
                lines.append(f"- `{path}`")
            if len(orphan_list) > 40:
                lines.append(f"- … and {len(orphan_list) - 40} more")
        lines.append("")
        dropped = [record for record in excluded if record.dataset == name]
        if dropped:
            lines.append(
                f"Excluded {len(dropped)} image(s) that sit outside the official split folders. "
                "They are not given a random split."
            )
            lines.append("")
            for record in dropped:
                label = "label present" if record.label_path else "no label"
                info = inspections.get(str(record.path))
                if info and info["ok"]:
                    detail = f"{info['width']}×{info['height']} {info['mode']} {info['format']}"
                elif info:
                    detail = info["error"]
                else:
                    detail = "not opened"
                lines.append(f"- `{record.path}` ({label}; {detail})")
            lines.append("")

    lines.extend(
        [
            "## What was not done",
            "",
            "No gate training was started. Review this report, then run `python train_gate.py`.",
            "",
        ]
    )
    return "\n".join(lines)


def print_summary(stats: dict) -> None:
    print()
    print(
        f"{'dataset':<16}{'class':<12}{'split':<8}{'images':>8}{'corrupt':>9}"
        f"{'gray':>6}{'alpha':>7}{'badext':>8}{'nolabel':>9}{'empty':>8}{'boxes':>8}"
    )
    for group in stats["groups"]:
        print(
            f"{group['dataset']:<16}{group['class_name']:<12}{group['split']:<8}"
            f"{group['images']:8d}{len(group['corrupt']):9d}{group['grayscale']:6d}"
            f"{group['alpha']:7d}{sum(group['non_preferred'].values()):8d}"
            f"{len(group['missing_label']):9d}{group['empty_label']:8d}{group['boxes']:8d}"
        )
    print()
    for group in stats["groups"]:
        common = ", ".join(f"{_size(wh)} ({count})" for wh, count in group["common_sizes"][:3])
        aspects = group["aspects"]
        if aspects:
            aspect_txt = f"{min(aspects):.3f}/{statistics.median(aspects):.3f}/{max(aspects):.3f}"
        else:
            aspect_txt = "-"
        print(
            f"{group['dataset']:<16}{group['split']:<8}"
            f"  min {_size(group['min_size']):<12} max {_size(group['max_size']):<12}"
            f"  common {common}  aspect {aspect_txt}"
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    base = project_root()
    print("Indexing datasets (layout is discovered, not assumed)...")
    records, excluded = build_index(config, base)
    print(f"Indexed {len(records)} images, excluded {len(excluded)} outside official splits")

    roots = {}
    trees = {}
    sidecars = {}
    orphans = {}
    label_stats = {}
    for spec in dataset_specs(config):
        root = Path(spec["root"])
        if not root.is_absolute():
            root = base / root
        roots[spec["name"]] = root
        trees[spec["name"]] = format_tree(root)
        sidecars[spec["name"]] = other_files(root)
        matched = {
            record.label_path.resolve()
            for record in records
            if record.dataset == spec["name"] and record.label_path is not None
        }
        orphan_list = []
        for path in label_files(root):
            stats = inspect_label(path)
            label_stats[str(path.resolve())] = stats
            if path.resolve() not in matched:
                orphan_list.append(path)
        orphans[spec["name"]] = orphan_list

    # Labels that were found via a sibling .txt (no labels/ folder) still need stats.
    for record in records:
        if record.label_path is None:
            continue
        key = str(record.label_path.resolve())
        if key not in label_stats:
            label_stats[key] = inspect_label(record.label_path)

    paths = [str(record.path) for record in records]
    paths.extend(str(record.path) for record in excluded)
    inspections = {}
    workers = max(1, args.workers)
    print(f"Opening {len(paths)} images with PIL ({workers} workers)...")
    if workers == 1:
        iterator = map(inspect_image, paths)
        for info in tqdm(iterator, total=len(paths)):
            inspections[info["path"]] = info
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for info in tqdm(pool.map(inspect_image, paths, chunksize=16), total=len(paths)):
                inspections[info["path"]] = info

    stats = aggregate(records, inspections, label_stats)
    report = render_report(
        config, records, stats, trees, sidecars, orphans, roots, excluded, inspections
    )
    output = Path(args.output)
    if not output.is_absolute():
        output = base / output
    output.write_text(report)
    print_summary(stats)
    corrupt = sum(len(group["corrupt"]) for group in stats["groups"])
    missing = sum(len(group["missing_label"]) for group in stats["groups"])
    print(f"Corrupt images: {corrupt}")
    print(f"Missing labels: {missing}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
