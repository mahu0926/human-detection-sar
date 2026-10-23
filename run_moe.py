"""Route images through the MoE and save annotated results.

Usage:
    python run_moe.py --source path/to/image_or_folder --save
    python run_moe.py --source path/to/folder --mode top2 --tta --save
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from moe.data import IMAGE_EXTENSIONS, project_root
from moe.experts import MoEPipeline, draw_result
from moe.utils import load_config, pick_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MoE human-detection routing")
    parser.add_argument("--source", required=True, help="Image file or folder")
    parser.add_argument("--mode", choices=("top1", "top2"), default="top1")
    parser.add_argument("--save", action="store_true", help="Write annotated images and a CSV log")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tta", action="store_true", help="Average gate probs over 5 crops")
    return parser.parse_args()


def list_sources(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Not an image file: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Source not found: {source}")
    files = []
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if any(part.startswith(".") or part in {".cache", "__pycache__"} for part in path.parts):
            continue
        files.append(path)
    files.sort()
    if not files:
        raise FileNotFoundError(f"No images under {source}")
    return files


def output_path_for(source_root: Path, image_path: Path, output_dir: Path) -> Path:
    if source_root.is_file():
        relative = Path(image_path.name)
    else:
        try:
            relative = image_path.relative_to(source_root)
        except ValueError:
            relative = Path(image_path.name)
    destination = output_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination.with_suffix(".jpg")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = pick_device(args.device)
    source = Path(args.source)
    if not source.is_absolute():
        source = Path.cwd() / source
    images = list_sources(source)
    output_dir = Path(args.output or config["output"]["moe_dir"])
    if not output_dir.is_absolute():
        output_dir = project_root() / output_dir

    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Images: {len(images)}")
    pipeline = MoEPipeline(config, device, tta=args.tta)
    names = pipeline.names

    rows = []
    for image_path in images:
        result = pipeline.run_image(image_path, args.mode)
        probs = result["gate_probs"]
        summary = (
            f"{image_path.name}  expert={result['expert']}  "
            f"probs=[{', '.join(f'{name}:{prob:.3f}' for name, prob in zip(names, probs))}]  "
            f"detections={result['num_detections']}"
        )
        print(summary)
        rows.append(result)
        if args.save:
            annotated = draw_result(result["image"], result, names)
            destination = output_path_for(source, image_path, output_dir)
            annotated.save(destination, quality=92)
            result["saved"] = str(destination)

    if args.save:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "routing_log.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "image",
                    "gate_maritime",
                    "gate_urban",
                    "gate_vegetation",
                    "mode",
                    "expert",
                    "num_detections",
                ]
            )
            for result in rows:
                probs = result["gate_probs"]
                writer.writerow(
                    [
                        result["path"],
                        f"{probs[0]:.6f}",
                        f"{probs[1]:.6f}",
                        f"{probs[2]:.6f}",
                        args.mode,
                        result["expert"],
                        result["num_detections"],
                    ]
                )
        print(f"Annotated images and log: {output_dir}")
        print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
