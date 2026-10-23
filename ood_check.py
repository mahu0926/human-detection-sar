"""Sanity check: route a folder of out-of-distribution aerial images through the gate.

If the gate scores ~100% on test but spreads randomly here, it is recognising
datasets, not terrain. Example: VisDrone images are mostly urban, so most
should route to `urban`.

Usage:
    python ood_check.py --source path/to/visdrone/images --expect urban --tta
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from moe.cache import prepare_gate_image
from moe.data import IMAGE_EXTENSIONS, project_root
from moe.gate_data import build_transforms, build_tta_transform
from moe.gate_model import load_gate
from moe.utils import load_config, pick_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--expect", default=None, help="class you expect most images to route to")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = pick_device(args.device)
    names = list(config["class_names"])
    ckpt = Path(config["gate"]["weights"])
    if not ckpt.is_absolute():
        ckpt = project_root() / ckpt
    model, meta = load_gate(str(ckpt), device)
    size = int(meta.get("input_size", config["input_size"]))
    transform = build_tta_transform(size) if args.tta else build_transforms(size, train=False)

    source = Path(args.source)
    files = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)[: args.limit]
    if not files:
        raise SystemExit(f"No images under {source}")

    counts: Counter = Counter()
    rows = []
    with torch.no_grad():
        for path in tqdm(files):
            with Image.open(path) as image:
                x = transform(prepare_gate_image(image))
            x = x.to(device) if args.tta else x.unsqueeze(0).to(device)
            probs = torch.softmax(model(x), dim=1).mean(0).cpu().tolist()
            pred = max(range(len(names)), key=lambda i: probs[i])
            counts[names[pred]] += 1
            rows.append([str(path), *[f"{p:.4f}" for p in probs], names[pred]])

    out = project_root() / "runs" / "ood_check.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", *[f"p_{n}" for n in names], "routed_to"])
        writer.writerows(rows)

    total = len(files)
    print(f"\nRouted {total} images from {source}")
    for name in names:
        print(f"  {name:<11} {counts[name]:>5}  ({100 * counts[name] / total:5.1f}%)")
    if args.expect:
        print(f"Agreement with expected '{args.expect}': {100 * counts[args.expect] / total:.1f}%")
    print(f"Per-image CSV: {out}")


if __name__ == "__main__":
    main()
