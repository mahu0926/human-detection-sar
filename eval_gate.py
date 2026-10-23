"""Evaluate a trained gate on the test split.

Writes accuracy, per-class precision/recall/F1, a confusion-matrix PNG, and a
grid of misclassified images.

SeaDronesSee ships no public test split and is not randomly re-split, so class 0
is absent from test unless a video-safe holdout is added later. Pass ``--split val``
to score the validation set, which does include maritime.

Usage:
    python eval_gate.py
    python eval_gate.py --split val --checkpoint weights/gate_best.pt
"""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from moe.data import CLASS_NAMES, project_root
from moe.gate_data import make_datasets, make_loader
from moe.gate_model import load_gate
from moe.utils import load_config, pick_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the terrain gate")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-errors", type=int, default=16)
    return parser.parse_args()


@torch.no_grad()
def collect(model, loader, device):
    paths = []
    probabilities = []
    targets = []
    for images, labels, batch_paths in loader:
        images = images.to(device)
        probs = torch.softmax(model(images), dim=1).cpu()
        probabilities.append(probs)
        targets.append(labels)
        paths.extend(batch_paths)
    if not targets:
        return [], np.zeros((0, len(CLASS_NAMES))), np.zeros((0,), dtype=int)
    return paths, torch.cat(probabilities).numpy(), torch.cat(targets).numpy()


def select_errors(rows: list[dict], limit: int, seed: int) -> list[dict]:
    """Spread the grid across true classes instead of taking the first N."""
    by_class: dict[int, list[dict]] = defaultdict(list)
    rng = random.Random(seed)
    for row in rows:
        by_class[row["true"]].append(row)
    for group in by_class.values():
        rng.shuffle(group)
    picked: list[dict] = []
    while len(picked) < limit and any(by_class.values()):
        for class_id in sorted(by_class):
            if by_class[class_id]:
                picked.append(by_class[class_id].pop())
            if len(picked) >= limit:
                break
    return picked


def save_confusion(matrix: np.ndarray, names: list[str], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.2, 5.2))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis, fraction=0.046)
    axis.set_xticks(range(len(names)), names)
    axis.set_yticks(range(len(names)), names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Gate confusion matrix")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix[row, col])
            color = "white" if value > matrix.max() / 2 else "black"
            axis.text(col, row, str(value), ha="center", va="center", color=color)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_error_grid(rows: list[dict], names: list[str], path: Path) -> None:
    if not rows:
        return
    columns = 4
    n_rows = math.ceil(len(rows) / columns)
    fig, axes = plt.subplots(n_rows, columns, figsize=(columns * 3.3, n_rows * 3.5))
    flat = np.atleast_1d(axes).ravel()
    for axis, row in zip(flat, rows):
        with Image.open(row["path"]) as image:
            preview = image.convert("RGB")
        preview.thumbnail((320, 320))
        axis.imshow(preview)
        axis.set_title(
            f"true {names[row['true']]}\npred {names[row['pred']]} ({row['confidence']:.2f})",
            fontsize=9,
        )
        axis.axis("off")
    for axis in flat[len(rows) :]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_metrics(path: Path, split: str, accuracy: float, names: list[str], precision, recall, f1, support, present: list[str]) -> None:
    lines = [
        f"# Gate evaluation ({split})",
        "",
        f"Accuracy: {accuracy:.4f}",
        "",
        "Classes present in this split: " + (", ".join(present) if present else "(none)"),
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for index, name in enumerate(names):
        lines.append(
            f"| {name} | {precision[index]:.4f} | {recall[index]:.4f} | {f1[index]:.4f} | {int(support[index])} |"
        )
    macro_p = float(np.mean(precision))
    macro_r = float(np.mean(recall))
    macro_f = float(np.mean(f1))
    lines.append(f"| macro | {macro_p:.4f} | {macro_r:.4f} | {macro_f:.4f} | {int(support.sum())} |")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = pick_device(args.device)
    checkpoint = Path(args.checkpoint or config["gate"]["weights"])
    if not checkpoint.is_absolute():
        checkpoint = project_root() / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Gate checkpoint not found at {checkpoint}. Train it with `python train_gate.py` first."
        )

    output = Path(args.output or config["output"]["eval_dir"])
    if not output.is_absolute():
        output = project_root() / output
    output.mkdir(parents=True, exist_ok=True)

    datasets = make_datasets(config, return_paths=True)
    dataset = datasets[args.split]
    names = list(config["class_names"])
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Split: {args.split}  ({len(dataset)} images)")
    if len(dataset) == 0:
        raise RuntimeError(
            f"The {args.split} split is empty."
        )

    present_ids = sorted({record.class_id for record in dataset.records})
    present = [names[index] for index in present_ids]
    missing = [name for name in names if name not in present]
    if missing:
        print(
            "Note: this split has no images for "
            + ", ".join(missing)
            + ". Those rows stay in the table with support 0."
        )

    model, ckpt = load_gate(str(checkpoint), device)
    saved_names = list(ckpt.get("class_names", names))
    if saved_names != names:
        raise ValueError(f"Checkpoint classes {saved_names} do not match config {names}")

    workers = int(config["train"]["num_workers"])
    loader = make_loader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    paths, probs, targets = collect(model, loader, device)
    predictions = probs.argmax(axis=1)
    accuracy = float(accuracy_score(targets, predictions))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=list(range(len(names))),
        zero_division=0,
    )
    matrix = confusion_matrix(targets, predictions, labels=list(range(len(names))))

    print(f"Accuracy: {accuracy:.4f}")
    print(f"{'class':<14}{'precision':>12}{'recall':>10}{'f1':>10}{'support':>10}")
    for index, name in enumerate(names):
        print(
            f"{name:<14}{precision[index]:12.4f}{recall[index]:10.4f}{f1[index]:10.4f}{int(support[index]):10d}"
        )

    confusion_path = output / "confusion_matrix.png"
    save_confusion(matrix, names, confusion_path)
    errors = []
    for path, true, pred, prob in zip(paths, targets, predictions, probs):
        if int(true) != int(pred):
            errors.append(
                {
                    "path": path,
                    "true": int(true),
                    "pred": int(pred),
                    "confidence": float(prob[pred]),
                }
            )
    grid_path = output / "misclassified.png"
    chosen = select_errors(errors, args.max_errors, seed=int(config.get("seed", 42)))
    save_error_grid(chosen, names, grid_path)
    metrics_path = output / "metrics.md"
    write_metrics(metrics_path, args.split, accuracy, names, precision, recall, f1, support, present)
    print(f"Confusion matrix: {confusion_path}")
    print(f"Misclassified grid ({len(chosen)} of {len(errors)} errors): {grid_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
