"""Train the terrain gate. Early-stops on validation macro-F1.

Usage:
    python train_gate.py --config config.yaml
    python train_gate.py --backbone resnet18 --epochs 15
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from moe.data import CLASS_NAMES, project_root
from moe.gate_data import (
    format_counts,
    make_datasets,
    make_loader,
    make_weighted_sampler,
    split_records,
)
from moe.gate_model import BACKBONES, build_gate
from moe.utils import load_config, pick_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MoE terrain gate")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--backbone", choices=BACKBONES, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None, help="mps, cuda, or cpu")
    parser.add_argument("--output", default="weights/gate_best.pt")
    return parser.parse_args()


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.backbone:
        config["backbone"] = args.backbone
    train_cfg = config["train"]
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        train_cfg["lr"] = args.lr
    return config


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float, float]:
    model.eval()
    losses = []
    counts = []
    predictions = []
    targets = []
    for batch in loader:
        images, labels = batch[0], batch[1]
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        losses.append(loss.item() * labels.size(0))
        counts.append(labels.size(0))
        predictions.append(logits.argmax(dim=1).cpu())
        targets.append(labels.cpu())
    if not counts:
        return 0.0, 0.0, 0.0
    y_true = torch.cat(targets).numpy()
    y_pred = torch.cat(predictions).numpy()
    average_loss = sum(losses) / sum(counts)
    accuracy = float((y_true == y_pred).mean())
    macro_f1 = float(
        f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)
    )
    return average_loss, accuracy, macro_f1


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total = 0.0
    seen = 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * labels.size(0)
        seen += labels.size(0)
    return total / max(seen, 1)


def save_checkpoint(path: Path, model, config, epoch: int, macro_f1: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "backbone": config["backbone"],
            "input_size": int(config["input_size"]),
            "class_names": list(config["class_names"]),
            "epoch": epoch,
            "val_macro_f1": macro_f1,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = pick_device(args.device)
    train_cfg = config["train"]

    grouped = split_records(config)
    print(f"Device: {device}")
    print(f"Backbone: {config['backbone']}")
    print("Images per split (gate label = source dataset):")
    print(format_counts(grouped["train"] + grouped["val"] + grouped["test"]))
    if not grouped["train"] or not grouped["val"]:
        raise RuntimeError("Need both a train split and a val split to train the gate.")

    datasets = make_datasets(config, return_paths=False)
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = make_weighted_sampler(grouped["train"], generator=generator)
    pin = device.type == "cuda"
    workers = int(train_cfg["num_workers"])
    train_loader = make_loader(
        datasets["train"],
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=workers,
        sampler=sampler,
        pin_memory=pin,
    )
    val_loader = make_loader(
        datasets["val"],
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
    )

    model = build_gate(
        config["backbone"],
        num_classes=len(config["class_names"]),
        pretrained=bool(train_cfg.get("pretrained", True)),
    ).to(device)
    # The sampler already balances classes, so the loss itself is unweighted.
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    epochs = int(train_cfg["epochs"])
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(train_cfg.get("min_lr", 0.0)),
    )

    output = project_root() / args.output
    history_path = project_root() / "runs" / "gate_train" / "history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    patience = int(train_cfg["patience"])
    stale = 0
    best_epoch = 0

    with history_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "lr", "train_loss", "val_loss", "val_acc", "val_macro_f1"])
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            writer.writerow(
                [epoch, f"{current_lr:.8f}", f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_acc:.6f}", f"{val_f1:.6f}"]
            )
            handle.flush()
            print(
                f"epoch {epoch:02d}/{epochs}  lr {current_lr:.6f}  "
                f"train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
                f"val_acc {val_acc:.4f}  val_macro_f1 {val_f1:.4f}"
            )
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch
                stale = 0
                save_checkpoint(output, model, config, epoch, val_f1)
                print(f"  saved {output}  (macro-F1 {val_f1:.4f})")
            else:
                stale += 1
                if stale >= patience:
                    print(f"Early stop at epoch {epoch}. Best macro-F1 {best_f1:.4f} at epoch {best_epoch}.")
                    break

    print(f"Best checkpoint: {output}")
    print("Class order:", ", ".join(f"{index} {name}" for index, name in enumerate(CLASS_NAMES)))


if __name__ == "__main__":
    main()
