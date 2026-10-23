"""End-to-end MoE evaluation: metrics + a visual HTML report.

Every test image is run through the gate once and through all three experts
once. Raw predictions are cached, then these methods are scored from the cache
(no extra inference):

  Expert 1/2/3 only   every image goes to that one expert
  Baselines           optional extra YOLOv5 models (experts.baselines), e.g. the
                      VisDrone baseline and the multi-domain model
  MoE top-1           the gate picks one expert
  MoE top-2           the gate picks one, plus the runner-up when max p < threshold
  All 3 + NMS         every expert on every image, boxes merged (brute-force ensemble)
  Oracle routing      the expert that matches the image's true terrain
                      (upper bound for any gate)

Metrics follow YOLOv5 val.py: mAP@0.5, mAP@0.5:0.95 (101-point interpolation),
and precision/recall at the max-F1 confidence. Only the person class is scored
(SeaDronesSee: swimmer, class 0; boats/buoys etc. are ignored).

Usage:
    python eval_moe.py --limit 10          # quick smoke test, 10 images per terrain
    python eval_moe.py                     # full test split
    python eval_moe.py --from-cache        # rebuild the report without re-running models
    open runs/moe_eval/report.html
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import pickle
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torchvision.ops import nms
from tqdm import tqdm

from moe.cache import prepare_gate_image
from moe.data import IMAGE_EXTENSIONS, build_index, find_label, project_root
from moe.experts import ExpertBank, load_font
from moe.gate_data import build_transforms, build_tta_transform
from moe.gate_model import load_gate
from moe.utils import load_config, pick_device, set_seed

IOUV = np.linspace(0.5, 0.95, 10)
_trapz = getattr(np, "trapezoid", None) or np.trapz

METHOD_COLORS = {
    "maritime": "#2563eb",
    "urban": "#ea580c",
    "vegetation": "#16a34a",
    "Baseline": "#6b7280",
    "MoE top-1": "#a78bfa",
    "MoE top-2": "#7c3aed",
    "All 3 experts + NMS": "#0ea5e9",
    "Oracle routing": "#111827",
}
TP_COLOR, FP_COLOR, FN_COLOR = (34, 197, 94), (239, 68, 68), (250, 204, 21)


# --------------------------------------------------------------------- args

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the full MoE (gate + experts)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", default="test", choices=("test", "val"))
    parser.add_argument("--device", default=None, help="mps, cuda or cpu")
    parser.add_argument("--imgsz", type=int, default=None, help="expert inference size (default: config)")
    parser.add_argument("--tta", action="store_true", help="5-crop gate averaging")
    parser.add_argument("--threshold", type=float, default=None, help="top-2 threshold (default: config)")
    parser.add_argument("--limit", type=int, default=0, help="images per terrain (0 = all)")
    parser.add_argument("--vis", type=int, default=12, help="annotated images per terrain in the report")
    parser.add_argument("--mode", choices=("top1", "top2"), default="top2", help="method shown in the gallery")
    parser.add_argument("--output", default="runs/moe_eval")
    parser.add_argument("--from-cache", action="store_true", help="reuse predictions.pkl")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


# ------------------------------------------------------------------ data

def collect_test_images(config: dict, split: str, limit: int, seed: int) -> dict[str, list[Path]]:
    names = list(config["class_names"])
    sets = (config.get("eval") or {}).get("test_sets")
    found: dict[str, list[Path]] = {}
    if sets:
        for terrain, folder in sets.items():
            if terrain not in names:
                raise ValueError(f"eval.test_sets key {terrain!r} is not one of {names}")
            root = Path(folder)
            root = root if root.is_absolute() else project_root() / root
            if not root.is_dir():
                raise FileNotFoundError(f"Test folder for {terrain} not found: {root}")
            found[terrain] = [
                path
                for path in root.rglob("*")
                if path.suffix.lower() in IMAGE_EXTENSIONS
                and not any(part.startswith(".") for part in path.relative_to(root).parts)
            ]
    else:
        records, _ = build_index(config)
        for record in records:
            if record.split == split:
                found.setdefault(record.class_name, []).append(record.path)
    rng = random.Random(seed)
    out = {}
    for terrain in names:
        files = sorted(found.get(terrain, []))
        if limit and len(files) > limit:
            files = sorted(rng.sample(files, limit))
        if files:
            out[terrain] = files
    return out


def load_gt(image_path: Path, width: int, height: int, person_class: int) -> tuple[np.ndarray, bool]:
    label = find_label(image_path)
    if label is None:
        return np.zeros((0, 4), np.float32), False
    rows = []
    for line in label.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5 or int(float(parts[0])) != person_class:
            continue
        x, y, w, h = (float(v) for v in parts[1:5])
        rows.append([(x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height])
    return np.asarray(rows, np.float32).reshape(-1, 4), True


# ------------------------------------------------------------- inference

def baseline_specs(config: dict) -> dict[str, str]:
    """experts.baselines {label: path}; experts.baseline (single path) still works."""
    experts = config["experts"]
    specs = dict(experts.get("baselines") or {})
    if experts.get("baseline"):
        specs.setdefault("Baseline", experts["baseline"])
    return specs


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def run_inference(config: dict, tests: dict[str, list[Path]], device: torch.device, imgsz: int, tta: bool) -> dict:
    names = list(config["class_names"])
    ev = config.get("eval") or {}
    person = int(ev.get("person_class", 0))

    gate_path = Path(config["gate"]["weights"])
    gate_path = gate_path if gate_path.is_absolute() else project_root() / gate_path
    gate, meta = load_gate(str(gate_path), device)
    size = int(meta.get("input_size", config["input_size"]))
    transform = build_tta_transform(size) if tta else build_transforms(size, train=False)

    bank = ExpertBank(config, device)
    models: dict[str, object] = {}
    weight_files: dict[str, str] = {}
    for name in names:
        model, _, _, weight = bank.get(name, allow_fallback=False)
        models[name] = model
        weight_files[name] = weight
    for label, weight in baseline_specs(config).items():
        path = Path(weight)
        path = path if path.is_absolute() else project_root() / path
        if not path.is_file():
            raise FileNotFoundError(f"Baseline {label!r} not found: {path}")
        models[f"base:{label}"] = bank.load_path(str(path))
        weight_files[label] = str(path)
    for model in models.values():
        model.conf = float(ev.get("conf", 0.001))
        model.iou = float(ev.get("iou", 0.6))
        model.max_det = int(ev.get("max_det", 300))

    timings: dict[str, list[float]] = {"gate": [], **{name: [] for name in models}}
    records = []
    total = sum(len(files) for files in tests.values())
    with torch.no_grad(), tqdm(total=total, desc="images") as bar:
        for terrain, files in tests.items():
            for path in files:
                with Image.open(path) as handle:
                    image = ImageOps.exif_transpose(handle).convert("RGB")
                width, height = image.size
                gt, has_label = load_gt(path, width, height, person)

                start = time.perf_counter()
                x = transform(prepare_gate_image(image))
                x = x.to(device) if tta else x.unsqueeze(0).to(device)
                probs = torch.softmax(gate(x), dim=1).mean(0).cpu().numpy()
                _sync(device)
                timings["gate"].append((time.perf_counter() - start) * 1000)

                preds = {}
                for name, model in models.items():
                    start = time.perf_counter()
                    result = model(image, size=imgsz)
                    array = result.xyxy[0].detach().cpu().numpy().astype(np.float32)
                    timings[name].append((time.perf_counter() - start) * 1000)
                    array = array[array[:, 5] == person] if len(array) else array.reshape(0, 6)
                    preds[name] = array[:, :5]
                records.append(
                    {
                        "path": str(path),
                        "terrain": terrain,
                        "size": (width, height),
                        "gt": gt,
                        "has_label": has_label,
                        "probs": probs,
                        "preds": preds,
                    }
                )
                bar.update(1)
    return {
        "records": records,
        "timings": timings,
        "names": names,
        "model_keys": list(models),
        "weights": weight_files,
        "imgsz": imgsz,
        "tta": tta,
        "device": str(device),
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


# ----------------------------------------------------------- routing

def route(probs: np.ndarray, mode: str, threshold: float) -> list[int]:
    order = np.argsort(-probs)
    chosen = [int(order[0])]
    if mode == "top2" and float(probs[order[0]]) < threshold:
        chosen.append(int(order[1]))
    return chosen


def combine(preds: list[np.ndarray], merge_iou: float, max_det: int) -> np.ndarray:
    if len(preds) == 1:
        return preds[0]
    merged = np.concatenate(preds, axis=0)
    if len(merged) == 0:
        return merged
    keep = nms(torch.from_numpy(merged[:, :4]), torch.from_numpy(merged[:, 4]), merge_iou).numpy()
    return merged[keep[:max_det]]


def build_methods(names: list[str], model_keys: list[str], threshold: float, merge_iou: float, max_det: int):
    methods = {}
    for index, name in enumerate(names):
        methods[f"Expert {index + 1} ({name}) only"] = (
            name,
            lambda rec, n=name, i=index: (rec["preds"][n], [i]),
        )
    for key in model_keys:
        if key.startswith("base:"):
            methods[key[5:]] = (key, lambda rec, k=key: (rec["preds"][k], []))
    for mode, label in (("top1", "MoE top-1"), ("top2", "MoE top-2")):
        def moe(rec, mode=mode):
            chosen = route(rec["probs"], mode, threshold)
            return combine([rec["preds"][names[c]] for c in chosen], merge_iou, max_det), chosen
        methods[label] = (label, moe)
    methods["All 3 experts + NMS"] = (
        "All 3 experts + NMS",
        lambda rec: (combine([rec["preds"][n] for n in names], merge_iou, max_det), list(range(len(names)))),
    )
    methods["Oracle routing"] = (
        "Oracle routing",
        lambda rec: (rec["preds"][rec["terrain"]], [names.index(rec["terrain"])]),
    )
    return methods


# ------------------------------------------------------------- metrics

def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area_a = np.clip(a[:, 2:] - a[:, :2], 0, None).prod(1)
    area_b = np.clip(b[:, 2:] - b[:, :2], 0, None).prod(1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match(gt: np.ndarray, pred: np.ndarray, thresholds=IOUV) -> tuple[np.ndarray, np.ndarray]:
    """YOLOv5 val.py matching. Returns (pred correct [n_pred, n_thr], gt matched at first threshold)."""
    correct = np.zeros((len(pred), len(thresholds)), bool)
    gt_hit = np.zeros(len(gt), bool)
    if len(gt) == 0 or len(pred) == 0:
        return correct, gt_hit
    iou = box_iou(gt, pred[:, :4])
    for k, thr in enumerate(thresholds):
        x = np.nonzero(iou >= thr)
        if not x[0].size:
            continue
        matches = np.stack([x[0], x[1], iou[x]], axis=1)
        if x[0].size > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int), k] = True
        if k == 0:
            gt_hit[matches[:, 0].astype(int)] = True
    return correct, gt_hit


def _smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    nf = round(len(y) * f * 2) // 2 + 1
    pad = np.ones(nf // 2)
    yp = np.concatenate((pad * y[0], y, pad * y[-1]))
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def _ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(_trapz(np.interp(x, mrec, mpre), x))


def detection_metrics(correct: np.ndarray, conf: np.ndarray, n_gt: int) -> dict:
    out = {"map50": 0.0, "map": 0.0, "p": 0.0, "r": 0.0, "f1": 0.0, "n_gt": int(n_gt), "n_pred": int(len(conf))}
    if n_gt == 0 or len(conf) == 0:
        return out
    order = np.argsort(-conf)
    tp, conf = correct[order].astype(float), conf[order]
    tpc, fpc = tp.cumsum(0), (1 - tp).cumsum(0)
    recall = tpc / (n_gt + 1e-16)
    precision = tpc / (tpc + fpc)
    ap = [_ap(recall[:, j], precision[:, j]) for j in range(tp.shape[1])]
    px = np.linspace(0, 1, 1000)
    r = np.interp(-px, -conf, recall[:, 0], left=0)
    p = np.interp(-px, -conf, precision[:, 0], left=1)
    f1 = 2 * p * r / (p + r + 1e-16)
    k = int(_smooth(f1, 0.1).argmax())
    out.update(map50=ap[0], map=float(np.mean(ap)), p=float(p[k]), r=float(r[k]), f1=float(f1[k]))
    return out


def score(records: list[dict], methods: dict, names: list[str]) -> tuple[dict, dict]:
    """Returns metrics[method][terrain|'All (pooled)'|'Macro avg'] and per-record method outputs."""
    outputs = {label: [] for label in methods}
    per = {label: {} for label in methods}
    for label, (_, fn) in methods.items():
        buckets: dict[str, list] = {}
        for rec in records:
            pred, chosen = fn(rec)
            correct, _ = match(rec["gt"], pred)
            outputs[label].append((pred, chosen))
            for key in (rec["terrain"], "All (pooled)"):
                buckets.setdefault(key, []).append((correct, pred[:, 4] if len(pred) else np.zeros(0), len(rec["gt"])))
        for key, items in buckets.items():
            correct = np.concatenate([c for c, _, _ in items]) if items else np.zeros((0, 10), bool)
            conf = np.concatenate([s for _, s, _ in items]) if items else np.zeros(0)
            per[label][key] = detection_metrics(correct, conf, sum(n for _, _, n in items))
        terrains = [t for t in names if t in per[label]]
        per[label]["Macro avg"] = {
            metric: float(np.mean([per[label][t][metric] for t in terrains])) for metric in ("map50", "map", "p", "r", "f1")
        }
    return per, outputs


# ---------------------------------------------------------- paper table

TEST_SET_LABELS = {
    "vegetation": "Heridal (vegetation)",
    "maritime": "SeaDronesSee (maritime)",
    "urban": "NTIC (urban)",
}
TABLE_ORDER = ("vegetation", "maritime", "urban")
MOE_LABEL = "MoE (ours)"


def table_rows(metrics: dict, model_keys: list[str]) -> list[str]:
    """Models compared in the paper table: baselines, all-3 ensemble, MoE."""
    rows = [key[5:] for key in model_keys if key.startswith("base:")]
    return rows + ["All 3 experts + NMS", "MoE top-1"]


def write_results_table(out: Path, metrics: dict, records: list[dict], model_keys: list[str]) -> str:
    """Results table in the team's format, as Markdown and LaTeX (booktabs)."""
    terrains = [t for t in TABLE_ORDER if any(r["terrain"] == t for r in records)]
    rows = table_rows(metrics, model_keys)
    show = {m: (MOE_LABEL if m == "MoE top-1" else m) for m in rows}
    cols = (("p", "Precision"), ("r", "Recall"), ("map50", "mAP@50"), ("map", "mAP@50:95"))

    def best(group: str, key: str) -> float:
        return max(metrics[m].get(group, {}).get(key, 0.0) for m in rows)

    def cell(m: str, group: str, key: str, bold) -> str:
        value = metrics[m].get(group, {}).get(key, 0.0)
        text = f"{value:.3f}"
        return bold(text) if abs(value - best(group, key)) < 5e-4 else text

    md = ["| Test set | Model | Precision | Recall | mAP@50 | mAP@50:95 |", "|---|---|---|---|---|---|"]
    tex = [
        r"\begin{tabular}{llcccc}", r"\toprule",
        r"Test set & Model & Precision & Recall & mAP@50 & mAP@50:95 \\", r"\midrule",
    ]
    for terrain in terrains:
        n_img = sum(r["terrain"] == terrain for r in records)
        n_gt = sum(len(r["gt"]) for r in records if r["terrain"] == terrain)
        label = TEST_SET_LABELS.get(terrain, terrain)
        sub = f"{n_img:,} img, {n_gt:,} persons"
        for i, m in enumerate(rows):
            first = f"{label}<br>{sub}" if i == 0 else ""
            md.append(f"| {first} | {show[m]} | " + " | ".join(cell(m, terrain, k, lambda x: f"**{x}**") for k, _ in cols) + " |")
            tex_first = label if i == 0 else (sub if i == 1 else "")
            tex.append(f"{tex_first} & {show[m]} & " + " & ".join(cell(m, terrain, k, lambda x: rf"\textbf{{{x}}}") for k, _ in cols) + r" \\")
        if len(rows) == 1:
            tex.append(f"{sub} & & & & & \\\\")
        tex.append(r"\midrule")
    for i, m in enumerate(rows):
        first = "Mean over domains" if i == 0 else ""
        md.append(f"| {first} | {show[m]} | | | " + " | ".join(cell(m, "Macro avg", k, lambda x: f"**{x}**") for k in ("map50", "map")) + " |")
        tex.append(f"{first} & {show[m]} & & & " + " & ".join(cell(m, "Macro avg", k, lambda x: rf"\textbf{{{x}}}") for k in ("map50", "map")) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]

    text = "\n".join(md) + "\n"
    (out / "results_table.md").write_text(text)
    (out / "results_table.tex").write_text("\n".join(tex) + "\n")
    return text


def results_table_html(metrics: dict, records: list[dict], model_keys: list[str]) -> str:
    terrains = [t for t in TABLE_ORDER if any(r["terrain"] == t for r in records)]
    rows = table_rows(metrics, model_keys)
    keys = ("p", "r", "map50", "map")
    out = ["<table class='paper'><tr><th class='m'>Test set</th><th class='m'>Model</th>"
           "<th>Precision</th><th>Recall</th><th>mAP@50</th><th>mAP@50:95</th></tr>"]

    def cells(group, ks):
        res = []
        for m in rows:
            line = []
            for k in ks:
                v = metrics[m].get(group, {}).get(k, 0.0)
                top = max(metrics[x].get(group, {}).get(k, 0.0) for x in rows)
                line.append(f"<td{' class=best' if abs(v - top) < 5e-4 else ''}>{v:.3f}</td>")
            res.append(line)
        return res

    for terrain in terrains:
        n_img = sum(r["terrain"] == terrain for r in records)
        n_gt = sum(len(r["gt"]) for r in records if r["terrain"] == terrain)
        grid = cells(terrain, keys)
        for i, m in enumerate(rows):
            head = (f"<th class='m' rowspan='{len(rows)}'>{TEST_SET_LABELS.get(terrain, terrain)}"
                    f"<br><span class='muted'>{n_img:,} img, {n_gt:,} persons</span></th>") if i == 0 else ""
            name = MOE_LABEL if m == "MoE top-1" else m
            out.append(f"<tr class='{'moe' if m == 'MoE top-1' else ''}'>{head}<td class='m'>{html.escape(name)}</td>{''.join(grid[i])}</tr>")
    grid = cells("Macro avg", ("map50", "map"))
    for i, m in enumerate(rows):
        head = f"<th class='m' rowspan='{len(rows)}'>Mean over domains</th>" if i == 0 else ""
        name = MOE_LABEL if m == "MoE top-1" else m
        out.append(f"<tr class='{'moe' if m == 'MoE top-1' else ''}'>{head}<td class='m'>{html.escape(name)}</td><td></td><td></td>{''.join(grid[i])}</tr>")
    out.append("</table>")
    return "".join(out)


# --------------------------------------------------------------- report

def save_charts(metrics: dict, routing: dict, names: list[str], out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    groups = [t for t in names] + ["Macro avg"]
    labels = list(metrics)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, key, title in ((axes[0], "map50", "mAP@0.5"), (axes[1], "map", "mAP@0.5:0.95")):
        width = 0.8 / len(labels)
        for i, label in enumerate(labels):
            values = [metrics[label].get(g, {}).get(key, 0.0) for g in groups]
            color = next((c for k, c in METHOD_COLORS.items() if f"({k})" in label or label.startswith(k)), "#9ca3af")
            ax.bar(np.arange(len(groups)) + (i - (len(labels) - 1) / 2) * width, values, width, label=label, color=color,
                   edgecolor="white", linewidth=0.6, hatch="//" if label == "Oracle routing" else None)
        ax.set_xticks(np.arange(len(groups)), groups)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "map_by_terrain.png", dpi=150)
    plt.close(fig)
    files.append("map_by_terrain.png")

    matrix = np.asarray(routing["confusion"], float)
    norm = matrix / np.clip(matrix.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(norm, cmap="Purples", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{int(matrix[i, j])}\n{norm[i, j]:.0%}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.6 else "#111827", fontsize=9)
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("routed to (top-1)")
    ax.set_ylabel("true terrain")
    ax.set_title("Gate routing")
    fig.tight_layout()
    fig.savefig(out / "routing_confusion.png", dpi=150)
    plt.close(fig)
    files.append("routing_confusion.png")
    return files


def draw_example(rec: dict, pred: np.ndarray, chosen: list[int], names: list[str], vis_conf: float, out_path: Path) -> dict:
    pred = pred[pred[:, 4] >= vis_conf] if len(pred) else pred
    order = np.argsort(-pred[:, 4]) if len(pred) else np.zeros(0, int)
    pred = pred[order]
    correct, gt_hit = match(rec["gt"], pred, thresholds=np.array([0.5]))
    tp, fp, fn = int(correct[:, 0].sum()), int((~correct[:, 0]).sum()), int((~gt_hit).sum())

    with Image.open(rec["path"]) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB")
    scale = min(1.0, 1600 / max(image.size))
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    draw = ImageDraw.Draw(image)
    stroke = max(3, image.width // 500)
    for box, hit in zip(rec["gt"], gt_hit):
        if not hit:
            b = box * scale
            draw.rectangle([b[0] - 3, b[1] - 3, b[2] + 3, b[3] + 3], outline=FN_COLOR, width=stroke + 1)
    for row, ok in zip(pred, correct[:, 0]):
        b = row[:4] * scale
        draw.rectangle([b[0] - 2, b[1] - 2, b[2] + 2, b[3] + 2], outline=TP_COLOR if ok else FP_COLOR, width=stroke)
    font = load_font(max(20, image.width // 48))
    routed = " + ".join(names[c] for c in chosen) if chosen else "-"
    probs = "  ".join(f"{n[:3]} {p:.2f}" for n, p in zip(names, rec["probs"]))
    banner = f"routed: {routed}   |   gate {probs}   |   TP {tp}  FP {fp}  FN {fn}"
    box = draw.textbbox((0, 0), banner, font=font)
    draw.rectangle([0, 0, box[2] - box[0] + 20, box[3] - box[1] + 16], fill=(17, 24, 39))
    draw.text((10, 8), banner, fill=(255, 255, 255), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=88)
    return {"tp": tp, "fp": fp, "fn": fn, "routed": routed}


def pick_examples(records, outputs, label, names, per_terrain, vis_conf, seed):
    rng = random.Random(seed)
    picked = {}
    for terrain in names:
        idx = [i for i, rec in enumerate(records) if rec["terrain"] == terrain]
        if not idx:
            continue
        def errors(i):
            pred, _ = outputs[label][i]
            pred = pred[pred[:, 4] >= vis_conf] if len(pred) else pred
            correct, gt_hit = match(records[i]["gt"], pred, thresholds=np.array([0.5]))
            return int((~correct[:, 0]).sum()) + int((~gt_hit).sum())
        hard_n = per_terrain // 2
        hardest = sorted(idx, key=errors, reverse=True)[:hard_n]
        rest = [i for i in idx if i not in set(hardest)]
        sample = rng.sample(rest, min(per_terrain - len(hardest), len(rest)))
        picked[terrain] = [(i, "hard") for i in hardest] + [(i, "random") for i in sorted(sample)]
    return picked


def fmt(v: float) -> str:
    return f"{v:.3f}"


def write_report(out: Path, cache: dict, metrics: dict, routing: dict, speed: dict, gallery: dict,
                 charts: list[str], args, threshold: float, vis_conf: float, test_counts: dict, test_source: str,
                 paper_table: str = "") -> Path:
    names = cache["names"]
    cols = names + ["Macro avg", "All (pooled)"]
    best = {}
    for col in cols:
        for key in ("map50", "map"):
            vals = [(metrics[m].get(col, {}).get(key, -1), m) for m in metrics if m != "Oracle routing"]
            best[(col, key)] = max(vals)[1] if vals else None

    def row(label):
        cells = [f"<th class='m'>{html.escape(label)}</th>"]
        for col in cols:
            m = metrics[label].get(col, {})
            for key in ("map50", "map"):
                cls = " class='best'" if best.get((col, key)) == label else ""
                cells.append(f"<td{cls}>{fmt(m.get(key, 0))}</td>")
            if col not in ("Macro avg",):
                cells.append(f"<td class='pr'>{fmt(m.get('p', 0))} / {fmt(m.get('r', 0))}</td>")
            else:
                cells.append(f"<td class='pr'>{fmt(m.get('p', 0))} / {fmt(m.get('r', 0))}</td>")
        tr = " class='oracle'" if label == "Oracle routing" else (" class='moe'" if label.startswith("MoE") else "")
        return f"<tr{tr}>" + "".join(cells) + "</tr>"

    head1 = "<tr><th></th>" + "".join(f"<th colspan='3' class='grp'>{html.escape(c)}</th>" for c in cols) + "</tr>"
    head2 = "<tr><th>method</th>" + "".join("<th>mAP50</th><th>mAP50-95</th><th>P / R</th>" for _ in cols) + "</tr>"
    gt_row = "<tr class='muted'><th class='m'>GT persons / images</th>" + "".join(
        f"<td colspan='3'>{metrics[next(iter(metrics))].get(c, {}).get('n_gt', '')}"
        f"{' / ' + str(test_counts.get(c, '')) if c in test_counts else ''}</td>" for c in cols) + "</tr>"
    table = f"<table class='metrics'>{head1}{head2}{''.join(row(m) for m in metrics)}{gt_row}</table>"

    moe = metrics["MoE top-2"]["Macro avg"]["map50"]
    moe1 = metrics["MoE top-1"]["Macro avg"]["map50"]
    oracle = metrics["Oracle routing"]["Macro avg"]["map50"]
    singles = {m: v["Macro avg"]["map50"] for m, v in metrics.items() if not m.startswith(("MoE", "All 3", "Oracle"))}
    best_single = max(singles, key=singles.get)
    cards = [
        ("MoE top-2 · macro mAP@0.5", fmt(moe), f"top-1: {fmt(moe1)}"),
        ("Best single model", fmt(singles[best_single]), html.escape(best_single)),
        ("Oracle routing (upper bound)", fmt(oracle), f"gap to MoE: {oracle - moe:+.3f}"),
        ("Gate routing accuracy", f"{routing['accuracy']:.1%}", f"top-2 triggered on {routing['top2_rate_all']:.1%} of images"),
    ]
    cards_html = "".join(
        f"<div class='card'><div class='k'>{k}</div><div class='v'>{v}</div><div class='s'>{s}</div></div>" for k, v, s in cards
    )

    routing_rows = "".join(
        f"<tr><th class='m'>{t}</th><td>{routing['per_terrain'][t]['n']}</td><td>{routing['per_terrain'][t]['acc']:.1%}</td>"
        f"<td>{routing['per_terrain'][t]['top2_rate']:.1%}</td><td>{routing['per_terrain'][t]['mean_conf']:.3f}</td></tr>"
        for t in names if t in routing["per_terrain"]
    )
    speed_rows = "".join(f"<tr><th class='m'>{html.escape(k)}</th><td>{v:.1f} ms</td></tr>" for k, v in speed.items())

    tabs, panels = [], []
    for i, (terrain, items) in enumerate(gallery.items()):
        tabs.append(f"<button class='tab{' on' if i == 0 else ''}' data-t='{terrain}'>{terrain} ({len(items)})</button>")
        figs = "".join(
            f"<figure><a href='{html.escape(it['img'])}' target='_blank'><img loading='lazy' src='{html.escape(it['img'])}'></a>"
            f"<figcaption><b>{html.escape(it['name'])}</b> <span class='tag {it['kind']}'>{it['kind']}</span><br>"
            f"routed: {html.escape(it['routed'])} · GT {it['n_gt']} · "
            f"<span class='tp'>TP {it['tp']}</span> <span class='fp'>FP {it['fp']}</span> <span class='fn'>FN {it['fn']}</span>"
            f"</figcaption></figure>"
            for it in items
        )
        panels.append(f"<div class='panel{' on' if i == 0 else ''}' id='p-{terrain}'>{figs}</div>")

    weights = "".join(f"<li><b>{html.escape(k)}</b>: <code>{html.escape(Path(v).name)}</code></li>" for k, v in cache["weights"].items())
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MoE Evaluation Report</title>
<style>
:root{{--bg:#fff;--fg:#111827;--mut:#6b7280;--line:#e5e7eb;--soft:#f9fafb;--acc:#7c3aed}}
body{{margin:0;font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;background:var(--bg);color:var(--fg)}}
main{{max-width:1280px;margin:0 auto;padding:28px 20px 60px}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:19px;margin:36px 0 10px}}
.sub{{color:var(--mut);margin:0 0 20px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}}
.card{{border:1px solid var(--line);border-radius:12px;padding:14px 16px;background:var(--soft)}}
.card .k{{color:var(--mut);font-size:13px}} .card .v{{font-size:30px;font-weight:700;margin:2px 0}} .card .s{{color:var(--mut);font-size:13px}}
.scroll{{overflow-x:auto;border:1px solid var(--line);border-radius:12px}}
table{{border-collapse:collapse;width:100%;font-size:13.5px}}
th,td{{padding:7px 10px;border-bottom:1px solid var(--line);text-align:center;white-space:nowrap}}
th.m{{text-align:left;font-weight:600}} th.grp{{background:var(--soft)}}
td.best{{font-weight:700;color:var(--acc)}} td.pr{{color:var(--mut)}}
tr.moe{{background:#faf5ff}} table.paper th.m{{vertical-align:top}} table.paper td.m{{text-align:left}} .muted{{color:var(--mut);font-weight:400;font-size:12.5px}} tr.oracle{{color:var(--mut);font-style:italic}} tr.muted td,tr.muted th{{color:var(--mut);font-size:12.5px}}
img.chart{{max-width:100%;border:1px solid var(--line);border-radius:12px;margin-top:8px}}
.two{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px;align-items:start}} .two>*{{min-width:0}}
.tabs{{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 14px}}
.tab{{border:1px solid var(--line);background:var(--soft);border-radius:999px;padding:6px 14px;cursor:pointer;font:inherit}}
.tab.on{{background:var(--fg);color:#fff;border-color:var(--fg)}}
.panel{{display:none;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:14px}} .panel.on{{display:grid}}
figure{{margin:0;border:1px solid var(--line);border-radius:12px;overflow:hidden;background:var(--soft)}}
figure img{{width:100%;display:block}} figcaption{{padding:8px 10px;font-size:13px;color:#374151;word-break:break-all}}
.tag{{font-size:11px;padding:1px 7px;border-radius:999px;background:#e5e7eb}} .tag.hard{{background:#fee2e2;color:#991b1b}}
.tp{{color:#15803d}} .fp{{color:#b91c1c}} .fn{{color:#a16207}}
.legend span{{display:inline-block;width:14px;height:14px;border:3px solid;vertical-align:-3px;margin:0 4px 0 12px}}
ul.notes li{{margin:4px 0}} code{{background:var(--soft);padding:1px 5px;border-radius:4px}}
@media (max-width:760px){{.two{{grid-template-columns:1fr}} .panel{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Terrain-gated MoE · evaluation report</h1>
<p class="sub">{cache['created']} · test set: {html.escape(test_source)} · expert input {cache['imgsz']}px · gate TTA {'on' if cache['tta'] else 'off'} · top-2 threshold {threshold} · device {cache['device']}</p>
<div class="cards">{cards_html}</div>

<h2>Results table</h2>
<p class="sub">Same layout as the team table. Bold = best per column within each test set. MoE (ours) = gate + top-1 routing. Copy-ready versions: <code>results_table.md</code>, <code>results_table.tex</code>.</p>
<div class="scroll">{paper_table}</div>

<h2>All methods (person class)</h2>
<p class="sub">Bold purple = best non-oracle value in that column. P / R are at the max-F1 confidence, as in YOLOv5 val.py. Macro avg = mean over the three terrains; All (pooled) = every image together (dominated by the largest test set).</p>
<div class="scroll">{table}</div>
<img class="chart" src="charts/{charts[0]}" alt="mAP by terrain">

<h2>Gate routing</h2>
<div class="two">
<div><div class="scroll"><table><tr><th class="m">true terrain</th><th>images</th><th>routed correctly</th><th>top-2 triggered</th><th>mean max p</th></tr>{routing_rows}</table></div>
<h2 style="margin-top:22px">Speed (mean per image)</h2>
<div class="scroll"><table>{speed_rows}</table></div></div>
<img class="chart" src="charts/{charts[1]}" alt="routing confusion">
</div>

<h2>Examples · {html.escape(args.mode.replace('top', 'MoE top-'))} predictions at conf ≥ {vis_conf}</h2>
<p class="legend"><span style="border-color:#22c55e"></span>correct detection (TP, IoU≥0.5)<span style="border-color:#ef4444"></span>false positive<span style="border-color:#facc15"></span>missed person (FN)</p>
<p class="sub">Half of each tab is the images with the most errors (<i>hard</i>), half random. Click an image to open it full size.</p>
<div class="tabs">{''.join(tabs)}</div>{''.join(panels)}

<h2>Setup and caveats</h2>
<ul class="notes">
{weights}
<li>Test images per terrain: {', '.join(f'{k} {v}' for k, v in test_counts.items())}.</li>
<li>Every expert sees the original full-resolution image, resized to {cache['imgsz']}px by YOLOv5. The gate sees a 224px crop.</li>
<li>If a test image was in an expert's training split, that expert's numbers are optimistic. Check that the test folders match each expert's held-out split.</li>
<li>Oracle routing is the best any gate could do with these experts. The gap between MoE and oracle is the cost of routing mistakes.</li>
<li>All 3 experts + NMS runs every expert on every image (3× the compute). If MoE matches or beats it, routing is worth it.</li>
</ul>
</main>
<script>
document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>{{
document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('on',x===b));
document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('on',p.id==='p-'+b.dataset.t));}}));
</script></body></html>"""
    path = out / "report.html"
    path.write_text(page)
    return path


# ------------------------------------------------------------------ main

def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(args.seed)
    ev = config.get("eval") or {}
    names = list(config["class_names"])
    threshold = float(args.threshold if args.threshold is not None else config["gate"]["confidence_threshold"])
    merge_iou = float(config["experts"].get("merge_iou", 0.5))
    max_det = int(ev.get("max_det", 300))
    vis_conf = float(ev.get("vis_conf", 0.25))

    out = Path(args.output)
    out = out if out.is_absolute() else project_root() / out
    out.mkdir(parents=True, exist_ok=True)
    cache_file = out / "predictions.pkl"
    test_source = "custom folders (eval.test_sets)" if ev.get("test_sets") else f"gate '{args.split}' split"

    if args.from_cache:
        if not cache_file.is_file():
            raise FileNotFoundError(f"No cache at {cache_file}. Run without --from-cache first.")
        cache = pickle.loads(cache_file.read_bytes())
        print(f"Loaded {len(cache['records'])} cached images from {cache_file}")
    else:
        tests = collect_test_images(config, args.split, args.limit, args.seed)
        if not tests:
            raise RuntimeError("No test images found.")
        device = pick_device(args.device)
        imgsz = int(args.imgsz or config["experts"]["image_size"])
        print(f"Device: {device}   expert imgsz: {imgsz}   gate TTA: {args.tta}")
        print("Test images: " + ", ".join(f"{t} {len(f)}" for t, f in tests.items()) + f"   ({test_source})")
        cache = run_inference(config, tests, device, imgsz, args.tta)
        cache_file.write_bytes(pickle.dumps(cache))

    records = cache["records"]
    missing = sum(not r["has_label"] for r in records)
    if missing:
        print(f"Note: {missing} images had no label file and count as having no persons.")
    test_counts = {t: sum(r["terrain"] == t for r in records) for t in names if any(r["terrain"] == t for r in records)}

    methods = build_methods(names, cache["model_keys"], threshold, merge_iou, max_det)
    metrics, outputs = score(records, methods, names)

    # routing
    confusion = np.zeros((len(names), len(names)), int)
    per_terrain = {}
    for t_index, terrain in enumerate(names):
        recs = [r for r in records if r["terrain"] == terrain]
        if not recs:
            continue
        top = [int(np.argmax(r["probs"])) for r in recs]
        for k in top:
            confusion[t_index, k] += 1
        per_terrain[terrain] = {
            "n": len(recs),
            "acc": float(np.mean([k == t_index for k in top])),
            "top2_rate": float(np.mean([r["probs"].max() < threshold for r in recs])),
            "mean_conf": float(np.mean([r["probs"].max() for r in recs])),
        }
    routing = {
        "confusion": confusion.tolist(),
        "accuracy": float(np.trace(confusion) / max(1, confusion.sum())),
        "top2_rate_all": float(np.mean([r["probs"].max() < threshold for r in records])),
        "per_terrain": per_terrain,
    }

    # speed
    t = {k: float(np.mean(v)) for k, v in cache["timings"].items() if v}
    expert_ms = float(np.mean([t[n] for n in names]))
    speed = {"Gate (CNN)": t["gate"]}
    for i, n in enumerate(names):
        speed[f"Expert {i + 1} ({n})"] = t[n]
    for key, value in t.items():
        if key.startswith("base:"):
            speed[key[5:]] = value
    speed["MoE top-1 (gate + 1 expert)"] = t["gate"] + expert_ms
    speed["MoE top-2 (gate + 1–2 experts)"] = t["gate"] + expert_ms * (1 + routing["top2_rate_all"])
    speed["Run all 3 experts"] = 3 * expert_ms

    # files
    charts_dir = out / "charts"
    charts_dir.mkdir(exist_ok=True)
    charts = save_charts(metrics, routing, names, charts_dir)
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "terrain", "mAP50", "mAP50-95", "precision", "recall", "f1", "n_gt", "n_pred"])
        for label, cols in metrics.items():
            for col, m in cols.items():
                writer.writerow([label, col, f"{m['map50']:.4f}", f"{m['map']:.4f}", f"{m['p']:.4f}", f"{m['r']:.4f}",
                                 f"{m['f1']:.4f}", m.get("n_gt", ""), m.get("n_pred", "")])
    (out / "summary.json").write_text(json.dumps({"metrics": metrics, "routing": routing, "speed_ms": speed,
                                                   "threshold": threshold, "imgsz": cache["imgsz"], "tta": cache["tta"],
                                                   "test_counts": test_counts, "weights": cache["weights"]}, indent=2))

    label = "MoE top-2" if args.mode == "top2" else "MoE top-1"
    picked = pick_examples(records, outputs, label, names, args.vis, vis_conf, args.seed)
    gallery = {}
    for terrain, items in picked.items():
        gallery[terrain] = []
        for i, kind in tqdm(items, desc=f"drawing {terrain}", leave=False):
            rec = records[i]
            pred, chosen = outputs[label][i]
            name = Path(rec["path"]).name
            rel = f"vis/{terrain}/{Path(name).stem}.jpg"
            stats = draw_example(rec, pred, chosen, names, vis_conf, out / rel)
            gallery[terrain].append({"img": rel, "name": name, "kind": kind, "n_gt": len(rec["gt"]), **stats})

    table_md = write_results_table(out, metrics, records, cache["model_keys"])
    paper_html = results_table_html(metrics, records, cache["model_keys"])
    report = write_report(out, cache, metrics, routing, speed, gallery, charts, args, threshold, vis_conf, test_counts,
                          test_source, paper_html)

    # console summary
    print()
    header = f"{'method':<30}" + "".join(f"{t[:10]:>12}" for t in list(test_counts) + ["Macro avg"])
    print("mAP@0.5 per terrain")
    print(header)
    for lbl, cols in metrics.items():
        print(f"{lbl:<30}" + "".join(f"{cols.get(t, {}).get('map50', 0):>12.3f}" for t in list(test_counts) + ["Macro avg"]))
    print(f"\nGate routing accuracy: {routing['accuracy']:.1%}   top-2 triggered: {routing['top2_rate_all']:.1%}")
    print("\nResults table (also in results_table.md / results_table.tex):")
    print(table_md)
    print(f"Report: {report}")
    print(f"Metrics CSV: {out / 'metrics.csv'}")


if __name__ == "__main__":
    main()
