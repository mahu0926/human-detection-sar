"""Experts, routing and box merging for the terrain-gated MoE.

``MoEPipeline`` = gate (MobileNetV3) -> router -> YOLOv5 expert(s) -> merge.
The gate sees a downscaled 224px crop; experts always see the original image.
"""

from __future__ import annotations

import pathlib as _pathlib
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms

from moe.cache import prepare_gate_image
from moe.data import project_root
from moe.gate_data import build_transforms, build_tta_transform
from moe.gate_model import load_gate

# Expert checkpoints were pickled under Python 3.13, where pathlib's classes live
# in `pathlib._local`. Alias it so they also unpickle on Python 3.12 and older.
if "pathlib._local" not in sys.modules:
    sys.modules["pathlib._local"] = _pathlib

EXPERT_COLORS = {
    "maritime": (37, 99, 235),
    "urban": (234, 88, 12),
    "vegetation": (22, 163, 74),
}


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def choose_experts(probs: np.ndarray, mode: str, threshold: float) -> list[int]:
    """Top-1 always. Top-2 mode adds the runner-up when the gate is unsure."""
    order = np.argsort(-probs)
    chosen = [int(order[0])]
    if mode == "top2" and float(probs[order[0]]) < threshold:
        chosen.append(int(order[1]))
    return chosen


def box_iou(box: np.ndarray, other: np.ndarray) -> float:
    x1 = max(float(box[0]), float(other[0]))
    y1 = max(float(box[1]), float(other[1]))
    x2 = min(float(box[2]), float(other[2]))
    y2 = min(float(box[3]), float(other[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_b = max(0.0, float(other[2] - other[0])) * max(0.0, float(other[3] - other[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def merge_nms(detections: list[dict], iou_threshold: float) -> list[dict]:
    """Class-aware NMS. Experts that both emit class 0 (human) suppress each other."""
    if not detections:
        return []
    kept: list[dict] = []
    classes = sorted({int(det["cls"]) for det in detections})
    for class_id in classes:
        group = [det for det in detections if int(det["cls"]) == class_id]
        boxes = torch.tensor([det["xyxy"] for det in group], dtype=torch.float32)
        scores = torch.tensor([det["score"] for det in group], dtype=torch.float32)
        keep = nms(boxes, scores, iou_threshold)
        kept.extend(group[int(index)] for index in keep.tolist())
    kept.sort(key=lambda det: det["score"], reverse=True)
    return kept


def merge_wbf(detections: list[dict], iou_threshold: float) -> list[dict]:
    """Greedy weighted-box fusion within each class."""
    if not detections:
        return []
    fused: list[dict] = []
    classes = sorted({int(det["cls"]) for det in detections})
    for class_id in classes:
        remaining = [det for det in detections if int(det["cls"]) == class_id]
        remaining.sort(key=lambda det: det["score"], reverse=True)
        while remaining:
            seed = remaining.pop(0)
            cluster = [seed]
            stay = []
            for det in remaining:
                if box_iou(np.asarray(seed["xyxy"]), np.asarray(det["xyxy"])) >= iou_threshold:
                    cluster.append(det)
                else:
                    stay.append(det)
            remaining = stay
            weights = np.array([det["score"] for det in cluster], dtype=np.float64)
            coords = np.average(
                np.stack([np.asarray(det["xyxy"], dtype=np.float64) for det in cluster]),
                axis=0,
                weights=weights,
            )
            best = max(cluster, key=lambda det: det["score"])
            fused.append(
                {
                    "xyxy": coords.tolist(),
                    "score": float(weights.mean()),
                    "cls": class_id,
                    "cls_name": best["cls_name"],
                    "expert": "+".join(dict.fromkeys(det["expert"] for det in cluster)),
                }
            )
    fused.sort(key=lambda det: det["score"], reverse=True)
    return fused


def merge_detections(detections: list[dict], method: str, iou_threshold: float) -> list[dict]:
    if method == "wbf":
        return merge_wbf(detections, iou_threshold)
    if method == "nms":
        return merge_nms(detections, iou_threshold)
    raise ValueError(f"Unknown merge method {method!r}. Use nms or wbf.")


def _class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    if isinstance(names, (list, tuple)) and class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def detections_from_results(results, expert_name: str) -> list[dict]:
    xyxy = results.xyxy[0]
    array = xyxy.detach().cpu().numpy() if hasattr(xyxy, "detach") else np.asarray(xyxy)
    names = getattr(results, "names", None)
    parsed = []
    for row in array:
        class_id = int(row[5])
        parsed.append(
            {
                "xyxy": [float(value) for value in row[:4]],
                "score": float(row[4]),
                "cls": class_id,
                "cls_name": _class_name(names, class_id),
                "expert": expert_name,
            }
        )
    return parsed


class ExpertBank:
    """Lazy YOLOv5 loader. One model object per resolved weight file."""

    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        self.cache: dict[str, tuple[object, torch.device, bool]] = {}
        expert_cfg = config["experts"]
        self.weight_paths = {
            name: project_root() / relative
            for name, relative in expert_cfg["weights"].items()
        }
        self.fallback = expert_cfg.get("fallback", "yolov5s.pt")
        self.conf = float(expert_cfg["conf"])
        self.iou = float(expert_cfg["iou"])
        self.image_size = int(expert_cfg["image_size"])

    def _load(self, weight_arg: str) -> tuple[object, torch.device]:
        repo = self.config["experts"].get("repo")
        repo_path = (project_root() / repo) if repo else None
        if repo_path is not None and (repo_path / "hubconf.py").is_file():
            model = torch.hub.load(str(repo_path), "custom", path=weight_arg, source="local", _verbose=False)
        else:
            if repo:
                warn(f"YOLOv5 repo not found at {repo_path}. Downloading ultralytics/yolov5 via torch.hub.")
            model = torch.hub.load("ultralytics/yolov5", "custom", path=weight_arg, trust_repo=True)
        model.conf = self.conf
        model.iou = self.iou
        try:
            model.to(self.device)
            used = self.device
        except Exception as exc:
            warn(f"Could not move YOLOv5 to {self.device} ({exc}). Using CPU.")
            model.to(torch.device("cpu"))
            used = torch.device("cpu")
        return model, used

    def load_path(self, weight_path: str):
        """Load an arbitrary YOLOv5 checkpoint (used for an optional baseline)."""
        if weight_path not in self.cache:
            model, used = self._load(weight_path)
            self.cache[weight_path] = (model, used, False)
        return self.cache[weight_path][0]

    def get(self, expert_name: str, allow_fallback: bool = True):
        path = self.weight_paths[expert_name]
        fallback = not path.is_file()
        if fallback and not allow_fallback:
            raise FileNotFoundError(f"Expert weight for {expert_name} not found: {path}")
        weight_arg = self.fallback if fallback else str(path)
        if fallback:
            warn(
                f"Expert weight missing for {expert_name}: {path}. "
                f"Falling back to stock {self.fallback}. "
                "Replace that file when it arrives; no code change is required."
            )
        if weight_arg not in self.cache:
            model, used = self._load(weight_arg)
            self.cache[weight_arg] = (model, used, fallback)
        model, used, did_fallback = self.cache[weight_arg]
        return model, used, did_fallback, weight_arg

    def predict(self, image: Image.Image, expert_name: str) -> tuple[list[dict], str]:
        model, _device, _fallback, weight_arg = self.get(expert_name)
        results = model(image, size=self.image_size)
        return detections_from_results(results, expert_name), weight_arg


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def draw_result(image: Image.Image, result: dict, names: list[str]) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    stroke = max(2, min(width, height) // 400)
    font = load_font(max(16, min(width, height) // 40))
    probs = result["gate_probs"]
    banner = "expert: " + result["expert"] + "   " + "  ".join(
        f"{name} {prob:.2f}" for name, prob in zip(names, probs)
    )
    text_box = draw.textbbox((0, 0), banner, font=font)
    pad = 8
    draw.rectangle(
        [0, 0, text_box[2] - text_box[0] + pad * 2, text_box[3] - text_box[1] + pad * 2],
        fill=(0, 0, 0),
    )
    draw.text((pad, pad), banner, fill=(255, 255, 255), font=font)
    for det in result["detections"]:
        color_name = det["expert"].split("+")[0]
        color = EXPERT_COLORS.get(color_name, (220, 38, 38))
        box = [float(value) for value in det["xyxy"]]
        draw.rectangle(box, outline=color, width=stroke)
        tag = f"{det['cls_name']} {det['score']:.2f} {det['expert']}"
        tag_box = draw.textbbox((0, 0), tag, font=font)
        tag_w = tag_box[2] - tag_box[0]
        tag_h = tag_box[3] - tag_box[1]
        top = max(0, box[1] - tag_h - 4)
        draw.rectangle([box[0], top, box[0] + tag_w + 6, top + tag_h + 4], fill=color)
        draw.text((box[0] + 3, top + 2), tag, fill=(255, 255, 255), font=font)
    return canvas


class MoEPipeline:
    def __init__(self, config: dict, device: torch.device, tta: bool = False):
        gate_path = Path(config["gate"]["weights"])
        if not gate_path.is_absolute():
            gate_path = project_root() / gate_path
        if not gate_path.is_file():
            raise FileNotFoundError(
                f"Gate checkpoint not found at {gate_path}. "
                "Train it with `python train_gate.py`. Expert weights may be missing; "
                "the gate checkpoint may not."
            )
        self.names = list(config["class_names"])
        self.model, checkpoint = load_gate(str(gate_path), device)
        saved = list(checkpoint.get("class_names", self.names))
        if saved != self.names:
            raise ValueError(f"Checkpoint classes {saved} do not match config {self.names}")
        saved_size = int(checkpoint.get("input_size", config["input_size"]))
        if saved_size != int(config["input_size"]):
            warn(
                f"Config input_size is {config['input_size']} but the checkpoint was "
                f"trained at {saved_size}. Using the checkpoint size."
            )
        self.input_size = saved_size
        self.tta = tta
        self.transform = (
            build_tta_transform(self.input_size) if tta else build_transforms(self.input_size, train=False)
        )
        self.device = device
        self.threshold = float(config["gate"]["confidence_threshold"])
        self.merge_method = config["experts"].get("merge", "nms")
        self.merge_iou = float(config["experts"].get("merge_iou", 0.5))
        self.experts = ExpertBank(config, device)

    @torch.no_grad()
    def gate_probs(self, image: Image.Image) -> np.ndarray:
        tensor = self.transform(prepare_gate_image(image))
        tensor = tensor.to(self.device) if self.tta else tensor.unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(tensor), dim=1).mean(0)
        return probs.detach().cpu().numpy()

    def run_image(self, path: Path, mode: str) -> dict:
        with Image.open(path) as handle:
            original = handle.convert("RGB")
        # The gate gets a downscaled copy; `original` stays full resolution for the expert.
        probs = self.gate_probs(original)
        chosen = choose_experts(probs, mode, self.threshold)
        detections = []
        weights_used = []
        for class_id in chosen:
            expert_name = self.names[class_id]
            found, weight_arg = self.experts.predict(original, expert_name)
            detections.extend(found)
            weights_used.append(f"{expert_name}:{weight_arg}")
        if len(chosen) > 1:
            detections = merge_detections(detections, self.merge_method, self.merge_iou)
        expert_label = "+".join(self.names[class_id] for class_id in chosen)
        return {
            "path": str(path),
            "gate_probs": [float(value) for value in probs],
            "expert_ids": chosen,
            "expert": expert_label,
            "detections": detections,
            "num_detections": len(detections),
            "weights_used": weights_used,
            "image": original,
        }
