# Terrain-Gated Mixture of Experts for Drone SAR Human Detection

A small CNN gate looks at each drone image, decides its terrain (maritime, urban or vegetation), and sends the full-resolution image to the YOLOv5 expert trained for that terrain.

![architecture](docs/moe_architecture.png)

| Expert | Trained on | Weights |
|---|---|---|
| 1 · Maritime | VisDrone → SeaDronesSee | `weights/best_seabornsee.pt` |
| 2 · Urban | VisDrone → NTIC | `weights/nitc_expert_export/weights/best.pt` |
| 3 · Vegetation | VisDrone → Heridal | `weights/heridal_random_best.pt` |
| Gate | MobileNetV3-Large, 3 classes | `weights/gate_best.pt` |

## Results

End-to-end person detection on the team's shared test sets (`test_images/`), the same images used for the Baseline and Multi-domain models.

| Test set | Model | Precision | Recall | mAP@50 | mAP@50:95 |
|---|---|---|---|---|---|
| Heridal (vegetation)<br>159 img, 288 persons | Baseline | 0.826 | 0.740 | 0.807 | 0.415 |
| | Multi-domain (downsampled) | **0.842** | 0.757 | 0.828 | 0.404 |
| | Multi-domain (weighted) | 0.834 | **0.788** | **0.843** | **0.433** |
| | **MoE (ours)** | 0.803 | 0.781 | 0.801 | 0.369 |
| SeaDronesSee (maritime)<br>1,047 img, 4,372 persons | Baseline | 0.330 | 0.259 | 0.187 | 0.064 |
| | Multi-domain (downsampled) | 0.808 | 0.762 | 0.760 | 0.320 |
| | Multi-domain (weighted) | **0.876** | **0.848** | **0.844** | **0.408** |
| | **MoE (ours)** | 0.834 | 0.809 | 0.806 | 0.371 |
| NITC (urban)<br>182 img, 1,367 persons | Baseline | 0.109 | 0.040 | 0.024 | 0.005 |
| | Multi-domain (downsampled) | **0.985** | **0.990** | **0.992** | **0.814** |
| | Multi-domain (weighted) | 0.980 | 0.980 | 0.991 | 0.806 |
| | **MoE (ours)** | 0.980 | 0.987 | **0.992** | 0.800 |
| Mean over domains | Baseline | | | 0.339 | 0.161 |
| | Multi-domain (downsampled) | | | 0.860 | 0.513 |
| | Multi-domain (weighted) | | | **0.893** | **0.549** |
| | **MoE (ours)** | | | 0.866 | 0.514 |

- The gate routes **100%** of the 1,388 test images to the correct expert.
- MoE vs. Baseline: mean mAP@50 0.339 → 0.866. MoE matches downsampled multi-domain fine-tuning (0.866 vs 0.860) and beats it on maritime (+0.046), but multi-domain with weighted sampling is best overall (0.893).
- Routing matters: running all 3 experts on every image and merging with NMS gives only 0.757 mean mAP@50 (3x the compute); each expert alone scores 0.28–0.34.

Full report with charts and annotated examples: [`runs/moe_eval_sar/report.html`](runs/moe_eval_sar/report.html) (download or clone, then open it in a browser). Baseline and Multi-domain numbers are from the team's YOLOv5 `val.py` runs.

## Setup

```bash
conda create -y -n moe python=3.12
conda activate moe
python -m pip install -r requirements.txt
```

Put the datasets in this folder under these names (paths are set in `config.yaml`):

```
seadronessee/                               # maritime (Hugging Face: dronefreak/SeaDronesSee)
NTIC_human_dataset_70_percent/              # urban
Heridal Human Detection.v1i.yolov5pytorch/  # vegetation (Roboflow export)
extra_urban/VisDrone2019-DET-train/         # extra urban images, gate training only
test_images/                                # team's shared test sets (sar_person_*_test/)
```

## Run

The trained gate and expert weights are already in `weights/`. To run the MoE:

```bash
python eval_moe.py --output runs/moe_eval_sar   # evaluate on test_images/ (~5 min on an M-series Mac)
open runs/moe_eval_sar/report.html               # metrics, charts, annotated examples

python run_moe.py --source <image or folder> --save   # detect on any images -> runs/moe/
```

To retrain the gate:

```bash
python build_gate_cache.py                 # one-time 320px image cache (~3 min)
python train_gate.py                       # -> weights/gate_best.pt
python eval_gate.py --split test           # accuracy + confusion matrix -> runs/gate_eval/
```

`eval_moe.py` also writes `results_table.md` and `results_table.tex`: per-test-set Precision / Recall / mAP@50 / mAP@50:95 plus a mean-over-domains row, in the same layout as the team table. To add the team's Baseline and Multi-domain models as rows, list their weights in `config.yaml`:

```yaml
experts:
  baselines:
    Baseline: weights/<baseline>.pt
    Multi-domain: weights/<multi_domain>.pt
```

Useful flags: `eval_moe.py --limit 10` (quick check), `--from-cache` (re-render the report), `--tta` (5-crop gate). `run_moe.py --mode top2` adds the second-best expert when the gate is unsure.

## Layout

```
moe/                 library code
  data.py            dataset discovery and splits
  cache.py           320px gate-image cache
  gate_data.py       gate dataset and augmentation
  gate_model.py      MobileNetV3 gate
  experts.py         YOLOv5 experts, router, box merging
  utils.py           config, device, seed
build_gate_cache.py  train_gate.py  eval_gate.py  ood_check.py  run_moe.py  eval_moe.py
tools/audit_datasets.py   one-off dataset audit -> docs/audit_report.md
config.yaml          all settings
weights/             gate + expert checkpoints (YOLOv5 code in weights/nitc_expert_export/yolov5)
```

## Notes

- Test sets: `eval.test_sets` in `config.yaml` points at the team's shared split in `test_images/`. The gate's own training/validation splits (official NTIC and Heridal splits, SeaDronesSee val split by frame ID) are separate; `runs/moe_eval/` holds an earlier evaluation on those.
- These are in-domain results, and NTIC test frames come from the same videos as its training frames, so urban scores are optimistic.
- Some SeaDronesSee labels are offset by a few pixels, which lowers maritime mAP even where detections are correct.
- `ood_check.py --source <folder> --expect urban` routes outside images to check the gate learned terrain, not dataset cues (VisDrone val: 100% urban).
