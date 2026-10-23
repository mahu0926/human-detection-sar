# Dataset audit

Generated: 2026-10-22T02:20:31

The gate's label is the source dataset, not a bounding-box class.
Original images were only read. Nothing on disk was resized or copied.

## Summary

| Dataset | Class | Split | Images | Readable | Corrupt | Grayscale | Alpha | Other mode | Non-JPG/PNG | Missing label | Empty label | Bad label lines |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heridal | 2 vegetation | test | 157 | 157 | 0 | 0 | 0 | 0 | 0 | 0 | 45 | 0 |
| heridal | 2 vegetation | train | 1106 | 1106 | 0 | 0 | 0 | 0 | 0 | 0 | 431 | 0 |
| heridal | 2 vegetation | val | 316 | 316 | 0 | 0 | 0 | 0 | 0 | 0 | 85 | 0 |
| ntic | 1 urban | test | 365 | 365 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ntic | 1 urban | train | 1277 | 1277 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ntic | 1 urban | val | 182 | 182 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| seadronessee | 0 maritime | train | 8930 | 8930 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| seadronessee | 0 maritime | val | 1547 | 1547 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Resolution is the decoded PIL size. Min and max are by pixel area.
Aspect is width/height, reported as min / median / max.

| Dataset | Split | Min W×H | Max W×H | Most common W×H | Aspect min / median / max |
| --- | --- | --- | --- | --- | --- |
| heridal | test | 4000×3000 | 4000×3000 | 4000×3000 (157) | 1.333 / 1.333 / 1.333 |
| heridal | train | 4000×3000 | 4000×3000 | 4000×3000 (1106) | 1.333 / 1.333 / 1.333 |
| heridal | val | 4000×2250 | 4000×3000 | 4000×3000 (312), 4000×2250 (4) | 1.333 / 1.333 / 1.778 |
| ntic | test | 1920×1080 | 1920×1080 | 1920×1080 (365) | 1.778 / 1.778 / 1.778 |
| ntic | train | 1920×1080 | 1920×1080 | 1920×1080 (1277) | 1.778 / 1.778 / 1.778 |
| ntic | val | 1920×1080 | 1920×1080 | 1920×1080 (182) | 1.778 / 1.778 / 1.778 |
| seadronessee | train | 1225×926 | 5456×3632 | 3840×2160 (5800), 1920×1080 (1576), 5456×3632 (860) | 0.666 / 1.778 / 1.778 |
| seadronessee | val | 1229×932 | 5456×3632 | 3840×2160 (1030), 1920×1080 (258), 5456×3632 (149) | 0.666 / 1.778 / 1.778 |

## Gate label counts

These counts are what the gate trains on. They are not object-instance counts.

| Class | Train | Val | Test | Total | Share |
| --- | --- | --- | --- | --- | --- |
| maritime | 8930 | 1547 | 0 | 10477 | 75.5% |
| urban | 1277 | 182 | 365 | 1824 | 13.1% |
| vegetation | 1106 | 316 | 157 | 1579 | 11.4% |

## Notes for training

- Class imbalance is large: the biggest class has 10477 images and the smallest has 1579 (6.6×). `train_gate.py` uses a `WeightedRandomSampler` with inverse-frequency weights.
- Resolutions are not shared across datasets (see the geometry table). A gate that sees the raw file can classify by image size or camera color instead of terrain. Training therefore random-crops, flips, color-jitters, and slightly blurs. Validation and routing letterbox to the configured input size without rewriting the file. Experts still open the original full-resolution image.
- Official split folders are kept. A seeded 70/20/10 split is applied only when a dataset has no split folders. SeaDronesSee is `preserve_splits: true` because neighboring frames come from the same video.
- SeaDronesSee has no test split in this mirror (the official test set has no public labels). It was not invented by reshuffling train/val. Test-set gate metrics will not include maritime until a video-safe holdout exists. Score maritime with `python eval_gate.py --split val`.
- Every indexed image decoded with PIL and has a matching label file.

## seadronessee — class 0 maritime

Root: `/Users/manasvialam/Documents/CMU/Semester_1/Deep learning/Project/Midterm/seadronessee`

Split policy: official folders only (`preserve_splits`). Random assignment is refused.

### Folder structure

```text
seadronessee/
├── 1 .jpg, 1 .md
└── data/
    ├── 2 .yaml
    ├── images/
    │   ├── train/
    │   │   ├── shard_000/
    │   │   │   └── 3000 .jpg, 1 .jsonl
    │   │   ├── shard_001/
    │   │   │   └── 3000 .jpg, 1 .jsonl
    │   │   └── shard_002/
    │   │       └── 2930 .jpg, 1 .jsonl
    │   └── val/
    │       └── shard_000/
    │           └── 1547 .jpg, 1 .jsonl
    └── labels/
        ├── train/
        │   ├── shard_000/
        │   │   └── 3000 .txt
        │   ├── shard_001/
        │   │   └── 3000 .txt
        │   └── shard_002/
        │       └── 2930 .txt
        └── val/
            └── shard_000/
                └── 1547 .txt
```

Non-image files skipped by the gate index: 4 `.jsonl`, 2 `.yaml`, 1 `.md`.

### train (official)

- Images: 8930 (8930 .jpg)
- Readable: 8930. Corrupt: 0.
- Modes: 8930 RGB. Grayscale: 0. Alpha: 0.
- Labels: 8930 matched, 0 missing, 0 empty, 57760 boxes, 0 malformed lines.
- Most common sizes: 3840×2160 × 5800, 1920×1080 × 1576, 5456×3632 × 860, 3632×5456 × 282, 1230×933 × 126
- Aspect buckets: 16:9-ish 8236, 4:3-ish 412, portrait 282.

### val (official)

- Images: 1547 (1547 .jpg)
- Readable: 1547. Corrupt: 0.
- Modes: 1547 RGB. Grayscale: 0. Alpha: 0.
- Labels: 1547 matched, 0 missing, 0 empty, 9630 boxes, 0 malformed lines.
- Most common sizes: 3840×2160 × 1030, 1920×1080 × 258, 5456×3632 × 149, 3632×5456 × 47, 1231×933 × 20
- Aspect buckets: 16:9-ish 1437, 4:3-ish 63, portrait 47.

Orphan label files (no image): 0.

Excluded 1 image(s) that sit outside the official split folders. They are not given a random split.

- `/Users/manasvialam/Documents/CMU/Semester_1/Deep learning/Project/Midterm/seadronessee/seadronessee_banner.jpg` (no label; 2068×1036 RGB JPEG)

## ntic — class 1 urban

Root: `/Users/manasvialam/Documents/CMU/Semester_1/Deep learning/Project/Midterm/NTIC_human_dataset_70_percent`

Split policy: official folders when present; otherwise a seeded 70/20/10 split.

### Folder structure

```text
NTIC_human_dataset_70_percent/
├── 70_percent_test/
│   ├── images/
│   │   └── 365 .jpg
│   └── labels/
│       └── 365 .txt
├── 70_percent_train/
│   ├── images/
│   │   └── 1277 .jpg
│   └── labels/
│       └── 1277 .txt
└── 70_percent_val/
    ├── images/
    │   └── 182 .jpg
    └── labels/
        └── 182 .txt
```

### test (official)

- Images: 365 (365 .jpg)
- Readable: 365. Corrupt: 0.
- Modes: 365 RGB. Grayscale: 0. Alpha: 0.
- Labels: 365 matched, 0 missing, 0 empty, 2755 boxes, 0 malformed lines.
- Most common sizes: 1920×1080 × 365
- Aspect buckets: 16:9-ish 365.

### train (official)

- Images: 1277 (1277 .jpg)
- Readable: 1277. Corrupt: 0.
- Modes: 1277 RGB. Grayscale: 0. Alpha: 0.
- Labels: 1277 matched, 0 missing, 0 empty, 9666 boxes, 0 malformed lines.
- Most common sizes: 1920×1080 × 1277
- Aspect buckets: 16:9-ish 1277.

### val (official)

- Images: 182 (182 .jpg)
- Readable: 182. Corrupt: 0.
- Modes: 182 RGB. Grayscale: 0. Alpha: 0.
- Labels: 182 matched, 0 missing, 0 empty, 1375 boxes, 0 malformed lines.
- Most common sizes: 1920×1080 × 182
- Aspect buckets: 16:9-ish 182.

Orphan label files (no image): 0.

## heridal — class 2 vegetation

Root: `/Users/manasvialam/Documents/CMU/Semester_1/Deep learning/Project/Midterm/Heridal Human Detection.v1i.yolov5pytorch`

Split policy: official folders when present; otherwise a seeded 70/20/10 split.

### Folder structure

```text
Heridal Human Detection.v1i.yolov5pytorch/
├── 1 .txt, 1 .yaml
├── test/
│   ├── images/
│   │   └── 157 .jpg
│   └── labels/
│       └── 157 .txt
├── train/
│   ├── images/
│   │   └── 1106 .jpg
│   └── labels/
│       └── 1106 .txt
└── valid/
    ├── images/
    │   └── 316 .jpg
    └── labels/
        └── 316 .txt
```

Non-image files skipped by the gate index: 1 `.txt`, 1 `.yaml`.

### test (official)

- Images: 157 (157 .jpg)
- Readable: 157. Corrupt: 0.
- Modes: 157 RGB. Grayscale: 0. Alpha: 0.
- Labels: 157 matched, 0 missing, 45 empty, 970 boxes, 0 malformed lines.
- Most common sizes: 4000×3000 × 157
- Aspect buckets: 4:3-ish 157.

### train (official)

- Images: 1106 (1106 .jpg)
- Readable: 1106. Corrupt: 0.
- Modes: 1106 RGB. Grayscale: 0. Alpha: 0.
- Labels: 1106 matched, 0 missing, 431 empty, 1509 boxes, 0 malformed lines.
- Most common sizes: 4000×3000 × 1106
- Aspect buckets: 4:3-ish 1106.

### val (official)

- Images: 316 (316 .jpg)
- Readable: 316. Corrupt: 0.
- Modes: 316 RGB. Grayscale: 0. Alpha: 0.
- Labels: 316 matched, 0 missing, 85 empty, 594 boxes, 0 malformed lines.
- Most common sizes: 4000×3000 × 312, 4000×2250 × 4
- Aspect buckets: 4:3-ish 312, 16:9-ish 4.

Orphan label files (no image): 0.

## What was not done

No gate training was started. Review this report, then run `python train_gate.py`.


## Split update (applied after audit)

| Class | Train | Val | Test |
|---|---|---|---|
| maritime (SeaDronesSee) | 8930 | 773 | 774 |
| urban (NTIC) | 1277 | 182 | 365 |
| vegetation (Heridal) | 1106 | 316 | 157 |

- SeaDronesSee val is split by frame ID with no shuffling: IDs 48–8649 are val, IDs 8650–16787 are test. Only 223 of the 1,547 original val frames have a train frame at an adjacent ID.
- NTIC leakage: all three official splits draw from the same 7 videos, so its val/test scores are optimistic.
- Heridal: no source image appears in more than one split (1,579 unique bases).
- Gate inputs are no longer letterboxed, since padding exposed aspect ratio (4:3 = Heridal only). All splits use short-side resize + square crop.
