"""One-time: write a short-side-320 JPEG of every gate image to gate_cache/.

Originals are untouched. Re-running skips files already cached.

Usage:
    python build_gate_cache.py            # all splits, all CPU cores
    python build_gate_cache.py --workers 4
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from moe.cache import CACHE_QUALITY, cache_path_for, downscale
from moe.data import build_index
from moe.utils import load_config


def _one(path_str: str) -> str | None:
    path = Path(path_str)
    target = cache_path_for(path)
    if target.is_file():
        return None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as image:
            small = downscale(image)
        tmp = target.with_suffix(".tmp.jpg")
        small.save(tmp, format="JPEG", quality=CACHE_QUALITY)
        os.replace(tmp, target)
        return None
    except Exception as exc:  # report and keep going
        return f"{path}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()
    records, _ = build_index(load_config(args.config))
    paths = [str(r.path) for r in records]
    print(f"Caching {len(paths)} images with {args.workers} workers")
    errors = []
    with ProcessPoolExecutor(args.workers) as pool:
        for result in tqdm(pool.map(_one, paths, chunksize=16), total=len(paths)):
            if result:
                errors.append(result)
    print(f"Done. {len(errors)} errors")
    for line in errors[:20]:
        print("  ", line)


if __name__ == "__main__":
    main()
