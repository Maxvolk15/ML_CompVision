"""Синтетический датасет COCO-формата для проверки пайплайна до подключения COCO.
Запуск: python scripts/make_sample_data.py --n 40"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]

CATEGORIES = [
    {"id": 1, "name": "person", "color": (220, 60, 60)},
    {"id": 2, "name": "bicycle", "color": (60, 180, 75)},
    {"id": 4, "name": "motorcycle", "color": (255, 165, 0)},
    {"id": 3, "name": "car", "color": (0, 130, 200)},
    {"id": 8, "name": "truck", "color": (145, 30, 180)},
]
IMG_W, IMG_H = 320, 240


def make_split(split: str, n: int, seed: int):
    rng = random.Random(seed)
    img_dir = ROOT / "data" / "raw" / "coco" / "images" / split
    img_dir.mkdir(parents=True, exist_ok=True)

    images, annotations = [], []
    ann_id = 1
    for img_id in range(1, n + 1):
        img = Image.new("RGB", (IMG_W, IMG_H), (210, 210, 210))
        draw = ImageDraw.Draw(img)
        n_obj = rng.randint(1, 4)
        for _ in range(n_obj):
            cat = rng.choice(CATEGORIES)
            w = rng.randint(30, 90); h = rng.randint(30, 90)
            x = rng.randint(0, IMG_W - w); y = rng.randint(0, IMG_H - h)
            draw.rectangle([x, y, x + w, y + h], fill=cat["color"])
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": cat["id"],
                "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0,
            })
            ann_id += 1
        fname = f"{img_id:06d}.jpg"
        img.save(img_dir / fname, quality=90)
        images.append({"id": img_id, "file_name": fname,
                       "width": IMG_W, "height": IMG_H})

    ann = {"images": images, "annotations": annotations, "categories": CATEGORIES}
    ann_dir = ROOT / "data" / "raw" / "coco" / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    out = ann_dir / f"instances_{split}.json"
    out.write_text(json.dumps(ann, ensure_ascii=False), encoding="utf-8")
    print(f"{split}: {len(images)} изображений, {len(annotations)} аннотаций -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="изображений в train")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    make_split("train2017", args.n, args.seed)
    make_split("val2017", max(8, args.n // 4), args.seed + 1)
    print("Готово. Теперь можно проверить пайплайн")


if __name__ == "__main__":
    main()
