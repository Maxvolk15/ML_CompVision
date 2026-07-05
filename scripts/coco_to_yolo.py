"""Конвертация COCO -> формат Ultralytics YOLO для 5 классов.

Читает classes.coco_mapping из конфига, фильтрует и ремаппит категории,
пишет метки labels/<split>/*.txt (нормированные cx cy w h) и перегенерирует
configs/coco_yolo.yaml с абсолютным путём.

Запуск: python scripts/coco_to_yolo.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def cat2label(mapping):
    return {int(cid): int(idx) for idx, ids in mapping.items() for cid in ids}


def convert_split(images_dir: Path, ann_file: Path, c2l: dict) -> int:
    if not ann_file.exists():
        print(f"пропуск: нет {ann_file}")
        return 0
    coco = json.loads(ann_file.read_text(encoding="utf-8"))
    images = {im["id"]: im for im in coco["images"]}
    labels_dir = Path(str(images_dir).replace("images", "labels"))
    labels_dir.mkdir(parents=True, exist_ok=True)

    lines: dict[int, list[str]] = {}
    for a in coco["annotations"]:
        if a.get("iscrowd", 0) or a["category_id"] not in c2l:
            continue
        im = images[a["image_id"]]
        w, h = im["width"], im["height"]
        x, y, bw, bh = a["bbox"]
        if bw <= 0 or bh <= 0:
            continue
        cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
        line = f"{c2l[a['category_id']]} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}"
        lines.setdefault(a["image_id"], []).append(line)

    for img_id, rows in lines.items():
        stem = Path(images[img_id]["file_name"]).stem
        (labels_dir / f"{stem}.txt").write_text("\n".join(rows), encoding="utf-8")
    print(f"{ann_file.name}: метки для {len(lines)} изображений -> {labels_dir}")
    return len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    classes = cfg["classes"]
    c2l = cat2label(classes["coco_mapping"])
    data = cfg["data"]

    convert_split(ROOT / data["train_images"], ROOT / data["train_ann"], c2l)
    convert_split(ROOT / data["val_images"], ROOT / data["val_ann"], c2l)

    names = classes["names"]
    yolo_yaml = {
        "path": str((ROOT / data["root"]).resolve()),
        "train": "images/train2017",
        "val": "images/val2017",
        "nc": len(names),
        "names": {i: n for i, n in enumerate(names)},
    }
    out = ROOT / "configs" / "coco_yolo.yaml"
    out.write_text(yaml.safe_dump(yolo_yaml, allow_unicode=True, sort_keys=False),
                   encoding="utf-8")
    print(f"обновлён {out}\nТеперь: python main.py train --model yolo")


if __name__ == "__main__":
    main()
