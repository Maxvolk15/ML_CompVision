"""Инференс обученной модели на изображениях (файл или папка).

Рисует рамки с именами классов и сохраняет результат в results/predictions/.
Для YOLO используется встроенный предикт Ultralytics (с сохранением).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def _list_images(source):
    p = Path(source)
    if p.is_dir():
        return sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT)
    return [p]


def _letterbox(img, size):
    import cv2
    h, w = img.shape[:2]
    s = size / max(h, w)
    resized = cv2.resize(img, (int(round(w * s)), int(round(h * s))))
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[:resized.shape[0], :resized.shape[1]] = resized
    return canvas


def run_inference(cfg, model_name, source, weights=None, score_thr=0.3):
    out = (Path(cfg["project"]["output_dir"]) / "predictions").resolve()
    out.mkdir(parents=True, exist_ok=True)
    names = cfg["classes"]["names"]

    if model_name == "yolo":
        from ultralytics import YOLO
        w = weights or str((Path(cfg["project"]["output_dir"]) / "yolo/weights/best.pt").resolve())
        YOLO(w).predict(source=str(source), conf=score_thr, save=True,
                        project=str(out), name="yolo", exist_ok=True)
        print(f"YOLO: аннотированные изображения -> {out / 'yolo'}")
        return

    import cv2
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from src.models.base import build_model
    from src.utils.utils import get_device

    device = get_device(cfg["project"]["device"])
    model = build_model(model_name, cfg).to(device)
    w = weights or f"{cfg['project']['output_dir']}/checkpoints/{model.name}_best.pth"
    model.load_state_dict(torch.load(w, map_location=device))
    model.eval()

    size = cfg["models"][model_name].get("img_size", cfg["preprocess"]["img_size"])
    mean = np.array(cfg["preprocess"]["normalize"]["mean"], np.float32)
    std = np.array(cfg["preprocess"]["normalize"]["std"], np.float32)

    for path in _list_images(source):
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        canvas = _letterbox(img, size)
        x = (canvas.astype(np.float32) / 255 - mean) / std
        tensor = torch.from_numpy(x).permute(2, 0, 1).float().to(device)
        pred = model.predict([tensor])[0]
        boxes = pred["boxes"].detach().cpu().numpy()
        labels = pred["labels"].detach().cpu().numpy().astype(int)
        scores = pred["scores"].detach().cpu().numpy()
        keep = scores >= score_thr

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(canvas); ax.axis("off")
        for (x1, y1, x2, y2), lab, sc in zip(boxes[keep], labels[keep], scores[keep]):
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                           fill=False, color="lime", lw=2))
            name = names[lab] if 0 <= lab < len(names) else str(lab)
            ax.text(x1, y1 - 3, f"{name} {sc:.2f}", color="black", fontsize=9,
                    bbox=dict(facecolor="lime", alpha=0.7, pad=1, edgecolor="none"))
        dst = out / f"{path.stem}_{model_name}.jpg"
        fig.savefig(dst, bbox_inches="tight", dpi=120); plt.close(fig)
        print(f"{path.name}: найдено {int(keep.sum())} -> {dst}")
