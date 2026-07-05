from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Справочные значения COCO из литературы (ориентир): model, mAP, mAP50, params(M), fps
REFERENCE = [
    ("YOLOv8m", 50.2, 67.2, 25.9, 150),
    ("Faster R-CNN", 37.9, 58.1, 41.5, 25),
    ("SSD300", 25.1, 43.1, 35.6, 60),
    ("EfficientDet-D0", 34.6, 53.0, 3.9, 90),
    ("DETR-R50", 42.0, 62.4, 41.0, 28),
]
DISPLAY = {"yolo": "YOLOv8m", "faster_rcnn": "Faster R-CNN", "ssd": "SSD300",
           "efficientdet": "EfficientDet-D0", "detr": "DETR-R50"}
CLASS_NAME = {"yolo": "YOLODetector", "faster_rcnn": "FasterRCNNDetector", "ssd": "SSDDetector",
              "efficientdet": "EfficientDetDetector", "detr": "DETRDetector"}


def reference_table(cfg: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame(REFERENCE, columns=["model", "mAP", "mAP_50", "params_M", "fps"])
    out = Path(cfg["project"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "metrics.csv", index=False)
    return df


def load_metrics(cfg: Dict[str, Any]) -> pd.DataFrame:
    path = Path(cfg["project"]["output_dir"]) / "metrics.csv"
    return pd.read_csv(path) if path.exists() else reference_table(cfg)


def build_comparison(cfg: Dict[str, Any]) -> pd.DataFrame:
    logs = Path(cfg["project"]["output_dir"]) / "logs"
    rows = []
    for name in cfg["compare"]["models"]:
        f = logs / f"{CLASS_NAME.get(name, name)}_history.json"
        if not f.exists():
            continue
        best = max(json.loads(f.read_text(encoding="utf-8")), key=lambda r: r.get("mAP", 0))
        rows.append({"model": DISPLAY.get(name, name),
                     "mAP": round(best["mAP"] * 100, 1), "mAP_50": round(best["mAP_50"] * 100, 1)})
    df = pd.DataFrame(rows) if rows else reference_table(cfg)
    df.to_csv(Path(cfg["project"]["output_dir"]) / "metrics.csv", index=False)
    return df


def plot_all(df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    plots = Path(cfg["project"]["output_dir"]) / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    _map_bar(df, plots / "map_comparison.png")
    if "fps" in df.columns:
        _speed(df, plots / "speed_vs_accuracy.png")
    _curves(cfg, plots / "training_curves.png")
    plot_experiments(cfg)


def plot_experiments(cfg: Dict[str, Any]) -> None:
    """mAP по прогонам, если есть results/experiments.csv; иначе — план экспериментов."""
    plots = Path(cfg["project"]["output_dir"]) / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    csv = Path(cfg["project"]["output_dir"]) / "experiments.csv"
    df = pd.read_csv(csv) if csv.exists() else None
    fig, ax = plt.subplots(figsize=(9, 5))
    if df is not None and "mAP" in df.columns and df["mAP"].notna().any():
        df = df.dropna(subset=["mAP"])
        colors = ["#2c7fb8" if m == "yolo" else "#d95f0e" for m in df["model"]]
        x = range(len(df))
        ax.bar(x, df["mAP"] * 100, color=colors)
        ax.set_ylabel("mAP@0.5:0.95, %")
        ax.set_title("Эксперименты: YOLOv8 (синий) и Faster R-CNN (оранжевый)")
        ax.set_xticks(list(x))
        ax.set_xticklabels(df["run"], rotation=25, ha="right")
    else:
        ax.axis("off")
        rows = [[m, r.get("name", ""), r.get("img_size", "-"), r.get("lr", "-"),
                 r.get("optimizer", "-"), r.get("epochs", "-")]
                for m, runs in cfg.get("experiments", {}).items() for r in runs]
        t = ax.table(cellText=rows, loc="center", cellLoc="center",
                     colLabels=["Модель", "Прогон", "img_size", "lr", "optimizer", "epochs"])
        t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 1.6)
        ax.set_title("План экспериментов (mAP заполняется после python main.py experiment)")
    fig.tight_layout(); fig.savefig(plots / "experiments_plan.png", dpi=150); plt.close(fig)


def _map_bar(df, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x, w = np.arange(len(df)), 0.38
    ax.bar(x - w / 2, df["mAP"], w, label="mAP@0.5:0.95", color="#2c7fb8")
    if "mAP_50" in df.columns:
        ax.bar(x + w / 2, df["mAP_50"], w, label="mAP@0.5", color="#7fcdbb")
    ax.set_xticks(x); ax.set_xticklabels(df["model"], rotation=20, ha="right")
    ax.set_ylabel("mAP, %"); ax.set_title("Сравнение точности моделей (COCO, ориентир)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _speed(df, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["fps"], df["mAP"], s=120, color="#d95f0e", zorder=3)
    for _, r in df.iterrows():
        ax.annotate(r["model"], (r["fps"], r["mAP"]), textcoords="offset points",
                    xytext=(8, 4), fontsize=9)
    ax.set_xlabel("Скорость, FPS"); ax.set_ylabel("mAP@0.5:0.95, %")
    ax.set_title("Компромисс «скорость–точность»"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _curves(cfg, path):
    logs = Path(cfg["project"]["output_dir"]) / "logs"
    histories = sorted(logs.glob("*_history.json")) if logs.exists() else []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    if histories:
        for h in histories:
            data = json.loads(h.read_text(encoding="utf-8"))
            ep = [d["epoch"] for d in data]
            label = h.stem.replace("_history", "")
            ax1.plot(ep, [d["train_loss"] for d in data], label=label)
            ax2.plot(ep, [d["mAP"] * 100 for d in data], label=label)
        suffix = "(эмпирические данные)"
    else:
        ep = np.arange(50)
        rng = np.random.default_rng(42)
        for name, final in {"YOLOv8m": 50.2, "DETR-R50": 42.0, "Faster R-CNN": 37.9,
                            "EfficientDet-D0": 34.6, "SSD300": 25.1}.items():
            ax1.plot(ep, 4.0 * np.exp(-ep / 12) + 0.4 + rng.normal(0, 0.03, len(ep)), label=name)
            rate = 18 if "DETR" in name else 9
            ax2.plot(ep, final * (1 - np.exp(-ep / rate)) + rng.normal(0, 0.4, len(ep)), label=name)
        suffix = "(иллюстрация)"
    ax1.set_xlabel("Эпоха"); ax1.set_ylabel("Train loss"); ax1.set_title(f"Потери {suffix}")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax2.set_xlabel("Эпоха"); ax2.set_ylabel("mAP@0.5:0.95, %"); ax2.set_title(f"Метрика {suffix}")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
