"""CLI проекта детектирования участников дорожного движения.

  python main.py train --model faster_rcnn      # обучить модель
  python main.py eval  --model faster_rcnn --weights <path>
  python main.py compare                         # обучить и сравнить все модели
  python main.py experiment                      # YOLO+Faster с разными параметрами
  python main.py predict --model faster_rcnn --source <path> [--weights <path>] [--conf 0.3] # проверка на изображении/папке
  python main.py plot                            # графики из results/metrics.csv
  python main.py reference                       # справочные графики из литературы
"""
from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

from src.utils.utils import get_logger, load_config, set_seed

logger = get_logger()
MODELS = ["yolo", "faster_rcnn", "ssd", "efficientdet", "detr"]


def _device_banner(cfg):
    import torch
    pref = cfg["project"]["device"]
    avail = torch.cuda.is_available()
    if pref != "cpu" and not avail:
        logger.warning("CUDA НЕДОСТУПНА — обучение пойдёт на CPU (медленно)"
                       "В этом окружении нужна CUDA-сборка torch: "
                       "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
    else:
        dev = "cuda" if (pref != "cpu" and avail) else "cpu"
        logger.info(f"torch {torch.__version__} | CUDA доступна: {avail} | устройство: {dev}")


def _train_and_eval(cfg, name, run_name=None):
    from src.models.base import build_model
    from src.dataset.dataset import build_dataloaders
    from src.training.train import Trainer

    override = cfg["models"][name].get("train")
    if override:
        cfg = copy.deepcopy(cfg)
        cfg["training"].update(override)
    model = build_model(name, cfg)
    if name == "yolo":
        return model.fit("configs/coco_yolo.yaml", name=run_name or "yolo")
    history = Trainer(model, build_dataloaders(cfg), cfg).fit()
    best = max(history, key=lambda r: r.get("mAP", 0)) if history else {}
    return {"mAP": best.get("mAP"), "mAP_50": best.get("mAP_50")}


def cmd_train(args):
    cfg = load_config(args.config)
    _device_banner(cfg)
    set_seed(cfg["project"]["seed"])
    logger.info(f"Обучение: {args.model}")
    _train_and_eval(cfg, args.model)
    logger.info("Готово.")


def cmd_eval(args):
    import torch
    from src.models.base import build_model
    from src.dataset.dataset import build_dataloaders
    from src.evaluation.metrics import compute_map, precision_recall_f1
    from src.utils.utils import get_device

    cfg = load_config(args.config)
    _device_banner(cfg)
    set_seed(cfg["project"]["seed"])
    device = get_device(cfg["project"]["device"])
    model = build_model(args.model, cfg).to(device)
    weights = args.weights or f"{cfg['project']['output_dir']}/checkpoints/{model.name}_best.pth"
    if Path(weights).exists():
        model.load_state_dict(torch.load(weights, map_location=device))
        logger.info(f"Загружены веса: {weights}")
    else:
        logger.warning(f"Веса не найдены ({weights}) — модель НЕ обучена, метрики будут ~0. "
                       f"Сначала обучите: python main.py train --model {args.model}")
    preds, targets = [], []
    for images, tgts in build_dataloaders(cfg)["val"]:
        images = [im.to(device) for im in images]
        preds += model.predict(images)
        targets += list(tgts)
    m = compute_map(preds, targets)
    prf = precision_recall_f1(preds, targets)
    logger.info(f"{args.model}: mAP={m['mAP']:.4f} mAP50={m['mAP_50']:.4f} "
                f"P={prf['precision']:.3f} R={prf['recall']:.3f} F1={prf['f1']:.3f}")


def cmd_compare(args):
    cfg = load_config(args.config)
    _device_banner(cfg)
    for name in cfg["compare"]["models"]:
        logger.info(f"--- {name} ---")
        try:
            set_seed(cfg["project"]["seed"])
            _train_and_eval(cfg, name)
        except Exception as e:
            logger.error(f"{name} пропущена: {e}")
    from src.evaluation.compare import build_comparison, plot_all
    plot_all(build_comparison(cfg), cfg)


def cmd_experiment(args):
    cfg = load_config(args.config)
    _device_banner(cfg)
    experiments = cfg.get("experiments", {})
    only = [args.model] if args.model else list(experiments.keys())
    rows = []
    for name in only:
        for run in experiments.get(name, []):
            logger.info(f"[эксперимент] {name} :: {run}")
            rc = copy.deepcopy(cfg)
            for k in ("lr", "optimizer", "epochs", "batch_size"):
                if k in run:
                    rc["training"][k] = run[k]
            if "img_size" in run:
                rc["models"][name]["img_size"] = run["img_size"]
                rc["preprocess"]["img_size"] = run["img_size"]
            set_seed(cfg["project"]["seed"])
            try:
                metrics = _train_and_eval(rc, name, run_name=run["name"])
            except Exception as e:
                logger.error(f"Прогон {run['name']} пропущен: {e}")
                metrics = {"mAP": None, "mAP_50": None}
            rows.append({"run": run["name"], "model": name,
                         **{k: run.get(k, "") for k in ("img_size", "lr", "optimizer", "epochs")},
                         **metrics})
    if rows:
        out = Path(cfg["project"]["output_dir"]) / "experiments.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        logger.info(f"Сохранено: {out}")
    from src.evaluation.compare import plot_experiments
    plot_experiments(cfg)


def cmd_predict(args):
    cfg = load_config(args.config)
    from src.inference import run_inference
    run_inference(cfg, args.model, args.source, weights=args.weights,
                  score_thr=args.conf)


def cmd_plot(args):
    cfg = load_config(args.config)
    from src.evaluation.compare import load_metrics, plot_all
    plot_all(load_metrics(cfg), cfg)


def cmd_reference(args):
    cfg = load_config(args.config)
    from src.evaluation.compare import reference_table, plot_all
    plot_all(reference_table(cfg), cfg)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Детектирование участников дорожного движения")
    p.add_argument("--config", default="configs/default.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    pt = sub.add_parser("train"); pt.add_argument("--model", required=True, choices=MODELS)
    pt.set_defaults(func=cmd_train)
    pe = sub.add_parser("eval"); pe.add_argument("--model", required=True, choices=MODELS)
    pe.add_argument("--weights", default=None); pe.set_defaults(func=cmd_eval)
    sub.add_parser("compare").set_defaults(func=cmd_compare)
    pp2 = sub.add_parser("predict", help="инференс на изображении/папке")
    pp2.add_argument("--model", required=True, choices=MODELS)
    pp2.add_argument("--source", required=True, help="путь к изображению или папке")
    pp2.add_argument("--weights", default=None, help="путь к весам (по умолчанию из results/)")
    pp2.add_argument("--conf", type=float, default=0.3, help="порог уверенности")
    pp2.set_defaults(func=cmd_predict)
    px = sub.add_parser("experiment"); px.add_argument("--model", default=None,
                                                       choices=["yolo", "faster_rcnn"])
    px.set_defaults(func=cmd_experiment)
    sub.add_parser("plot").set_defaults(func=cmd_plot)
    sub.add_parser("reference").set_defaults(func=cmd_reference)

    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
