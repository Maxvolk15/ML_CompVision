from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from src.utils.utils import ensure_dir, get_device, get_logger


class Trainer:
    """Обучение torchvision-детекторов: optimizer, scheduler+warmup, AMP, early stopping."""

    def __init__(self, model, dataloaders: Dict[str, Any], cfg: Dict[str, Any]):
        self.cfg = cfg
        self.tr = cfg["training"]
        self.device = get_device(cfg["project"]["device"])
        self.model = model.to(self.device)
        self.loaders = dataloaders
        self.logger = get_logger(log_file=Path(cfg["project"]["output_dir"]) / "logs" / "train.log")
        self.logger.info(f"Устройство: {self.device}")
        self.optimizer = self._optimizer()
        self.scheduler = self._scheduler()
        amp_on = bool(self.tr.get("amp", False)) and self.device.type == "cuda"
        if amp_on and not getattr(self.model, "supports_amp", True):
            amp_on = False
            self.logger.info("AMP отключён для этой модели (fp16 нестабилен) — обучение в fp32")
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp_on)
        self.history: List[Dict[str, float]] = []
        self.ckpt_dir = ensure_dir(self.tr["checkpoint_dir"])

    def _optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.tr["optimizer"].lower() == "sgd":
            return torch.optim.SGD(params, lr=self.tr["lr"], momentum=self.tr["momentum"],
                                   weight_decay=self.tr["weight_decay"])
        return torch.optim.AdamW(params, lr=self.tr["lr"], weight_decay=self.tr["weight_decay"])

    def _scheduler(self):
        sched = self.tr.get("lr_scheduler", "none")
        if sched == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.tr["epochs"])
        if sched == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=max(1, self.tr["epochs"] // 3), gamma=0.1)
        return None

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        running, n = 0.0, 0
        warmup = self.tr.get("warmup_epochs", 0)
        steps = len(self.loaders["train"])
        for step, (images, targets) in enumerate(self.loaders["train"]):
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            if epoch < warmup:
                scale = (epoch * steps + step + 1) / (warmup * steps)
                for g in self.optimizer.param_groups:
                    g["lr"] = self.tr["lr"] * scale

            self.optimizer.zero_grad()
            with torch.amp.autocast(self.device.type, enabled=self.scaler.is_enabled()):
                loss_dict = self.model(images, targets)
                loss = sum(loss_dict.values()) if isinstance(loss_dict, dict) else loss_dict
            self.scaler.scale(loss).backward()
            if self.tr.get("grad_clip"):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tr["grad_clip"])
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running += float(loss)
            n += 1
            if step % self.tr.get("log_interval", 50) == 0:
                self.logger.info(f"epoch {epoch} | step {step}/{steps} | loss {float(loss):.4f} "
                                 f"| lr {self.optimizer.param_groups[0]['lr']:.2e}")
        return running / max(n, 1)

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        from src.evaluation.metrics import compute_map
        self.model.eval()
        preds, targets = [], []
        for images, tgts in self.loaders["val"]:
            images = [img.to(self.device) for img in images]
            preds += [{k: v.cpu() for k, v in p.items()} for p in self.model.predict(images)]
            targets += list(tgts)
        return compute_map(preds, targets)

    def fit(self) -> List[Dict[str, float]]:
        best_map, patience, bad = -1.0, self.tr.get("early_stopping_patience", 10**9), 0
        for epoch in range(self.tr["epochs"]):
            t0 = time.time()
            train_loss = self.train_one_epoch(epoch)
            metrics = self.validate()
            if self.scheduler:
                self.scheduler.step()
            self.history.append({"epoch": epoch, "train_loss": train_loss,
                                 "time_sec": round(time.time() - t0, 1), **metrics})
            self.logger.info(f"[epoch {epoch}] loss={train_loss:.4f} "
                             f"mAP={metrics['mAP']:.4f} mAP50={metrics['mAP_50']:.4f}")
            if metrics["mAP"] > best_map:
                best_map, bad = metrics["mAP"], 0
                torch.save(self.model.state_dict(), self.ckpt_dir / f"{self.model.name}_best.pth")
            else:
                bad += 1
                if bad >= patience:
                    self.logger.info(f"Early stopping на эпохе {epoch}")
                    break
        out = ensure_dir(Path(self.cfg["project"]["output_dir"]) / "logs")
        (out / f"{self.model.name}_history.json").write_text(
            json.dumps(self.history, indent=2, ensure_ascii=False), encoding="utf-8")
        return self.history
