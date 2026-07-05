from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    _TORCH = True
except ImportError:
    _TORCH = False
    Dataset = object

from src.utils.utils import xywh_to_xyxy


def build_transforms(cfg: Dict[str, Any], train: bool) -> Callable:
    img_size = cfg["preprocess"]["img_size"]
    mean = cfg["preprocess"]["normalize"]["mean"]
    std = cfg["preprocess"]["normalize"]["std"]
    aug = cfg["preprocess"]["augmentation"]
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        tfs: List[Any] = [A.LongestMaxSize(max_size=img_size),
                          A.PadIfNeeded(img_size, img_size, border_mode=0, value=0)]
        if train and aug.get("enabled", True):
            tfs += [A.HorizontalFlip(p=aug.get("horizontal_flip", 0.5)),
                    A.RandomBrightnessContrast(p=aug.get("color_jitter", 0.3)),
                    A.HueSaturationValue(p=aug.get("color_jitter", 0.3))]
        tfs += [A.Normalize(mean=mean, std=std), ToTensorV2()]
        return A.Compose(tfs, bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["labels"], min_visibility=0.2))
    except ImportError:
        return _FallbackTransform(img_size, mean, std)


class _FallbackTransform:
    """resize + normalize без albumentations."""

    def __init__(self, img_size, mean, std):
        self.img_size = img_size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, bboxes, labels):
        import cv2
        h, w = image.shape[:2]
        scale = self.img_size / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
        image = ((image.astype(np.float32) / 255.0) - self.mean) / self.std
        if _TORCH:
            image = torch.from_numpy(image).permute(2, 0, 1)
        bboxes = [[c * scale for c in b] for b in bboxes]
        return {"image": image, "bboxes": bboxes, "labels": labels}


class DetectionDataset(Dataset):
    """Датасет COCO-формата с фильтрацией и ремаппингом категорий в пользовательские классы."""

    def __init__(self, images_dir, ann_file, transforms=None, subset=None,
                 coco_mapping=None, class_names=None):
        if not _TORCH:
            raise RuntimeError("PyTorch не установлен.")
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        self.class_names = class_names or []
        self.cat2label = self._cat2label(coco_mapping)
        self._load(ann_file)
        if subset:
            self.ids = self.ids[:int(subset)]

    @staticmethod
    def _cat2label(coco_mapping):
        if not coco_mapping:
            return None
        return {int(cid): int(idx) for idx, ids in coco_mapping.items() for cid in ids}

    def _load(self, ann_file):
        with open(ann_file, "r", encoding="utf-8") as f:
            coco = json.load(f)
        self.images = {img["id"]: img for img in coco["images"]}
        if self.cat2label is None:
            self.cat2label = {c["id"]: i for i, c in enumerate(sorted(
                coco["categories"], key=lambda c: c["id"]))}
        self.anns: Dict[int, List[dict]] = {}
        for a in coco["annotations"]:
            if a["category_id"] in self.cat2label:
                self.anns.setdefault(a["image_id"], []).append(a)
        self.ids = [i for i in self.images if i in self.anns]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        import cv2
        # пропускаем отсутствующие/битые файлы: берём следующее валидное изображение
        n = len(self.ids)
        img = None
        for k in range(n):
            img_id = self.ids[(idx + k) % n]
            img = cv2.imread(str(self.images_dir / self.images[img_id]["file_name"]))
            if img is not None:
                break
        if img is None:
            raise RuntimeError("Не удалось прочитать ни одного изображения — проверьте data/raw/coco/images")
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes, labels = [], []
        for a in self.anns[img_id]:
            if a.get("iscrowd", 0):
                continue
            x1, y1, x2, y2 = xywh_to_xyxy(a["bbox"])
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat2label[a["category_id"]])
        if self.transforms is not None:
            out = self.transforms(image=image, bboxes=boxes, labels=labels)
            image, boxes, labels = out["image"], out["bboxes"], out["labels"]
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        return image, {
            "boxes": boxes_t,
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
            "area": (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])
                    if len(boxes) else torch.zeros(0),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }


def collate_fn(batch):
    return tuple(zip(*batch))


def build_dataloaders(cfg: Dict[str, Any]) -> Dict[str, "DataLoader"]:
    data, classes = cfg["data"], cfg.get("classes", {})
    mapping, names = classes.get("coco_mapping"), classes.get("names")
    train_ds = DetectionDataset(data["train_images"], data["train_ann"],
                                build_transforms(cfg, True), data.get("subset"), mapping, names)
    val_ds = DetectionDataset(data["val_images"], data["val_ann"],
                              build_transforms(cfg, False), data.get("subset"), mapping, names)
    bs, nw = cfg["training"]["batch_size"], cfg["project"]["num_workers"]
    return {
        "train": DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                            collate_fn=collate_fn, pin_memory=True),
        "val": DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw,
                          collate_fn=collate_fn, pin_memory=True),
    }
