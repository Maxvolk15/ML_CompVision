from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.utils.utils import box_iou


def compute_map(preds: List[Dict], targets: List[Dict], iou_thresholds=None) -> Dict[str, float]:
    """mAP@0.5 и mAP@0.5:0.95 на чистом NumPy (без pycocotools)."""
    if iou_thresholds is None:
        iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]

    classes = set()
    for t in targets:
        classes.update(_np(t["labels"]).astype(int).tolist())
    for p in preds:
        classes.update(_np(p["labels"]).astype(int).tolist())

    ap = {thr: [] for thr in iou_thresholds}
    for cls in sorted(classes):
        n_gt = sum(int((_np(t["labels"]).astype(int) == cls).sum()) for t in targets)
        if n_gt == 0:
            continue
        entries = []
        for i, p in enumerate(preds):
            labels = _np(p["labels"]).astype(int)
            scores = _np(p.get("scores", np.ones(len(labels))))
            boxes = _np(p["boxes"]).reshape(-1, 4)
            for b, s in zip(boxes[labels == cls], scores[labels == cls]):
                entries.append((float(s), i, b))
        entries.sort(key=lambda e: -e[0])

        for thr in iou_thresholds:
            gt = {i: _np(targets[i]["boxes"]).reshape(-1, 4)[_np(targets[i]["labels"]).astype(int) == cls]
                  for i in range(len(targets))}
            matched = {i: np.zeros(len(gt[i]), dtype=bool) for i in gt}
            tp, fp = np.zeros(len(entries)), np.zeros(len(entries))
            for k, (_, i, box) in enumerate(entries):
                best_iou, best_j = 0.0, -1
                for j, gb in enumerate(gt[i]):
                    if matched[i][j]:
                        continue
                    iou = box_iou(box, gb)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= thr and best_j >= 0:
                    tp[k] = 1
                    matched[i][best_j] = True
                else:
                    fp[k] = 1
            ap[thr].append(_ap(tp, fp, n_gt))

    map_50 = float(np.mean(ap[0.5])) if ap[0.5] else 0.0
    means = [np.mean(v) for v in ap.values() if v]
    return {"mAP": float(np.mean(means)) if means else 0.0,
            "mAP_50": map_50,
            "mAP_75": float(np.mean(ap[0.75])) if ap.get(0.75) else 0.0}


def _ap(tp: np.ndarray, fp: np.ndarray, n_gt: int) -> float:
    if len(tp) == 0:
        return 0.0
    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    recall = tp_c / (n_gt + 1e-9)
    precision = tp_c / (tp_c + fp_c + 1e-9)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def precision_recall_f1(preds, targets, iou_threshold=0.5, score_threshold=0.5) -> Dict[str, float]:
    """Агрегированные Precision, Recall, F1 при заданном пороге IoU."""
    tp = fp = fn = 0
    for pred, gt in zip(preds, targets):
        p_boxes = _np(pred["boxes"])
        p_labels = _np(pred["labels"])
        p_scores = _np(pred.get("scores", np.ones(len(p_boxes))))
        keep = p_scores >= score_threshold
        order = np.argsort(-p_scores[keep])
        p_boxes, p_labels = p_boxes[keep][order], p_labels[keep][order]

        g_boxes, g_labels = _np(gt["boxes"]), _np(gt["labels"])
        matched = np.zeros(len(g_boxes), dtype=bool)
        for pb, pl in zip(p_boxes, p_labels):
            best_iou, best_j = 0.0, -1
            for j, (gb, gl) in enumerate(zip(g_boxes, g_labels)):
                if matched[j] or gl != pl:
                    continue
                iou = box_iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_threshold and best_j >= 0:
                tp += 1
                matched[best_j] = True
            else:
                fp += 1
        fn += int((~matched).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
