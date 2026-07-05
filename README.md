# Детектирование участников дорожного движения

ML-проект по компьютерному зрению: сравнение моделей детекции объектов на пяти
пользовательских классах участников дорожного движения (подмножество COCO).
Сравниваются пять моделей, эксперименты с гиперпараметрами — на YOLOv8 и Faster R-CNN.

## Классы

| Индекс | Класс | Категории COCO |
| 0 | пешеход | `person` |
| 1 | велосипедист | `bicycle` |
| 2 | мотоциклист | `motorcycle` |
| 3 | легковой_авто | `car` |
| 4 | грузовой_авто | `bus`, `truck` |

Маппинг задаётся в `configs/default.yaml` (секция `classes`).

## Модели

YOLOv8 (Ultralytics), Faster R-CNN, SSD300, EfficientDet (RetinaNet), DETR.
Метрики: mAP@0.5:0.95, mAP@0.5, Precision, Recall, F1.

## Структура

```
configs/          default.yaml (все параметры), coco_yolo.yaml (для YOLO)
data/raw/coco/    датасет COCO (кладётся сюда, в git не хранится)
src/
  dataset/        загрузка COCO, фильтрация/ремаппинг классов, аугментации
  models/         5 моделей + общий интерфейс BaseDetector
  training/       цикл обучения
  evaluation/     метрики (mAP, P/R/F1) и графики
  inference.py    инференс на изображениях
  utils/          конфиг, seed, логирование, устройство
scripts/          coco_to_yolo.py, make_sample_data.py
notebooks/        exploration.ipynb (EDA)
results/          metrics.csv, experiments.csv, plots/
main.py           CLI
```

## Установка

```bash
pip install -r requirements.txt
```

С PyPI ставится CPU-сборка PyTorch. Для GPU поставьте CUDA-сборку в то же окружение
(для Python 3.14 — индекс cu128):

```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
```

`project.device: auto` в конфиге сам выберет GPU.

## Данные

Положите COCO в `data/raw/coco/`:

```
data/raw/coco/
├── images/{train2017,val2017}/
└── annotations/instances_{train2017,val2017}.json
```

Для обучения YOLO предварительно сконвертируйте разметку в формат YOLO:

```bash
python scripts/coco_to_yolo.py
```

torchvision-модели (faster_rcnn/ssd/efficientdet/detr) читают COCO-JSON напрямую.
Быстрая проверка без COCO — синтетический датасет: `python scripts/make_sample_data.py`.

## Запуск

```bash
python main.py train --model faster_rcnn        # обучить модель
python main.py eval  --model faster_rcnn        # метрики на валидации
python main.py compare                          # обучить и сравнить все 5 моделей
python main.py experiment                       # YOLO и Faster R-CNN с разными параметрами
python main.py predict --model faster_rcnn --source data/raw/coco/images/val2017
python main.py plot                             # графики из results/metrics.csv
```

Все параметры задаются в `configs/default.yaml`. Веса сохраняются в
`results/checkpoints/` (YOLO — в `results/yolo/weights/`), графики — в `results/plots/`.

## Технологии
Python, PyTorch, torchvision, Ultralytics, transformers, NumPy, matplotlib.
