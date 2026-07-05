"""Установочный скрипт пакета проекта.

Позволяет установить проект в editable-режиме:
    pip install -e .
после чего модули доступны как `from src.models import build_model`
"""
from setuptools import setup, find_packages

with open("requirements.txt", encoding="utf-8") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="cv-detection-benchmark",
    version="0.1.0",
    description="Сравнительный анализ современных моделей детектирования объектов",
    author="Аралушкин Максим Дмитриевич БВТ2402",
    python_requires=">=3.9",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "cvbench=main:main",
        ],
    },
)
