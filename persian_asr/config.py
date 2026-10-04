"""بارگذاری و ذخیره‌ی فایل‌های تنظیمات پوشه‌ی config/."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .config_schema import TRAINING_PARAMS, coerce, default_training_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(os.environ.get("PASR_CONFIG_DIR", PROJECT_ROOT / "config"))


def _load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_dataset_config() -> dict[str, Any]:
    return _load_yaml("dataset.yaml")


def load_sources() -> dict[str, Any]:
    cfg = _load_yaml("sources.yaml")
    for key in ("youtube_searches", "urls", "iranseda_pages", "local_dirs"):
        cfg[key] = cfg.get(key) or []
    return cfg


def load_models() -> dict[str, dict[str, Any]]:
    return _load_yaml("models.yaml")


def resolve_model_id(model: str) -> str:
    """کلید مدل (مثل whisper-small) یا مسیر/شناسه‌ی مستقیم را به شناسه‌ی HuggingFace تبدیل می‌کند."""
    models = load_models()
    if model in models:
        return models[model]["model_id"]
    return model


def load_training_config() -> dict[str, Any]:
    cfg = default_training_config()
    for k, v in _load_yaml("training.yaml").items():
        cfg[k] = coerce(k, v)
    return cfg


def save_training_config(values: dict[str, Any]) -> Path:
    """تنظیمات آموزش را همراه با توضیح فارسی هر پارامتر (به‌صورت کامنت) ذخیره می‌کند."""
    cfg = load_training_config()
    for k, v in values.items():
        cfg[k] = coerce(k, v)

    lines = [
        "# ─────────────────────────────────────────────────────────────",
        "# تنظیمات فاین‌تیون Whisper",
        "# این فایل از پنل هم ویرایش می‌شود؛ توضیح‌ها از persian_asr/config_schema.py می‌آیند.",
        "# ─────────────────────────────────────────────────────────────",
    ]
    group = None
    for p in TRAINING_PARAMS:
        if p.group != group:
            group = p.group
            lines += ["", f"# ── {group} ──"]
        lines.append(f"# {p.label}: {p.help}")
        lines.append(yaml.safe_dump({p.key: cfg[p.key]}, allow_unicode=True).strip())
    path = CONFIG_DIR / "training.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def data_dir() -> Path:
    d = os.environ.get("PASR_DATA_DIR") or load_dataset_config().get("data_dir", "data")
    p = Path(d)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def project_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p
