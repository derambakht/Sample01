"""نگهداری فهرست منابع و کلیپ‌ها به‌صورت فایل JSONL داخل پوشه‌ی data/.

- data/sources.jsonl : هر خط یک منبع (ویدیو، فایل ایران‌صدا، ضبط محلی)
- data/manifest.jsonl: هر خط یک کلیپ کوتاه صوتی با متن و وضعیت بازبینی
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Iterable

from .config import data_dir

STATUSES = ("pending", "approved", "rejected")
_LOCK = threading.Lock()


@dataclass
class Source:
    source_id: str
    url: str = ""
    title: str = ""
    kind: str = ""          # youtube | iranseda | local | recording | url
    query: str = ""         # عبارت جستجویی که این منبع را پیدا کرد
    city: str = ""
    audio_path: str = ""    # wav ۱۶ کیلوهرتز، نسبت به data/
    subs_path: str = ""     # فایل vtt زیرنویس (اگر بود)
    subs_kind: str = ""     # manual | auto | ""
    duration: float = 0.0
    segmented: bool = False


@dataclass
class Clip:
    id: str
    source_id: str
    audio_path: str             # نسبت به data/
    start: float = 0.0
    end: float = 0.0
    duration: float = 0.0
    text: str = ""              # متن نهایی (قابل ویرایش در پنل)
    auto_text: str = ""         # متن اولیه از زیرنویس یا Whisper، برای مقایسه
    text_source: str = ""       # subtitle_manual | subtitle_auto | whisper:<model> | manual
    status: str = "pending"     # pending | approved | rejected
    split: str = ""             # train | test | ""
    city: str = ""
    notes: str = ""
    extra: dict = field(default_factory=dict)


def _read(path: Path, cls):
    if not path.exists():
        return []
    names = {f.name for f in fields(cls)}
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                out.append(cls(**{k: v for k, v in d.items() if k in names}))
    return out


def _write(path: Path, items: Iterable) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
    os.replace(tmp, path)  # جایگزینی اتمی؛ اگر وسط نوشتن قطع شد فایل خراب نمی‌شود


def sources_path() -> Path:
    return data_dir() / "sources.jsonl"


def manifest_path() -> Path:
    return data_dir() / "manifest.jsonl"


def load_sources() -> list[Source]:
    return _read(sources_path(), Source)


def save_sources(items: list[Source]) -> None:
    with _LOCK:
        _write(sources_path(), items)


def upsert_source(src: Source) -> None:
    with _LOCK:
        items = [s for s in _read(sources_path(), Source) if s.source_id != src.source_id]
        items.append(src)
        _write(sources_path(), items)


def load_clips() -> list[Clip]:
    return _read(manifest_path(), Clip)


def save_clips(items: list[Clip]) -> None:
    with _LOCK:
        _write(manifest_path(), items)


def add_clips(new: list[Clip]) -> None:
    with _LOCK:
        items = _read(manifest_path(), Clip)
        existing = {c.id for c in items}
        items.extend(c for c in new if c.id not in existing)
        _write(manifest_path(), items)


def update_clip(clip_id: str, **changes) -> Clip | None:
    with _LOCK:
        items = _read(manifest_path(), Clip)
        found = None
        for c in items:
            if c.id == clip_id:
                for k, v in changes.items():
                    setattr(c, k, v)
                found = c
                break
        if found:
            _write(manifest_path(), items)
        return found


def abs_audio(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else data_dir() / p
