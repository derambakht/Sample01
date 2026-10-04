"""تقسیم کلیپ‌ها به train/test و خروجی گرفتن در قالب audiofolder هاگینگ‌فیس.

خروجی: data/export/{train,test}/metadata.csv + فایل‌های wav
این پوشه را می‌شود مستقیم با `load_dataset("audiofolder", data_dir="data/export")` خواند
یا به HuggingFace Hub آپلود کرد.
"""

from __future__ import annotations

import csv
import random
import shutil
from collections import defaultdict
from typing import Callable

from .config import data_dir, load_dataset_config
from .manifest import Clip, abs_audio, load_clips, save_clips
from .text import clean_transcript

Log = Callable[[str], None]


def usable(c: Clip, include_pending: bool) -> bool:
    if not c.text.strip() or c.status == "rejected":
        return False
    return c.status == "approved" or include_pending


def assign_splits(log: Log = print) -> dict[str, int]:
    """کلیپ‌ها را بین train و test تقسیم می‌کند و فیلد split را در مانیفست می‌نویسد.

    test فقط از کلیپ‌های تأییدشده ساخته می‌شود و اگر by_source روشن باشد،
    کل کلیپ‌های یک منبع با هم در یک بخش می‌روند (جلوگیری از نشت گوینده).
    """
    cfg = load_dataset_config().get("split", {})
    ratio = float(cfg.get("test_ratio", 0.2))
    by_source = bool(cfg.get("by_source", True))
    include_pending = bool(cfg.get("include_pending_in_train", False))
    rng = random.Random(int(cfg.get("seed", 42)))

    clips = load_clips()
    for c in clips:
        c.split = ""
    approved = [c for c in clips if usable(c, False)]
    total = sum(c.duration for c in approved)

    groups: dict[str, list[Clip]] = defaultdict(list)
    for c in approved:
        groups[c.source_id if by_source else c.id].append(c)
    keys = sorted(groups)
    if by_source and len(keys) < 3:
        log("⚠️ کمتر از ۳ منبع تأییدشده داریم؛ تقسیم در سطح کلیپ انجام می‌شود (ممکن است گوینده‌ی test در train هم باشد).")
        groups = {c.id: [c] for c in approved}
        keys = sorted(groups)
    rng.shuffle(keys)

    test_dur = 0.0
    for k in keys:
        dur = sum(c.duration for c in groups[k])
        # منبعی که خودش بیش از نصف سهم test است را فقط وقتی برمی‌داریم که test هنوز خالی است
        if test_dur < ratio * total and (test_dur == 0 or test_dur + dur <= ratio * total * 1.5):
            split = "test"
            test_dur += dur
        else:
            split = "train"
        for c in groups[k]:
            c.split = split

    test_sources = {c.source_id for c in clips if c.split == "test"}
    for c in clips:
        if not c.split and usable(c, include_pending):
            # کلیپ pending از منبعی که در test است به train نمی‌رود (نشت گوینده)
            if not (by_source and c.source_id in test_sources):
                c.split = "train"
    save_clips(clips)

    stats = {s: sum(1 for c in clips if c.split == s) for s in ("train", "test")}
    mins = {s: sum(c.duration for c in clips if c.split == s) / 60 for s in ("train", "test")}
    log(f"📊 train: {stats['train']} کلیپ ({mins['train']:.1f} دقیقه) | test: {stats['test']} کلیپ ({mins['test']:.1f} دقیقه)")
    return stats


def export_audiofolder(log: Log = print) -> str:
    out = data_dir() / "export"
    if out.exists():
        shutil.rmtree(out)
    clips = load_clips()
    for split in ("train", "test"):
        rows = [c for c in clips if c.split == split]
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "metadata.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file_name", "transcription", "duration", "source_id", "city", "status"])
            for c in rows:
                src = abs_audio(c.audio_path)
                dst = d / f"{c.id}.wav"
                try:
                    dst.hardlink_to(src)
                except OSError:
                    shutil.copy(src, dst)
                w.writerow([dst.name, clean_transcript(c.text), c.duration, c.source_id, c.city, c.status])
        log(f"📁 {split}: {len(rows)} فایل → {d}")
    return str(out)
