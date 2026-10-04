"""برش فایل‌های خام به کلیپ‌های کوتاه (۲ تا ۲۰ ثانیه) مناسب آموزش Whisper.

- اگر منبع زیرنویس داشته باشد: مرز کلیپ‌ها از زمان‌بندی زیرنویس و متن اولیه از خود زیرنویس می‌آید.
- اگر نداشته باشد: بازه‌های گفتار با VAD انرژی پیدا می‌شوند و متن بعداً با دستور `label` تولید می‌شود.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import audio
from .config import data_dir, load_dataset_config
from .manifest import (Clip, Source, abs_audio, add_clips, load_clips, load_sources, save_clips,
                       upsert_source)
from .text import clean_transcript

Log = Callable[[str], None]


@dataclass
class Cue:
    start: float
    end: float
    text: str


_TS = re.compile(r"(\d+):(\d{2}):(\d{2})[.,](\d{3})|(\d{2}):(\d{2})[.,](\d{3})")
_TAGS = re.compile(r"<[^>]+>")


def _ts(s: str) -> float:
    m = _TS.search(s)
    if not m:
        raise ValueError(s)
    if m.group(1) is not None:
        h, mi, se, ms = (int(m.group(i)) for i in range(1, 5))
    else:
        h, (mi, se, ms) = 0, (int(m.group(i)) for i in range(5, 8))
    return h * 3600 + mi * 60 + se + ms / 1000


def parse_vtt(path: str | Path) -> list[Cue]:
    """فایل VTT/SRT را می‌خواند. تکرارهای «غلتان» زیرنویس خودکار یوتیوب را هم حذف می‌کند."""
    raw = Path(path).read_text(encoding="utf-8", errors="ignore").replace("\r", "")
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", raw):
        lines = [ln for ln in block.split("\n") if ln.strip()]
        idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if idx is None:
            continue
        a, b = lines[idx].split("-->")[:2]
        try:
            start, end = _ts(a), _ts(b)
        except ValueError:
            continue
        text_lines = [_TAGS.sub("", ln).strip() for ln in lines[idx + 1:]]
        text_lines = [ln for ln in text_lines if ln]
        if text_lines:
            cues.append(Cue(start, end, "\n".join(text_lines)))

    # زیرنویس خودکار یوتیوب: هر cue خط قبلی را تکرار می‌کند و cueهای ۱۰ میلی‌ثانیه‌ای دارد.
    out: list[Cue] = []
    prev_lines: set[str] = set()
    for c in cues:
        if c.end - c.start < 0.05:
            continue
        lines = c.text.split("\n")
        new = [ln for ln in lines if ln not in prev_lines]
        prev_lines = set(lines)
        text = " ".join(new).strip()
        if text:
            out.append(Cue(c.start, c.end, text))
    return out


def group_cues(cues: list[Cue], target_s: float, max_s: float, max_gap_s: float = 1.0) -> list[Cue]:
    """جمله‌های پشت‌سرهم را تا طول هدف به هم می‌چسباند (بدون عبور از سقف طول)."""
    out: list[Cue] = []
    cur: Cue | None = None
    for c in cues:
        if cur and c.start - cur.end <= max_gap_s and c.end - cur.start <= max_s and cur.end - cur.start < target_s:
            cur = Cue(cur.start, max(cur.end, c.end), f"{cur.text} {c.text}")
        else:
            if cur:
                out.append(cur)
            cur = Cue(c.start, c.end, c.text)
    if cur:
        out.append(cur)
    return out


def segment_source(src: Source, method: str | None = None, log: Log = print) -> list[Clip]:
    cfg = load_dataset_config()
    seg = cfg.get("segmentation", {})
    method = method or seg.get("method", "auto")
    min_s, max_s = float(seg.get("min_seconds", 2)), float(seg.get("max_seconds", 20))
    target_s = float(seg.get("target_seconds", 8))
    pad = float(seg.get("padding_ms", 150)) / 1000
    vad_cfg = seg.get("vad", {})
    sr = int(cfg.get("audio", {}).get("sample_rate", audio.SR))

    wav = audio.load(abs_audio(src.audio_path), sr)
    total = len(wav) / sr
    clips_dir = data_dir() / "clips" / src.source_id
    spans: list[tuple[float, float, str, str]] = []  # start, end, text, text_source

    subs = abs_audio(src.subs_path) if src.subs_path else None
    use_subs = subs is not None and subs.exists() and method in ("auto", "subtitles")

    if use_subs and subs.suffix == ".txt":
        # فایل محلی با متن کامل: اگر کوتاه بود کل فایل یک کلیپ است
        text = clean_transcript(subs.read_text(encoding="utf-8"))
        if total <= max_s:
            spans.append((0.0, total, text, "manual"))
        else:
            log(f"⚠️ {src.title}: متن دارد ولی بلندتر از {max_s} ثانیه است؛ با VAD بریده می‌شود و متن باید دستی تقسیم شود")
            use_subs = False
    elif use_subs:
        kind = "subtitle_manual" if src.subs_kind == "manual" else "subtitle_auto"
        for c in group_cues(parse_vtt(subs), target_s, max_s):
            if c.end - c.start > max_s:  # یک cue خیلی طولانی؛ نمی‌شود متنش را دقیق تقسیم کرد
                continue
            spans.append((c.start, c.end, clean_transcript(c.text), kind))

    if not spans:
        if method == "subtitles":
            log(f"⏭️ {src.title}: زیرنویس ندارد")
            return []
        regions = audio.energy_vad(
            wav, sr,
            frame_ms=int(vad_cfg.get("frame_ms", 30)),
            threshold_db=float(vad_cfg.get("threshold_db", 12)),
            min_silence_ms=int(vad_cfg.get("min_silence_ms", 350)),
            min_speech_ms=int(vad_cfg.get("min_speech_ms", 250)),
        )
        pieces: list[tuple[float, float]] = []
        for a, b in regions:
            pieces += audio.split_long(wav, sr, a, b, max_s - 2 * pad)
        spans = [(a, b, "", "") for a, b in audio.group_regions(pieces, target_s, max_s - 2 * pad)]

    clips: list[Clip] = []
    for i, (a, b, text, tsrc) in enumerate(spans):
        a, b = max(0.0, a - pad), min(total, b + pad)
        if b - a < min_s:
            continue
        cid = f"{src.source_id}_{i:04d}"
        path = clips_dir / f"{cid}.wav"
        audio.save(path, wav[int(a * sr): int(b * sr)], sr)
        clips.append(Clip(id=cid, source_id=src.source_id, audio_path=str(path.relative_to(data_dir())),
                          start=round(a, 3), end=round(b, 3), duration=round(b - a, 3),
                          text=text, auto_text=text, text_source=tsrc, city=src.city))
    add_clips(clips)
    src.segmented = True
    upsert_source(src)
    mins = sum(c.duration for c in clips) / 60
    log(f"✂️ {src.title or src.source_id}: {len(clips)} کلیپ ({mins:.1f} دقیقه) — روش: {'زیرنویس' if spans and spans[0][3] else 'VAD'}")
    return clips


def segment_all(method: str | None = None, force: bool = False, log: Log = print) -> int:
    n = 0
    for src in load_sources():
        if src.segmented and not force:
            continue
        if force:  # کلیپ‌های قبلی این منبع (و بازبینی‌شان) پاک می‌شوند
            save_clips([c for c in load_clips() if c.source_id != src.source_id])
        try:
            n += len(segment_source(src, method, log=log))
        except Exception as ex:
            log(f"❌ برش {src.source_id} ناموفق: {ex}")
    log(f"\n✂️ مجموع کلیپ‌های جدید: {n}")
    return n
