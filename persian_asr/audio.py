"""ابزارهای صوتی: تبدیل به wav ۱۶k، خواندن/نوشتن، و تشخیص گفتار مبتنی بر انرژی (VAD)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000


def ensure_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg پیدا نشد. نصبش کن: sudo apt install ffmpeg  یا  brew install ffmpeg  یا  choco install ffmpeg")
    return exe


def to_wav16k(src: str | Path, dst: str | Path, sr: int = SR) -> Path:
    """هر فرمت صوتی/ویدیویی را به wav مونو ۱۶ کیلوهرتز تبدیل می‌کند."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ensure_ffmpeg(), "-y", "-loglevel", "error", "-i", str(src),
           "-vn", "-ac", "1", "-ar", str(sr), "-sample_fmt", "s16", str(dst)]
    subprocess.run(cmd, check=True)
    return dst


def load(path: str | Path, sr: int = SR) -> np.ndarray:
    """فایل صوتی را به آرایه‌ی float32 مونو با نرخ sr می‌خواند (در صورت نیاز با ffmpeg تبدیل می‌کند)."""
    try:
        wav, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        if file_sr == sr:
            return wav
    except Exception:
        pass
    cmd = [ensure_ffmpeg(), "-loglevel", "error", "-i", str(path), "-f", "s16le",
           "-ac", "1", "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def save(path: str | Path, wav: np.ndarray, sr: int = SR) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, sr, subtype="PCM_16")


def duration(path: str | Path) -> float:
    try:
        return float(sf.info(str(path)).duration)
    except Exception:
        return len(load(path)) / SR


def frame_db(wav: np.ndarray, sr: int, frame_ms: int) -> np.ndarray:
    hop = max(1, int(sr * frame_ms / 1000))
    n = len(wav) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    frames = wav[: n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def energy_vad(wav: np.ndarray, sr: int = SR, frame_ms: int = 30, threshold_db: float = 12,
               min_silence_ms: int = 350, min_speech_ms: int = 250) -> list[tuple[float, float]]:
    """بازه‌های گفتار را با مقایسه‌ی انرژی هر فریم با کف نویز پیدا می‌کند.

    کف نویز = صدک ۱۰ انرژی فریم‌ها. ساده و بدون مدل است؛ برای صدای تمیز (مصاحبه/رادیو) خوب کار می‌کند.
    """
    db = frame_db(wav, sr, frame_ms)
    if len(db) == 0:
        return []
    floor = np.percentile(db, 10)
    thr = max(floor + threshold_db, -55.0)
    speech = db > thr

    sec = frame_ms / 1000
    regions: list[list[int]] = []
    i = 0
    while i < len(speech):
        if speech[i]:
            j = i
            while j < len(speech) and speech[j]:
                j += 1
            regions.append([i, j])
            i = j
        else:
            i += 1

    # پر کردن سکوت‌های کوتاه بین دو تکه گفتار
    gap = int(min_silence_ms / frame_ms)
    merged: list[list[int]] = []
    for r in regions:
        if merged and r[0] - merged[-1][1] <= gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)

    min_len = int(min_speech_ms / frame_ms)
    return [(a * sec, b * sec) for a, b in merged if b - a >= min_len]


def split_long(wav: np.ndarray, sr: int, start: float, end: float, max_s: float,
               frame_ms: int = 30) -> list[tuple[float, float]]:
    """بازه‌ی طولانی‌تر از max_s را در کم‌انرژی‌ترین نقطه‌ی میانه‌اش می‌شکند (بازگشتی)."""
    if end - start <= max_s:
        return [(start, end)]
    seg = wav[int(start * sr): int(end * sr)]
    db = frame_db(seg, sr, frame_ms)
    sec = frame_ms / 1000
    lo, hi = int(len(db) * 0.3), int(len(db) * 0.7)
    cut = start + (lo + int(np.argmin(db[lo:hi]))) * sec if hi > lo else (start + end) / 2
    return split_long(wav, sr, start, cut, max_s, frame_ms) + split_long(wav, sr, cut, end, max_s, frame_ms)


def group_regions(regions: list[tuple[float, float]], target_s: float, max_s: float,
                  max_gap_s: float = 1.0) -> list[tuple[float, float]]:
    """تکه‌های گفتار نزدیک به هم را تا رسیدن به طول هدف به یک کلیپ تبدیل می‌کند."""
    out: list[tuple[float, float]] = []
    cur: list[float] | None = None
    for a, b in regions:
        if cur is None:
            cur = [a, b]
        elif a - cur[1] <= max_gap_s and b - cur[0] <= max_s and cur[1] - cur[0] < target_s:
            cur[1] = b
        else:
            out.append((cur[0], cur[1]))
            cur = [a, b]
    if cur is not None:
        out.append((cur[0], cur[1]))
    return out
