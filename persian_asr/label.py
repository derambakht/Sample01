"""تولید متن اولیه (pseudo-label) برای کلیپ‌هایی که متن ندارند، با یک مدل بزرگ Whisper.

⚠️ این متن‌ها خودشان خطا دارند (به‌خصوص روی کلمه‌های لهجه‌ای). اگر بدون اصلاح در آموزش استفاده شوند،
مدل کوچک فقط تقلید مدل بزرگ را یاد می‌گیرد. حتماً حداقل کلیپ‌های بخش test را در پنل اصلاح و تأیید کن.
"""

from __future__ import annotations

from typing import Callable

from . import audio
from .asr import transcribe
from .config import load_dataset_config
from .manifest import abs_audio, load_clips, save_clips
from .text import clean_transcript

Log = Callable[[str], None]


def label_clips(model: str | None = None, overwrite_auto_subs: bool = False, limit: int | None = None,
                batch_size: int | None = None, log: Log = print) -> int:
    cfg = load_dataset_config()
    lab = cfg.get("labeling", {})
    model = model or lab.get("model", "openai/whisper-large-v3")
    batch_size = batch_size or int(lab.get("batch_size", 8))
    language = cfg.get("language", "persian")

    clips = load_clips()
    todo = [c for c in clips if c.status != "rejected" and (
        not c.text.strip() or (overwrite_auto_subs and c.text_source == "subtitle_auto"))]
    if limit:
        todo = todo[:limit]
    if not todo:
        log("✅ همه‌ی کلیپ‌ها متن دارند.")
        return 0
    log(f"🧠 برچسب‌گذاری {len(todo)} کلیپ با {model} ...")

    by_id = {c.id: c for c in clips}
    chunk = batch_size * 8  # هر چند دسته یک بار ذخیره کن تا اگر قطع شد کار از دست نرود
    for i in range(0, len(todo), chunk):
        part = todo[i: i + chunk]
        wavs = [audio.load(abs_audio(c.audio_path)) for c in part]
        texts = transcribe(model, wavs, batch_size=batch_size, language=language)
        for c, t in zip(part, texts):
            t = clean_transcript(t)
            c = by_id[c.id]
            if c.text_source == "subtitle_auto":
                c.extra["subtitle_text"] = c.auto_text
            c.text, c.auto_text, c.text_source = t, t, f"whisper:{model}"
        save_clips(list(by_id.values()))
        log(f"   {min(i + chunk, len(todo))}/{len(todo)}")
    return len(todo)
