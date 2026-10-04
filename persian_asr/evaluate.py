"""مقایسه‌ی مدل پایه و مدل فاین‌تیون‌شده روی کلیپ‌های test.

خروجی در reports/ ذخیره می‌شود (JSON خلاصه + CSV مقایسه‌ی جمله‌به‌جمله) و پنل آن را نمایش می‌دهد.
WER = درصد کلمه‌های اشتباه، CER = درصد حرف‌های اشتباه؛ هر دو هرچه کمتر بهتر.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Callable

from . import audio
from .asr import transcribe
from .config import PROJECT_ROOT, load_dataset_config
from .manifest import abs_audio, load_clips
from .text import normalize_for_wer, wer_cer

Log = Callable[[str], None]
REPORTS = PROJECT_ROOT / "reports"


def evaluate(models: dict[str, str], limit: int | None = None, batch_size: int = 8,
             num_beams: int = 1, log: Log = print) -> Path:
    """models: نام نمایشی ← کلید/مسیر مدل. مثلاً {"base": "whisper-small", "finetuned": "outputs/.../final"}"""
    language = load_dataset_config().get("language", "persian")
    clips = [c for c in load_clips() if c.split == "test" and c.text.strip()]
    if limit:
        clips = clips[:limit]
    if not clips:
        raise SystemExit("❌ کلیپ test نداریم. اول دستور `split` را اجرا کن.")
    log(f"🧪 ارزیابی روی {len(clips)} کلیپ test")
    wavs = [audio.load(abs_audio(c.audio_path)) for c in clips]
    refs = [c.text for c in clips]

    results, preds = {}, {}
    for name, model in models.items():
        t0 = time.time()
        log(f"   ▶ {name}: {model}")
        p = transcribe(model, wavs, batch_size=batch_size, language=language, num_beams=num_beams)
        m = wer_cer(refs, p)
        m.update({"model": model, "seconds": round(time.time() - t0, 1)})
        results[name], preds[name] = m, p
        log(f"     WER={m['wer']*100:.1f}%  CER={m['cer']*100:.1f}%")

    names = list(models)
    if len(names) >= 2:
        a, b = results[names[0]], results[names[-1]]
        rel = (a["wer"] - b["wer"]) / a["wer"] * 100 if a["wer"] else 0.0
        log(f"📈 بهبود نسبی WER از «{names[0]}» به «{names[-1]}»: {rel:.1f}%")
        results["_improvement_pct"] = rel

    REPORTS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    summary = {"time": stamp, "n_clips": len(clips), "results": results}
    js = REPORTS / f"eval_{stamp}.json"
    js.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS / "latest.json").write_text(js.read_text(encoding="utf-8"), encoding="utf-8")

    from jiwer import wer as _wer

    with open(REPORTS / f"eval_{stamp}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "reference"] + [f"{n}" for n in names] + [f"wer_{n}" for n in names])
        for i, c in enumerate(clips):
            r = normalize_for_wer(c.text)
            per = [round(_wer(r, normalize_for_wer(preds[n][i]) or "∅") * 100, 1) if r else "" for n in names]
            w.writerow([c.id, c.text] + [preds[n][i] for n in names] + per)
    (REPORTS / "latest.csv").write_bytes((REPORTS / f"eval_{stamp}.csv").read_bytes())
    log(f"📝 گزارش: {js}")
    return js
