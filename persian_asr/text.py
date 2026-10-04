"""نرمال‌سازی متن فارسی.

دو سطح داریم:
- `clean_transcript`: برای متن آموزشی؛ فقط حروف عربی/فارسی را یکسان و فاصله‌ها را مرتب می‌کند.
- `normalize_for_wer`: برای محاسبه‌ی WER/CER؛ علائم نگارشی، اعراب و نیم‌فاصله را هم حذف می‌کند
  تا اختلاف‌های املایی بی‌اهمیت (مثل «ي» و «ی») خطا حساب نشوند.
"""

from __future__ import annotations

import re

_CHAR_MAP = str.maketrans({
    "ي": "ی", "ى": "ی", "ئ": "ی",
    "ك": "ک",
    "ة": "ه", "ۀ": "ه",
    "أ": "ا", "إ": "ا", "ٱ": "ا",
    "ؤ": "و",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
})

_DIACRITICS = re.compile(r"[ً-ٰٟـ]")  # اعراب، تنوین، تشدید، کشیده
_ZW = re.compile(r"[‌‍‎‏­﻿]")
_PUNCT = re.compile(r"[^\w\s]|_", re.UNICODE)
_SPACES = re.compile(r"\s+")
_BRACKETS = re.compile(r"\[[^\]]*\]|\([^)]*موسیقی[^)]*\)|♪[^♪]*♪?")


def clean_transcript(text: str) -> str:
    """یکسان‌سازی حروف و حذف برچسب‌هایی مثل [موسیقی] برای متن آموزشی."""
    if not text:
        return ""
    text = _BRACKETS.sub(" ", text)
    text = text.translate(_CHAR_MAP)
    text = _DIACRITICS.sub("", text)
    text = text.replace("‍", "").replace("﻿", "")
    return _SPACES.sub(" ", text).strip()


def normalize_for_wer(text: str) -> str:
    """نرمال‌سازی سخت‌گیرانه برای مقایسه‌ی منصفانه‌ی خروجی مدل با متن مرجع."""
    text = clean_transcript(text)
    text = _ZW.sub(" ", text)  # «می‌خوام» و «می خوام» یکسان شوند
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip().lower()


def wer_cer(references: list[str], predictions: list[str]) -> dict[str, float]:
    """WER (خطای کلمه) و CER (خطای حرف، بدون فاصله) را بعد از نرمال‌سازی برمی‌گرداند."""
    import jiwer

    refs, hyps = [], []
    for r, h in zip(references, predictions):
        r = normalize_for_wer(r)
        if r:  # مرجع خالی معنی ندارد
            refs.append(r)
            hyps.append(normalize_for_wer(h))
    if not refs:
        return {"wer": float("nan"), "cer": float("nan"), "n": 0}
    return {
        "wer": jiwer.wer(refs, hyps),
        "cer": jiwer.cer([r.replace(" ", "") for r in refs], [h.replace(" ", "") for h in hyps]),
        "n": len(refs),
    }
