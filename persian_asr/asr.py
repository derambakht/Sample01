"""بارگذاری مدل Whisper (پایه یا فاین‌تیون‌شده) و تبدیل صدا به متن به‌صورت دسته‌ای.

برچسب‌گذاری، ارزیابی و پنل همه از همین تابع‌ها استفاده می‌کنند تا نتیجه‌ها قابل مقایسه باشند.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import project_path, resolve_model_id


def device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve(model: str) -> str:
    """کلید models.yaml، مسیر محلی (مثلاً outputs/.../final) یا شناسه‌ی HuggingFace."""
    local = project_path(model)
    if local.exists():
        return str(local)
    return resolve_model_id(model)


@lru_cache(maxsize=3)
def load_model(model: str):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    path = _resolve(model)
    dev = device()
    dtype = torch.float16 if dev == "cuda" else torch.float32

    adapter = Path(path) / "adapter_config.json"
    if adapter.exists():  # خروجی LoRA بدون ادغام
        import json

        from peft import PeftModel

        base = json.loads(adapter.read_text())["base_model_name_or_path"]
        m = WhisperForConditionalGeneration.from_pretrained(base, dtype=dtype)
        m = PeftModel.from_pretrained(m, path).merge_and_unload()
    else:
        m = WhisperForConditionalGeneration.from_pretrained(path, dtype=dtype)
    try:
        processor = WhisperProcessor.from_pretrained(path)
    except Exception:
        processor = WhisperProcessor.from_pretrained(resolve_model_id("whisper-small"))
    m.to(dev).eval()
    return m, processor


def transcribe(model: str, wavs: list[np.ndarray], batch_size: int = 8, language: str = "persian",
               max_new_tokens: int = 225, num_beams: int = 1, progress=None) -> list[str]:
    """فهرستی از آرایه‌های صوتی ۱۶k را به متن تبدیل می‌کند."""
    import torch

    m, processor = load_model(model)
    dev = next(m.parameters()).device
    dtype = next(m.parameters()).dtype
    out: list[str] = []
    for i in range(0, len(wavs), batch_size):
        batch = [w[: 30 * 16000] for w in wavs[i: i + batch_size]]
        feats = processor.feature_extractor(batch, sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            ids = m.generate(feats.to(dev, dtype), language=language, task="transcribe",
                             max_new_tokens=max_new_tokens, num_beams=num_beams)
        out += [t.strip() for t in processor.batch_decode(ids, skip_special_tokens=True)]
        if progress:
            progress(min(i + batch_size, len(wavs)), len(wavs))
    return out
