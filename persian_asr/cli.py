"""خط فرمان پروژه. هر مرحله یک زیر‌دستور است:

    python -m persian_asr.cli download      # دانلود از منابع config/sources.yaml
    python -m persian_asr.cli segment       # برش به کلیپ‌های کوتاه
    python -m persian_asr.cli label         # متن اولیه با Whisper بزرگ برای کلیپ‌های بی‌متن
    python -m persian_asr.cli stats         # آمار دیتاست
    python -m persian_asr.cli split         # تقسیم train/test (بر اساس منبع)
    python -m persian_asr.cli export        # خروجی audiofolder در data/export
    python -m persian_asr.cli train         # فاین‌تیون با config/training.yaml
    python -m persian_asr.cli evaluate      # مقایسه‌ی مدل پایه و فاین‌تیون‌شده
    python -m persian_asr.cli panel         # اجرای پنل وب
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter


def _print(msg: str) -> None:
    print(msg, flush=True)


def cmd_download(a):
    from .download import download_all, import_local

    if a.local:
        for folder in a.local:
            import_local(folder, a.city or "", log=_print)
    download_all(extra_urls=a.url, extra_queries=a.query, max_results=a.max_results, log=_print)


def cmd_segment(a):
    from .segment import segment_all

    segment_all(method=a.method, force=a.force, log=_print)


def cmd_label(a):
    from .label import label_clips

    label_clips(model=a.model, overwrite_auto_subs=a.replace_auto_subs, limit=a.limit,
                batch_size=a.batch_size, log=_print)


def cmd_stats(a):
    from .config import load_dataset_config
    from .manifest import load_clips, load_sources

    cfg = load_dataset_config().get("target", {})
    clips, sources = load_clips(), load_sources()
    st = Counter(c.status for c in clips)
    sp = Counter(c.split or "-" for c in clips)
    ts = Counter((c.text_source or "بدون متن").split(":")[0] for c in clips)
    approved = [c for c in clips if c.status == "approved"]
    print(f"منابع: {len(sources)} ({sum(s.duration for s in sources)/3600:.2f} ساعت صدای خام)")
    print(f"کلیپ‌ها: {len(clips)} ({sum(c.duration for c in clips)/3600:.2f} ساعت)")
    print(f"وضعیت: {dict(st)}")
    print(f"منبع متن: {dict(ts)}")
    print(f"تقسیم: {dict(sp)}")
    print(f"تأییدشده: {len(approved)} / هدف حداقل {cfg.get('min_clips', 300)} "
          f"({sum(c.duration for c in approved)/60:.1f} دقیقه)")


def cmd_split(a):
    from .split import assign_splits, export_audiofolder

    assign_splits(log=_print)
    if a.export:
        export_audiofolder(log=_print)


def cmd_export(a):
    from .split import export_audiofolder

    export_audiofolder(log=_print)


def cmd_train(a):
    from .config_schema import coerce
    from .train import train

    overrides = {}
    for kv in a.set or []:
        k, v = kv.split("=", 1)
        overrides[k] = coerce(k, v)
    if a.model:
        overrides["model"] = a.model
    train(overrides)


def cmd_evaluate(a):
    from .config import load_training_config, project_path
    from .evaluate import evaluate

    cfg = load_training_config()
    models = {"base": a.base or cfg["model"]}
    ft = a.finetuned or str(project_path(cfg["output_dir"]) / "final")
    if project_path(ft).exists():
        models["finetuned"] = ft
    else:
        print(f"⚠️ مدل فاین‌تیون‌شده پیدا نشد ({ft}); فقط مدل پایه ارزیابی می‌شود.")
    evaluate(models, limit=a.limit, batch_size=a.batch_size, num_beams=a.beams, log=_print)


def cmd_panel(a):
    from .panel.app import main

    main(host=a.host, port=a.port, share=a.share)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="persian_asr", description="ساخت دیتاست لهجه‌ی فارسی و فاین‌تیون Whisper")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("download", help="دانلود صدا از یوتیوب/ایران‌صدا/پوشه‌ی محلی")
    s.add_argument("--url", action="append", help="لینک اضافه (چندبار قابل تکرار)")
    s.add_argument("--query", action="append", help="عبارت جستجوی یوتیوب اضافه")
    s.add_argument("--max-results", type=int, help="سقف نتایج هر جستجو (جایگزین مقدار فایل تنظیمات)")
    s.add_argument("--local", action="append", help="پوشه‌ی فایل‌های صوتی محلی")
    s.add_argument("--city", help="نام شهر برای فایل‌های محلی")
    s.set_defaults(func=cmd_download)

    s = sub.add_parser("segment", help="برش فایل‌های خام به کلیپ")
    s.add_argument("--method", choices=["auto", "subtitles", "vad"])
    s.add_argument("--force", action="store_true", help="منابع قبلاً بریده‌شده را دوباره ببُر (بازبینی‌ها پاک می‌شود)")
    s.set_defaults(func=cmd_segment)

    s = sub.add_parser("label", help="تولید متن اولیه با Whisper")
    s.add_argument("--model", help="مثلاً openai/whisper-large-v3 یا whisper-small")
    s.add_argument("--replace-auto-subs", action="store_true", help="متن زیرنویس خودکار یوتیوب را هم با Whisper جایگزین کن")
    s.add_argument("--limit", type=int)
    s.add_argument("--batch-size", type=int)
    s.set_defaults(func=cmd_label)

    s = sub.add_parser("stats", help="آمار دیتاست")
    s.set_defaults(func=cmd_stats)

    s = sub.add_parser("split", help="تقسیم train/test")
    s.add_argument("--export", action="store_true", help="بعد از تقسیم، خروجی audiofolder هم بساز")
    s.set_defaults(func=cmd_split)

    s = sub.add_parser("export", help="خروجی audiofolder در data/export")
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("train", help="فاین‌تیون Whisper")
    s.add_argument("--model", help="کلید مدل از config/models.yaml")
    s.add_argument("--set", action="append", metavar="KEY=VALUE", help="بازنویسی یک پارامتر، مثلاً --set learning_rate=1e-5")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("evaluate", help="مقایسه‌ی WER مدل پایه و فاین‌تیون‌شده")
    s.add_argument("--base", help="مدل پایه (پیش‌فرض: مدل training.yaml)")
    s.add_argument("--finetuned", help="مسیر مدل فاین‌تیون‌شده (پیش‌فرض: <output_dir>/final)")
    s.add_argument("--limit", type=int)
    s.add_argument("--batch-size", type=int, default=8)
    s.add_argument("--beams", type=int, default=1, help="beam search (۱ = greedy، سریع‌تر)")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("panel", help="اجرای پنل وب")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=7860)
    s.add_argument("--share", action="store_true", help="لینک عمومی موقت gradio (برای کولب)")
    s.set_defaults(func=cmd_panel)
    return p


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # چاپ فارسی در ترمینال ویندوز
    a = build_parser().parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
