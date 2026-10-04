"""تعریف پارامترهای آموزش، مقدار پیش‌فرض و توضیح فارسی هر کدام.

این فایل تنها منبع توضیح پارامترهاست: پنل از آن فرم می‌سازد و
`config/training.yaml` هم با همین توضیح‌ها (به‌صورت کامنت) نوشته می‌شود.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Param:
    key: str
    default: Any
    label: str
    help: str
    group: str
    kind: str = "number"  # number | int | bool | text | choice
    choices: list[str] = field(default_factory=list)


TRAINING_PARAMS: list[Param] = [
    # ── مدل ──
    Param("model", "whisper-small", "مدل پایه",
          "کلید یکی از مدل‌های config/models.yaml. مدلی که وزن‌هایش با دیتاست لهجه به‌روز می‌شود.",
          "مدل", kind="choice"),
    Param("output_dir", "outputs/whisper-small-isfahani", "پوشه‌ی خروجی",
          "چک‌پوینت‌ها، لاگ آموزش و مدل نهایی (در زیرپوشه‌ی final) اینجا ذخیره می‌شوند.",
          "مدل", kind="text"),
    Param("freeze_encoder", False, "فریز کردن انکودر",
          "انکودر (بخش شنیداری) ثابت می‌ماند و فقط دیکودر یاد می‌گیرد. حافظه کمتر و خطر overfit کمتر، ولی تطبیق با تلفظ لهجه ضعیف‌تر.",
          "مدل", kind="bool"),
    # ── LoRA ──
    Param("use_lora", False, "استفاده از LoRA",
          "به‌جای همه‌ی وزن‌ها فقط ماتریس‌های کوچک کمکی آموزش می‌بینند. برای medium/large یا GPU کم‌حافظه روشنش کن.",
          "LoRA", kind="bool"),
    Param("lora_r", 32, "رتبه‌ی LoRA (r)",
          "اندازه‌ی ماتریس‌های کمکی. بزرگ‌تر = ظرفیت یادگیری بیشتر و حافظه‌ی بیشتر. ۱۶ تا ۶۴ معمول است.",
          "LoRA", kind="int"),
    Param("lora_alpha", 64, "ضریب LoRA (alpha)",
          "شدت اثر وزن‌های LoRA. معمولاً دو برابر r گذاشته می‌شود.",
          "LoRA", kind="int"),
    Param("lora_dropout", 0.05, "Dropout در LoRA",
          "درصد خاموش‌کردن تصادفی در لایه‌های LoRA برای جلوگیری از حفظ‌کردن داده‌ی کم.",
          "LoRA"),
    # ── بهینه‌سازی ──
    Param("learning_rate", 1.0e-5, "نرخ یادگیری",
          "اندازه‌ی گام به‌روزرسانی وزن‌ها. برای فاین‌تیون کامل ۱e-5 و برای LoRA حدود ۱e-3. زیاد باشد مدل فارسی‌اش را «فراموش» می‌کند.",
          "بهینه‌سازی"),
    Param("num_train_epochs", 5, "تعداد دوره (epoch)",
          "چند بار کل داده‌ی train دیده شود. با داده‌ی کم (۳۰۰-۸۰۰ کلیپ) ۳ تا ۸ دوره مناسب است.",
          "بهینه‌سازی"),
    Param("max_steps", -1, "حداکثر گام",
          "اگر بزرگ‌تر از صفر باشد به‌جای epoch، آموزش بعد از این تعداد گام متوقف می‌شود. ‎-1 یعنی غیرفعال.",
          "بهینه‌سازی", kind="int"),
    Param("per_device_train_batch_size", 8, "اندازه‌ی batch آموزش",
          "تعداد کلیپ در هر گام روی هر GPU. اگر خطای کمبود حافظه (OOM) گرفتی کمش کن.",
          "بهینه‌سازی", kind="int"),
    Param("gradient_accumulation_steps", 2, "انباشت گرادیان",
          "گرادیان چند گام جمع می‌شود و بعد وزن‌ها آپدیت می‌شوند؛ batch مؤثر = batch × این عدد. جایگزین batch بزرگ با حافظه‌ی کم.",
          "بهینه‌سازی", kind="int"),
    Param("warmup_steps", 50, "گام‌های گرم‌کردن",
          "نرخ یادگیری در این تعداد گام اول از صفر به مقدار اصلی می‌رسد تا شروع آموزش مدل را خراب نکند.",
          "بهینه‌سازی", kind="int"),
    Param("weight_decay", 0.01, "Weight decay",
          "جریمه‌ی کوچک برای وزن‌های بزرگ؛ کمی جلوی overfit روی دیتاست کوچک را می‌گیرد.",
          "بهینه‌سازی"),
    Param("lr_scheduler_type", "linear", "زمان‌بندی نرخ یادگیری",
          "نحوه‌ی کاهش نرخ یادگیری در طول آموزش. linear ساده و مطمئن است؛ cosine نرم‌تر کم می‌کند.",
          "بهینه‌سازی", kind="choice", choices=["linear", "cosine", "constant_with_warmup"]),
    # ── داده و تقویت ──
    Param("apply_spec_augment", True, "SpecAugment",
          "بخش‌هایی از طیف صدا را تصادفی می‌پوشاند تا مدل با داده‌ی کم بهتر تعمیم دهد. برای دیتاست کوچک روشن بماند.",
          "داده", kind="bool"),
    Param("mask_time_prob", 0.05, "احتمال ماسک زمانی",
          "سهم فریم‌های زمانی که در SpecAugment پوشانده می‌شوند. ۰.۰۵ تا ۰.۱ معمول است.",
          "داده"),
    Param("max_label_length", 225, "حداکثر طول متن (توکن)",
          "متن‌های طولانی‌تر از این تعداد توکن از آموزش حذف می‌شوند. سقف Whisper ۴۴۸ است.",
          "داده", kind="int"),
    # ── ارزیابی و ذخیره ──
    Param("eval_steps", 100, "فاصله‌ی ارزیابی (گام)",
          "هر چند گام یک بار WER روی داده‌ی test محاسبه شود. کمترش کنی نمودار دقیق‌تر ولی آموزش کندتر می‌شود.",
          "ارزیابی", kind="int"),
    Param("save_steps", 100, "فاصله‌ی ذخیره (گام)",
          "هر چند گام چک‌پوینت ذخیره شود. باید مضربی از فاصله‌ی ارزیابی باشد تا بهترین مدل انتخاب شود.",
          "ارزیابی", kind="int"),
    Param("logging_steps", 10, "فاصله‌ی ثبت لاگ (گام)",
          "هر چند گام loss در لاگ نوشته شود (برای نمودار پنل).",
          "ارزیابی", kind="int"),
    Param("per_device_eval_batch_size", 8, "اندازه‌ی batch ارزیابی",
          "تعداد کلیپ در هر گام ارزیابی. تولید متن حافظه‌ی بیشتری می‌گیرد؛ در صورت OOM کمش کن.",
          "ارزیابی", kind="int"),
    Param("generation_max_length", 225, "حداکثر طول خروجی",
          "حداکثر تعداد توکنی که مدل هنگام ارزیابی تولید می‌کند.",
          "ارزیابی", kind="int"),
    Param("early_stopping_patience", 3, "صبر توقف زودهنگام",
          "اگر WER این تعداد ارزیابی پشت‌سرهم بهتر نشد آموزش متوقف می‌شود. ۰ یعنی غیرفعال.",
          "ارزیابی", kind="int"),
    Param("save_total_limit", 2, "تعداد چک‌پوینت نگه‌داشته‌شده",
          "چک‌پوینت‌های قدیمی‌تر پاک می‌شوند تا دیسک پر نشود (بهترین مدل همیشه نگه داشته می‌شود).",
          "ارزیابی", kind="int"),
    # ── سخت‌افزار ──
    Param("fp16", True, "دقت نیمه (fp16)",
          "روی GPU انویدیا حافظه را نصف و سرعت را بیشتر می‌کند. روی CPU خودکار خاموش می‌شود.",
          "سخت‌افزار", kind="bool"),
    Param("gradient_checkpointing", True, "Gradient checkpointing",
          "حافظه‌ی GPU را خیلی کمتر می‌کند به قیمت حدود ۲۰٪ کندی. برای small به بالا روشن بماند.",
          "سخت‌افزار", kind="bool"),
    Param("dataloader_num_workers", 2, "تعداد worker بارگذاری داده",
          "تعداد پردازه‌هایی که هم‌زمان صدا را آماده می‌کنند. روی ویندوز اگر خطا داد ۰ بگذار.",
          "سخت‌افزار", kind="int"),
    Param("seed", 42, "Seed تصادفی",
          "عدد ثابت برای تکرارپذیر بودن نتایج بین اجراهای مختلف.",
          "سخت‌افزار", kind="int"),
]

PARAMS_BY_KEY = {p.key: p for p in TRAINING_PARAMS}


def default_training_config() -> dict[str, Any]:
    return {p.key: p.default for p in TRAINING_PARAMS}


def coerce(key: str, value: Any) -> Any:
    """مقدار ورودی (مثلاً از فرم پنل) را به نوع درست پارامتر تبدیل می‌کند."""
    p = PARAMS_BY_KEY.get(key)
    if p is None or value is None:
        return value
    if p.kind == "bool":
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if p.kind == "int":
        return int(float(value))
    if p.kind == "number":
        return float(value)
    return str(value)
