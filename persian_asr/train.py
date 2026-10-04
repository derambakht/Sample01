"""فاین‌تیون Whisper روی کلیپ‌های بخش train با تنظیمات config/training.yaml.

مراحل:
1. بارگذاری مدل پایه و پردازشگر (با زبان فارسی و وظیفه‌ی transcribe)
2. ساخت دیتاست از مانیفست (split=train/test) و تبدیل صدا به log-mel
3. آموزش با Seq2SeqTrainer و محاسبه‌ی WER روی test در فاصله‌های منظم
4. ذخیره‌ی بهترین مدل در <output_dir>/final (در حالت LoRA، ادغام‌شده با مدل پایه)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import audio
from .config import load_dataset_config, load_training_config, project_path, resolve_model_id
from .manifest import abs_audio, load_clips
from .text import clean_transcript, wer_cer


class ClipDataset:
    """دیتاست سبک PyTorch: صدا را از دیسک می‌خواند و ویژگی log-mel و برچسب توکنی برمی‌گرداند."""

    def __init__(self, items: list[tuple[str, list[int]]], feature_extractor):
        self.items = items
        self.fe = feature_extractor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, labels = self.items[i]
        wav = audio.load(path)[: 30 * 16000]
        feats = self.fe(wav, sampling_rate=16000).input_features[0]
        return {"input_features": feats, "labels": labels}


@dataclass
class Collator:
    """ویژگی‌ها و برچسب‌ها را در یک batch پد می‌کند؛ پدینگ برچسب با ‎-100 تا در loss حساب نشود."""

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features):
        inputs = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features], return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]  # مدل خودش توکن شروع را اضافه می‌کند
        inputs["labels"] = labels
        return inputs


def _build_items(split: str, tokenizer, max_label_length: int) -> list[tuple[str, list[int]]]:
    items, skipped = [], 0
    for c in load_clips():
        if c.split != split:
            continue
        ids = tokenizer(clean_transcript(c.text)).input_ids
        if len(ids) > max_label_length or c.duration > 30:
            skipped += 1
            continue
        items.append((str(abs_audio(c.audio_path)), ids))
    if skipped:
        print(f"⚠️ {skipped} کلیپ {split} به‌خاطر طول زیاد کنار گذاشته شد")
    return items


def train(overrides: dict[str, Any] | None = None) -> Path:
    import torch
    from transformers import (EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              TrainerCallback, WhisperForConditionalGeneration, WhisperProcessor)

    cfg = load_training_config()
    cfg.update(overrides or {})
    language = load_dataset_config().get("language", "persian")
    model_id = resolve_model_id(cfg["model"])
    out_dir = project_path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps({**cfg, "model_id": model_id}, ensure_ascii=False, indent=2))
    cuda = torch.cuda.is_available()
    print(f"🚀 مدل پایه: {model_id} | دستگاه: {'GPU ' + torch.cuda.get_device_name(0) if cuda else 'CPU'}")

    # ── مدل و پردازشگر ──
    processor = WhisperProcessor.from_pretrained(model_id, language=language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.generation_config.language = language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.use_cache = False
    model.config.apply_spec_augment = bool(cfg["apply_spec_augment"])
    model.config.mask_time_prob = float(cfg["mask_time_prob"])
    if cfg["freeze_encoder"]:
        model.freeze_encoder()

    if cfg["use_lora"]:
        from peft import LoraConfig, get_peft_model

        if cfg["gradient_checkpointing"]:
            model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(
            r=int(cfg["lora_r"]), lora_alpha=int(cfg["lora_alpha"]), lora_dropout=float(cfg["lora_dropout"]),
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"], bias="none"))
        model.print_trainable_parameters()

    # ── داده ──
    train_items = _build_items("train", processor.tokenizer, int(cfg["max_label_length"]))
    test_items = _build_items("test", processor.tokenizer, int(cfg["max_label_length"]))
    if not train_items:
        raise SystemExit("❌ هیچ کلیپ train نداریم. اول کلیپ‌ها را تأیید و دستور `split` را اجرا کن.")
    print(f"📊 train: {len(train_items)} | test: {len(test_items)}")
    train_ds = ClipDataset(train_items, processor.feature_extractor)
    test_ds = ClipDataset(test_items, processor.feature_extractor) if test_items else None

    def compute_metrics(pred):
        pred_ids = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        label_ids = np.where(pred.label_ids == -100, processor.tokenizer.pad_token_id, pred.label_ids)
        preds = processor.batch_decode(pred_ids, skip_special_tokens=True)
        refs = processor.batch_decode(label_ids, skip_special_tokens=True)
        m = wer_cer(refs, preds)
        return {"wer": 100 * m["wer"], "cer": 100 * m["cer"]}

    class JsonLogCallback(TrainerCallback):
        """هر لاگ (loss، WER، ...) را در train_log.jsonl می‌نویسد تا پنل نمودار بکشد."""

        def on_log(self, args, state, control, logs=None, **kw):
            with open(out_dir / "train_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"step": state.global_step, "epoch": state.epoch, "time": time.time(), **(logs or {})}) + "\n")

    (out_dir / "train_log.jsonl").unlink(missing_ok=True)
    has_eval = test_ds is not None
    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_steps=int(cfg["warmup_steps"]),
        weight_decay=float(cfg["weight_decay"]),
        num_train_epochs=float(cfg["num_train_epochs"]),
        max_steps=int(cfg["max_steps"]),
        gradient_checkpointing=bool(cfg["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False} if cfg["gradient_checkpointing"] else None,
        fp16=bool(cfg["fp16"]) and cuda,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=int(cfg["eval_steps"]),
        save_strategy="steps",
        save_steps=int(cfg["save_steps"]),
        save_total_limit=int(cfg["save_total_limit"]),
        logging_steps=int(cfg["logging_steps"]),
        predict_with_generate=True,
        generation_max_length=int(cfg["generation_max_length"]),
        load_best_model_at_end=has_eval,
        metric_for_best_model="wer" if has_eval else None,
        greater_is_better=False,
        report_to=["tensorboard"] if _has_tensorboard() else "none",
        dataloader_num_workers=int(cfg["dataloader_num_workers"]),
        label_names=["labels"],
        remove_unused_columns=False,
        seed=int(cfg["seed"]),
    )
    callbacks = [JsonLogCallback()]
    if has_eval and int(cfg["early_stopping_patience"]) > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(cfg["early_stopping_patience"])))

    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds,
        data_collator=Collator(processor, model.config.decoder_start_token_id
                               if not cfg["use_lora"] else model.base_model.model.config.decoder_start_token_id),
        compute_metrics=compute_metrics, processing_class=processor, callbacks=callbacks,
    )
    t0 = time.time()
    trainer.train()
    print(f"⏱️ زمان آموزش: {(time.time() - t0) / 60:.1f} دقیقه")

    # ── ذخیره‌ی مدل نهایی ──
    final = out_dir / "final"
    best = trainer.model
    if cfg["use_lora"]:
        best = best.merge_and_unload()  # مدل مستقل؛ بدون نیاز به peft برای استفاده
    best.config.use_cache = True
    best.save_pretrained(final)
    processor.save_pretrained(final)
    (final / "training_summary.json").write_text(json.dumps({
        "base_model": model_id, "train_clips": len(train_items), "test_clips": len(test_items),
        "minutes": (time.time() - t0) / 60, "best_metric": trainer.state.best_metric,
        "config": cfg,
    }, ensure_ascii=False, indent=2))
    print(f"✅ مدل نهایی ذخیره شد: {final}")
    return final


def _has_tensorboard() -> bool:
    try:
        import tensorboard  # noqa: F401
        return True
    except ImportError:
        return False
