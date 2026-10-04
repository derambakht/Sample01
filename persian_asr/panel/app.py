"""پنل وب پروژه (Gradio).

تب‌ها:
1. داشبورد      — چقدر صدا داریم و چقدر تا هدف مانده
2. بررسی نمونه‌ها — گوش دادن، اصلاح متن، تأیید/رد هر کلیپ
3. افزودن داده   — دانلود از یوتیوب/ایران‌صدا، برش، برچسب‌گذاری، ضبط صدای جدید
4. مدل و آموزش   — انتخاب مدل، تنظیم پارامترها (با توضیح)، اجرای آموزش و نمودار پیشرفت
5. تست و مقایسه  — مقایسه‌ی مدل پایه و فاین‌تیون‌شده روی test و روی صدای دلخواه

اجرا:  python -m persian_asr.cli panel
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter, deque
from pathlib import Path

import gradio as gr
import pandas as pd

from .. import audio
from ..config import (PROJECT_ROOT, data_dir, load_dataset_config, load_models, load_training_config,
                      project_path, save_training_config)
from ..config_schema import TRAINING_PARAMS
from ..manifest import Clip, Source, abs_audio, add_clips, load_clips, load_sources, update_clip, upsert_source
from ..text import clean_transcript, wer_cer

# ───────────────────────────── اجرای دستورهای طولانی در پس‌زمینه ─────────────────────────────

_JOB: dict = {"proc": None, "name": ""}
_JOB_LOCK = threading.Lock()


def run_cli(args: list[str], name: str):
    """یک زیر‌دستور CLI را به‌صورت پردازه‌ی جدا اجرا و خروجی‌اش را زنده نمایش می‌دهد."""
    with _JOB_LOCK:
        if _JOB["proc"] is not None and _JOB["proc"].poll() is None:
            yield f"⏳ کار «{_JOB['name']}» هنوز در حال اجراست. صبر کن یا «توقف» را بزن."
            return
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        cmd = [sys.executable, "-m", "persian_asr.cli", *args]
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                start_new_session=True)
        _JOB.update(proc=proc, name=name)
    lines: deque[str] = deque(maxlen=400)
    lines.append("$ python -m persian_asr.cli " + " ".join(shlex.quote(x) for x in args))
    last = 0.0
    for line in proc.stdout:  # type: ignore[union-attr]
        # نوار پیشرفت tqdm با \r خط را بازنویسی می‌کند؛ فقط آخرین حالت را نگه داریم
        line = line.rstrip("\n").split("\r")[-1]
        lines.append(line)
        if time.time() - last > 0.5:
            last = time.time()
            yield "\n".join(lines)
    proc.wait()
    lines.append(f"\n{'✅ تمام شد' if proc.returncode == 0 else f'❌ خطا (کد {proc.returncode})'}")
    yield "\n".join(lines)


def stop_job():
    p = _JOB.get("proc")
    if p is not None and p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except Exception:
            p.terminate()
        return f"⛔ کار «{_JOB['name']}» متوقف شد."
    return "کاری در حال اجرا نیست."


# ───────────────────────────── تب ۱: داشبورد ─────────────────────────────

def _bar(value: float, goal: float, label: str) -> str:
    pct = 0 if goal <= 0 else min(100.0, value / goal * 100)
    color = "#16a34a" if pct >= 100 else "#2563eb" if pct >= 50 else "#f59e0b"
    return (f'<div class="pbar"><div class="pbar-label">{label}</div>'
            f'<div class="pbar-track"><div class="pbar-fill" style="width:{pct:.0f}%;background:{color}"></div></div>'
            f'<div class="pbar-pct">{pct:.0f}%</div></div>')


HIST_BINS = [0, 2, 4, 6, 8, 10, 12, 15, 20, 30]
HIST_LABELS = [f"{a}-{b}s" for a, b in zip(HIST_BINS, HIST_BINS[1:])]


def dashboard():
    cfg = load_dataset_config()
    tgt = cfg.get("target", {})
    clips, sources = load_clips(), load_sources()
    approved = [c for c in clips if c.status == "approved"]
    st = Counter(c.status for c in clips)
    hours_all = sum(c.duration for c in clips) / 3600
    hours_ok = sum(c.duration for c in approved) / 3600
    avg = (sum(c.duration for c in clips) / len(clips)) if clips else 0
    split = Counter(c.split for c in clips if c.split)

    cards = f"""
<div class="cards">
  <div class="card"><div class="num">{len(sources)}</div><div class="lbl">منبع خام ({sum(s.duration for s in sources)/3600:.1f} ساعت)</div></div>
  <div class="card"><div class="num">{len(clips)}</div><div class="lbl">کل کلیپ‌ها ({hours_all:.2f} ساعت)</div></div>
  <div class="card ok"><div class="num">{st.get('approved', 0)}</div><div class="lbl">تأییدشده ({hours_ok*60:.0f} دقیقه)</div></div>
  <div class="card wait"><div class="num">{st.get('pending', 0)}</div><div class="lbl">در انتظار بازبینی</div></div>
  <div class="card bad"><div class="num">{st.get('rejected', 0)}</div><div class="lbl">ردشده</div></div>
  <div class="card"><div class="num">{avg:.1f}s</div><div class="lbl">میانگین طول کلیپ</div></div>
  <div class="card"><div class="num">{split.get('train', 0)} / {split.get('test', 0)}</div><div class="lbl">train / test</div></div>
</div>
<h4>پیشرفت نسبت به هدف (شهر: {cfg.get('city', '-')})</h4>
{_bar(len(approved), tgt.get('min_clips', 300), f"حداقل {tgt.get('min_clips', 300)} کلیپ تأییدشده")}
{_bar(len(approved), tgt.get('goal_clips', 800), f"هدف {tgt.get('goal_clips', 800)} کلیپ")}
{_bar(hours_ok, tgt.get('goal_hours', 1.5), f"هدف {tgt.get('goal_hours', 1.5)} ساعت صدای تأییدشده")}
"""
    bins, labels = HIST_BINS, HIST_LABELS
    hist = Counter()
    for c in clips:
        for (a, b), lab in zip(zip(bins, bins[1:]), labels):
            if a <= c.duration < b:
                hist[lab] += 1
                break
    df_hist = pd.DataFrame({"طول": labels, "تعداد": [hist[lab] for lab in labels]})

    per_src = Counter(c.source_id for c in clips)
    per_ok = Counter(c.source_id for c in approved)
    rows = [{
        "عنوان": (s.title or s.source_id)[:60], "نوع": s.kind, "دقیقه": round(s.duration / 60, 1),
        "زیرنویس": s.subs_kind or "-", "کلیپ": per_src[s.source_id], "تأییدشده": per_ok[s.source_id],
        "جستجو": s.query, "شناسه": s.source_id,
    } for s in sources]
    df_src = pd.DataFrame(rows or [{"عنوان": "هنوز منبعی نداریم — از تب «افزودن داده» شروع کن"}])

    tsrc = Counter((c.text_source or "بدون متن").split(":")[0] for c in clips)
    df_tsrc = pd.DataFrame({"منبع متن": list(tsrc), "تعداد": list(tsrc.values())}) if tsrc else \
        pd.DataFrame({"منبع متن": [], "تعداد": []})
    return cards, df_hist, df_tsrc, df_src


# ───────────────────────────── تب ۲: بررسی نمونه‌ها ─────────────────────────────

FILTERS = {"همه": None, "در انتظار بازبینی": "pending", "تأییدشده": "approved", "ردشده": "rejected",
           "بدون متن": "_empty", "بخش test": "_test"}


def _filtered_ids(flt: str, query: str = "") -> list[str]:
    key = FILTERS.get(flt)
    out = []
    for c in load_clips():
        if key == "_empty" and c.text.strip():
            continue
        if key == "_test" and c.split != "test":
            continue
        if key in STATUS_KEYS and c.status != key:
            continue
        if query and query not in c.text and query not in c.source_id:
            continue
        out.append(c.id)
    return out


STATUS_KEYS = {"pending", "approved", "rejected"}
STATUS_FA = {"pending": "🟡 در انتظار", "approved": "🟢 تأییدشده", "rejected": "🔴 ردشده"}


def _show(ids: list[str], idx: int):
    if not ids:
        return (gr.update(value=None), "", "", "**کلیپی با این فیلتر پیدا نشد.**", 0, "")
    idx = max(0, min(idx, len(ids) - 1))
    clips = {c.id: c for c in load_clips()}
    c = clips.get(ids[idx])
    if c is None:
        return (gr.update(value=None), "", "", "کلیپ پیدا نشد", idx, "")
    src = {s.source_id: s for s in load_sources()}.get(c.source_id)
    info = (f"**{idx + 1} / {len(ids)}** · `{c.id}` · {STATUS_FA.get(c.status, c.status)} · "
            f"{c.duration:.1f} ثانیه · منبع متن: `{c.text_source or '-'}`"
            + (f" · split: `{c.split}`" if c.split else "")
            + (f"\n\nمنبع: [{src.title or src.source_id}]({src.url}) — {c.start:.0f}s تا {c.end:.0f}s" if src else ""))
    sub_text = c.extra.get("subtitle_text", "") if isinstance(c.extra, dict) else ""
    auto = c.auto_text + (f"\n\n(زیرنویس یوتیوب: {sub_text})" if sub_text else "")
    return (str(abs_audio(c.audio_path)), auto, c.text, info, idx, c.notes)


def review_load(flt, query):
    ids = _filtered_ids(flt, query)
    return (ids, *_show(ids, 0))


def review_move(ids, idx, step):
    return _show(ids, idx + step)


def review_set(ids, idx, text, notes, status, advance):
    if ids:
        idx = max(0, min(idx, len(ids) - 1))
        changes = {"text": clean_transcript(text), "notes": notes}
        if status:
            changes["status"] = status
        c = load_clips_by_id().get(ids[idx])
        if c and clean_transcript(text) != c.auto_text and status != "rejected":
            changes["text_source"] = "manual" if not c.text_source else (
                c.text_source if c.text_source.endswith("+edited") or c.text_source == "manual"
                else c.text_source + "+edited")
        update_clip(ids[idx], **changes)
    return _show(ids, idx + (1 if advance else 0))


def load_clips_by_id() -> dict[str, Clip]:
    return {c.id: c for c in load_clips()}


def review_goto(ids, n):
    return _show(ids, int(n or 1) - 1)


# ───────────────────────────── تب ۳: افزودن داده ─────────────────────────────

def job_download(urls: str, queries: str, max_results):
    args = ["download"]
    for u in urls.split("\n"):
        if u.strip():
            args += ["--url", u.strip()]
    for q in queries.split("\n"):
        if q.strip():
            args += ["--query", q.strip()]
    if max_results:
        args += ["--max-results", str(int(max_results))]
    yield from run_cli(args, "دانلود")


def job_segment(method, force):
    args = ["segment", "--method", method] + (["--force"] if force else [])
    yield from run_cli(args, "برش")


def job_label(model, replace_auto, batch):
    args = ["label", "--model", model, "--batch-size", str(int(batch or 8))]
    if replace_auto:
        args.append("--replace-auto-subs")
    yield from run_cli(args, "برچسب‌گذاری")


def job_split():
    yield from run_cli(["split", "--export"], "تقسیم train/test")


def add_recording(path, text, speaker):
    if not path:
        return "❗ اول صدا را ضبط یا آپلود کن."
    if not text.strip():
        return "❗ متن دقیق گفته‌شده را بنویس."
    cfg = load_dataset_config()
    sid = f"rec_{(speaker or 'anon').strip().replace(' ', '_')}"
    cid = f"{sid}_{uuid.uuid4().hex[:8]}"
    dst = data_dir() / "clips" / sid / f"{cid}.wav"
    wav = audio.load(path)
    audio.save(dst, wav)
    dur = len(wav) / audio.SR
    if dur > 30:
        return f"❗ صدا {dur:.0f} ثانیه است؛ حداکثر ۳۰ ثانیه (بهتر زیر ۲۰ ثانیه) باشد."
    srcs = {s.source_id: s for s in load_sources()}
    src = srcs.get(sid) or Source(source_id=sid, title=f"ضبط: {speaker or 'ناشناس'}", kind="recording",
                                  city=cfg.get("city", ""), segmented=True)
    src.duration += dur
    upsert_source(src)
    add_clips([Clip(id=cid, source_id=sid, audio_path=str(dst.relative_to(data_dir())), end=dur,
                    duration=round(dur, 2), text=clean_transcript(text), auto_text=clean_transcript(text),
                    text_source="manual", status="approved", city=cfg.get("city", ""))])
    return f"✅ کلیپ {cid} ({dur:.1f} ثانیه) اضافه و تأیید شد."


# ───────────────────────────── تب ۴: مدل و آموزش ─────────────────────────────

def model_info(key: str) -> str:
    m = load_models().get(key, {})
    if not m:
        return ""
    star = " ⭐ پیشنهاد کلاس" if m.get("recommended") else ""
    return (f"### {key}{star}\n{m.get('description', '')}\n\n"
            f"| شناسه | پارامترها | حافظه‌ی GPU (کامل) | حافظه‌ی GPU (LoRA) | نرخ یادگیری پیشنهادی |\n|---|---|---|---|---|\n"
            f"| `{m.get('model_id')}` | {m.get('params')} | {m.get('vram_full')} | {m.get('vram_lora')} | {m.get('learning_rate')} |")


def on_model_change(key, use_lora):
    m = load_models().get(key, {})
    lr = 1e-3 if use_lora else float(m.get("learning_rate", 1e-5))
    city = load_dataset_config().get("city", "accent")
    return model_info(key), lr, f"outputs/{key}-{city}"


def _command_preview(values: dict) -> str:
    return "python -m persian_asr.cli train\n# یا با بازنویسی موقت:\npython -m persian_asr.cli train " + \
        " ".join(f"--set {k}={values[k]}" for k in ("learning_rate", "num_train_epochs") if k in values)


def save_cfg(*vals):
    values = dict(zip([p.key for p in TRAINING_PARAMS], vals))
    path = save_training_config(values)
    return f"💾 ذخیره شد: `{path.relative_to(PROJECT_ROOT)}`\n```bash\n{_command_preview(values)}\n```"


def start_training(*vals):
    save_cfg(*vals)
    yield from run_cli(["train"], "آموزش")


def training_curves():
    cfg = load_training_config()
    log = project_path(cfg["output_dir"]) / "train_log.jsonl"
    loss, wer = [], []
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "loss" in d:
                loss.append({"step": d["step"], "مقدار": d["loss"], "نوع": "train loss"})
            if "eval_loss" in d:
                loss.append({"step": d["step"], "مقدار": d["eval_loss"], "نوع": "eval loss"})
            if "eval_wer" in d:
                wer.append({"step": d["step"], "مقدار": d["eval_wer"], "نوع": "WER %"})
            if "eval_cer" in d:
                wer.append({"step": d["step"], "مقدار": d["eval_cer"], "نوع": "CER %"})
    return (pd.DataFrame(loss) if loss else None), (pd.DataFrame(wer) if wer else None)


# ───────────────────────────── تب ۵: تست و مقایسه ─────────────────────────────

def model_choices() -> list[str]:
    found = []
    out = PROJECT_ROOT / "outputs"
    if out.exists():
        for p in sorted(out.glob("*/final")):
            found.append(str(p.relative_to(PROJECT_ROOT)))
        for p in sorted(out.glob("*/checkpoint-*")):
            if (p / "config.json").exists():
                found.append(str(p.relative_to(PROJECT_ROOT)))
    return found + list(load_models())


def test_clip_choices() -> list[tuple[str, str]]:
    return [(f"{c.id} — {c.text[:50]}", c.id) for c in load_clips() if c.split == "test"][:500]


def pick_test_clip(cid):
    c = load_clips_by_id().get(cid)
    if not c:
        return None, ""
    return str(abs_audio(c.audio_path)), c.text


def compare_models(path, ref, model_a, model_b, beams):
    from ..asr import transcribe

    if not path:
        return "❗ صدایی انتخاب نشده", "", ""
    wav = audio.load(path)
    rows, texts = [], []
    for m in (model_a, model_b):
        if not m:
            texts.append("")
            continue
        t0 = time.time()
        try:
            t = transcribe(m, [wav], batch_size=1, num_beams=int(beams or 1))[0]
        except Exception as ex:
            t = f"❌ خطا: {ex}"
        texts.append(t)
        score = ""
        if ref.strip() and not t.startswith("❌"):
            r = wer_cer([ref], [t])
            score = f"{r['wer']*100:.1f}% | {r['cer']*100:.1f}%"
        rows.append(f"| `{m}` | {score or '-'} | {time.time() - t0:.1f}s |")
    md = "| مدل | WER \\| CER | زمان |\n|---|---|---|\n" + "\n".join(rows)
    return md, texts[0], texts[1]


def latest_report():
    js = PROJECT_ROOT / "reports" / "latest.json"
    csvp = PROJECT_ROOT / "reports" / "latest.csv"
    if not js.exists():
        return "هنوز ارزیابی‌ای اجرا نشده. دکمه‌ی «اجرای ارزیابی» را بزن.", pd.DataFrame()
    d = json.loads(js.read_text(encoding="utf-8"))
    res = d["results"]
    lines = [f"### نتیجه‌ی ارزیابی روی {d['n_clips']} کلیپ test ({d['time']})",
             "| مدل | WER | CER | زمان |", "|---|---|---|---|"]
    for name, r in res.items():
        if name.startswith("_"):
            continue
        lines.append(f"| **{name}** `{r['model']}` | {r['wer']*100:.1f}% | {r['cer']*100:.1f}% | {r['seconds']}s |")
    if "_improvement_pct" in res:
        imp = res["_improvement_pct"]
        lines.append(f"\n## {'📉' if imp > 0 else '📈'} بهبود نسبی WER: **{imp:.1f}%**")
        lines.append("عدد مثبت یعنی مدل فاین‌تیون‌شده کلمه‌های کمتری را اشتباه شنیده است.")
    df = pd.read_csv(csvp) if csvp.exists() else pd.DataFrame()
    wer_cols = [c for c in df.columns if c.startswith("wer_")]
    if len(wer_cols) >= 2:
        df["تفاوت"] = df[wer_cols[0]] - df[wer_cols[-1]]
        df = df.sort_values("تفاوت", ascending=False)
    return "\n".join(lines), df


def job_evaluate(limit, beams):
    args = ["evaluate", "--beams", str(int(beams or 1))]
    if limit:
        args += ["--limit", str(int(limit))]
    yield from run_cli(args, "ارزیابی")


# ───────────────────────────── چیدمان پنل ─────────────────────────────

CSS = """
.gradio-container {direction: rtl; font-family: Vazirmatn, Tahoma, sans-serif !important; max-width: 1280px !important}
.ltr, .ltr textarea, .ltr input, .js-plotly-plot, .vega-embed, table code {direction: ltr; text-align: left}
.log textarea {direction: ltr; text-align: left; font-family: ui-monospace, monospace; font-size: 12px}
.cards {display:grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap:10px; margin-bottom:12px}
.card {border:1px solid var(--border-color-primary); border-radius:10px; padding:12px; text-align:center; background: var(--background-fill-secondary)}
.card .num {font-size:26px; font-weight:700}
.card .lbl {font-size:13px; opacity:.8}
.card.ok .num {color:#16a34a} .card.wait .num {color:#d97706} .card.bad .num {color:#dc2626}
.pbar {display:flex; align-items:center; gap:10px; margin:6px 0}
.pbar-label {width:260px; font-size:14px}
.pbar-track {flex:1; height:14px; border-radius:7px; background: var(--background-fill-secondary); overflow:hidden; border:1px solid var(--border-color-primary)}
.pbar-fill {height:100%; border-radius:7px}
.pbar-pct {width:48px; text-align:left; font-variant-numeric: tabular-nums}
.help {font-size: 13px; opacity: .85}
table td, table th, table span, table input {font-family: Vazirmatn, Tahoma, sans-serif !important; letter-spacing: normal !important}
"""
HEAD = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;700&display=swap">'


def build() -> gr.Blocks:
    tcfg = load_training_config()
    models = load_models()

    with gr.Blocks(title="پنل دیتاست و فاین‌تیون لهجه‌ی فارسی") as demo:
        gr.Markdown("# 🎙️ پنل دیتاست لهجه‌ی فارسی و فاین‌تیون Whisper\n"
                    "مسیر کار: **افزودن داده** ← **برش** ← **برچسب‌گذاری** ← **بررسی و تأیید** ← **تقسیم train/test** ← **آموزش** ← **تست و مقایسه**")

        # ── تب ۱ ──
        with gr.Tab("📊 داشبورد"):
            gr.Markdown("خلاصه‌ی حجم داده. برای فاین‌تیون نمایشی حداقل ۳۰۰ کلیپ تأییدشده (حدود ۴۵ دقیقه) لازم است.", elem_classes="help")
            refresh = gr.Button("🔄 به‌روزرسانی", size="sm")
            cards = gr.HTML()
            with gr.Row():
                hist = gr.BarPlot(x="طول", y="تعداد", title="توزیع طول کلیپ‌ها", sort=HIST_LABELS)
                tsrc = gr.BarPlot(x="منبع متن", y="تعداد", title="متن کلیپ‌ها از کجا آمده؟")
            src_table = gr.Dataframe(label="منابع دانلودشده", interactive=False, wrap=True)
            refresh.click(dashboard, outputs=[cards, hist, tsrc, src_table])
            demo.load(dashboard, outputs=[cards, hist, tsrc, src_table])

        # ── تب ۲ ──
        with gr.Tab("🎧 بررسی نمونه‌ها"):
            gr.Markdown("هر کلیپ را گوش بده، متن را دقیقاً مطابق گفته‌شده اصلاح کن (کلمه‌های لهجه‌ای را همان‌طور که تلفظ شده بنویس، "
                        "مثلاً «میخام»، «دِگه») و تأیید کن. کلیپ‌های موسیقی، چندنفره‌ی هم‌زمان، تهرانی یا نامفهوم را رد کن.", elem_classes="help")
            ids_state, idx_state = gr.State([]), gr.State(0)
            with gr.Row():
                flt = gr.Dropdown(list(FILTERS), value="در انتظار بازبینی", label="فیلتر")
                q = gr.Textbox(label="جستجو در متن یا شناسه‌ی منبع")
                go_n = gr.Number(label="رفتن به شماره", precision=0, minimum=1)
                load_btn = gr.Button("📂 بارگذاری", variant="secondary")
            info = gr.Markdown()
            player = gr.Audio(type="filepath", label="صدا", interactive=False, autoplay=True)
            auto_box = gr.Textbox(label="متن اولیه (زیرنویس یا Whisper)", interactive=False, lines=2, rtl=True)
            text_box = gr.Textbox(label="متن نهایی (قابل ویرایش)", lines=3, rtl=True,
                                  info="همین متن برای آموزش و ارزیابی استفاده می‌شود.")
            notes_box = gr.Textbox(label="یادداشت (اختیاری)", lines=1, rtl=True)
            with gr.Row():
                prev_b = gr.Button("→ قبلی")
                ok_b = gr.Button("✓ تأیید و بعدی", variant="primary")
                save_b = gr.Button("💾 فقط ذخیره‌ی متن")
                rej_b = gr.Button("✗ رد و بعدی", variant="stop")
                next_b = gr.Button("بعدی ←")
            show_out = [player, auto_box, text_box, info, idx_state, notes_box]
            load_btn.click(review_load, [flt, q], [ids_state, *show_out])
            flt.change(review_load, [flt, q], [ids_state, *show_out])
            q.submit(review_load, [flt, q], [ids_state, *show_out])
            go_n.submit(review_goto, [ids_state, go_n], show_out)
            prev_b.click(lambda ids, i: review_move(ids, i, -1), [ids_state, idx_state], show_out)
            next_b.click(lambda ids, i: review_move(ids, i, 1), [ids_state, idx_state], show_out)
            ok_b.click(lambda ids, i, t, n: review_set(ids, i, t, n, "approved", True),
                       [ids_state, idx_state, text_box, notes_box], show_out)
            rej_b.click(lambda ids, i, t, n: review_set(ids, i, t, n, "rejected", True),
                        [ids_state, idx_state, text_box, notes_box], show_out)
            save_b.click(lambda ids, i, t, n: review_set(ids, i, t, n, None, False),
                         [ids_state, idx_state, text_box, notes_box], show_out)

        # ── تب ۳ ──
        with gr.Tab("➕ افزودن داده"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ۱. دانلود\nعلاوه بر منابع `config/sources.yaml`، می‌توانی لینک یا عبارت جستجوی اضافه بدهی.", elem_classes="help")
                    urls = gr.Textbox(label="لینک‌ها (هر خط یک لینک: یوتیوب، آپارات، پلی‌لیست، فایل mp3)", lines=3, elem_classes="ltr")
                    queries = gr.Textbox(label="عبارت‌های جستجوی یوتیوب (هر خط یکی)", lines=2, rtl=True,
                                         placeholder="مثلاً: گفتگو با لهجه اصفهانی")
                    maxr = gr.Number(label="سقف نتایج هر جستجو (خالی = مقدار فایل تنظیمات)", precision=0)
                    dl_b = gr.Button("⬇️ شروع دانلود", variant="primary")
                    gr.Markdown("### ۲. برش به کلیپ\n`auto`: اگر زیرنویس بود از زمان‌بندی زیرنویس، وگرنه تشخیص گفتار (VAD).", elem_classes="help")
                    with gr.Row():
                        seg_m = gr.Radio(["auto", "subtitles", "vad"], value="auto", label="روش برش")
                        seg_f = gr.Checkbox(label="برش دوباره‌ی همه (بازبینی‌ها پاک می‌شود)")
                    seg_b = gr.Button("✂️ برش")
                    gr.Markdown("### ۳. برچسب‌گذاری خودکار\nبرای کلیپ‌های بی‌متن با یک Whisper بزرگ متن اولیه می‌سازد تا در بازبینی فقط اصلاحش کنی.", elem_classes="help")
                    lab_cfg = load_dataset_config().get("labeling", {})
                    with gr.Row():
                        lab_m = gr.Dropdown(["openai/whisper-large-v3", "openai/whisper-large-v3-turbo", "openai/whisper-medium",
                                             "openai/whisper-small"], value=lab_cfg.get("model", "openai/whisper-large-v3"),
                                            label="مدل برچسب‌گذار", allow_custom_value=True)
                        lab_bs = gr.Number(value=lab_cfg.get("batch_size", 8), label="batch", precision=0)
                    lab_r = gr.Checkbox(label="متن زیرنویس خودکار یوتیوب را هم با Whisper جایگزین کن",
                                        info="زیرنویس خودکار یوتیوب برای فارسی کیفیت پایینی دارد؛ معمولاً Whisper large بهتر است.")
                    lab_b = gr.Button("🧠 برچسب‌گذاری")
                    gr.Markdown("### ۴. تقسیم train/test\nکلیپ‌های تأییدشده را بر اساس منبع تقسیم و در `data/export` خروجی می‌گیرد.", elem_classes="help")
                    split_b = gr.Button("🔀 تقسیم و خروجی")
                with gr.Column():
                    log3 = gr.Textbox(label="گزارش اجرا", lines=26, max_lines=26, elem_classes="log", autoscroll=True)
                    stop3 = gr.Button("⛔ توقف کار جاری", size="sm")
            dl_b.click(job_download, [urls, queries, maxr], log3)
            seg_b.click(job_segment, [seg_m, seg_f], log3)
            lab_b.click(job_label, [lab_m, lab_r, lab_bs], log3)
            split_b.click(job_split, None, log3)
            stop3.click(stop_job, None, log3)

            with gr.Accordion("🎤 ضبط مستقیم صدای گوینده‌ی اصفهانی", open=False):
                gr.Markdown("بهترین داده: یک اصفهانی جمله‌ای را بخواند یا آزاد صحبت کند و متن دقیقش را بنویسی. کلیپ مستقیم «تأییدشده» ذخیره می‌شود.", elem_classes="help")
                with gr.Row():
                    rec = gr.Audio(sources=["microphone", "upload"], type="filepath", label="ضبط یا آپلود (حداکثر ۳۰ ثانیه)")
                    with gr.Column():
                        rec_text = gr.Textbox(label="متن دقیق گفته‌شده", lines=3, rtl=True)
                        rec_spk = gr.Textbox(label="نام/کد گوینده", placeholder="مثلاً speaker01",
                                             info="کلیپ‌های یک گوینده با هم در train یا test قرار می‌گیرند.")
                        rec_b = gr.Button("➕ افزودن به دیتاست", variant="primary")
                        rec_msg = gr.Markdown()
                rec_b.click(add_recording, [rec, rec_text, rec_spk], rec_msg)

        # ── تب ۴ ──
        with gr.Tab("⚙️ مدل و آموزش"):
            comps: dict[str, gr.components.Component] = {}
            model_md = gr.Markdown(model_info(tcfg["model"]))
            group = None
            acc = None
            for p in TRAINING_PARAMS:
                if p.group != group:
                    group = p.group
                    acc = gr.Accordion(f"تنظیمات {group}", open=group in ("مدل", "بهینه‌سازی"))
                with acc:
                    v = tcfg[p.key]
                    if p.key == "model":
                        c = gr.Dropdown(list(models), value=v, label=p.label, info=p.help)
                    elif p.kind == "choice":
                        c = gr.Dropdown(p.choices, value=v, label=p.label, info=p.help)
                    elif p.kind == "bool":
                        c = gr.Checkbox(value=v, label=p.label, info=p.help)
                    elif p.kind == "int":
                        c = gr.Number(value=v, label=p.label, info=p.help, precision=0)
                    elif p.kind == "text":
                        c = gr.Textbox(value=v, label=p.label, info=p.help, elem_classes="ltr")
                    else:
                        c = gr.Number(value=v, label=p.label, info=p.help)
                comps[p.key] = c
            ordered = [comps[p.key] for p in TRAINING_PARAMS]
            comps["model"].change(on_model_change, [comps["model"], comps["use_lora"]],
                                  [model_md, comps["learning_rate"], comps["output_dir"]])
            comps["use_lora"].change(lambda k, u: on_model_change(k, u)[1], [comps["model"], comps["use_lora"]],
                                     comps["learning_rate"])
            with gr.Row():
                save_b4 = gr.Button("💾 ذخیره‌ی تنظیمات")
                train_b = gr.Button("🚀 ذخیره و شروع آموزش", variant="primary")
                stop4 = gr.Button("⛔ توقف", variant="stop")
            save_msg = gr.Markdown()
            log4 = gr.Textbox(label="لاگ آموزش", lines=16, max_lines=16, elem_classes="log", autoscroll=True)
            gr.Markdown("### نمودار پیشرفت\nloss باید پایین بیاید؛ WER روی test هم باید کم شود. اگر loss آموزش کم شد ولی WER بالا رفت، مدل دارد حفظ می‌کند (overfit).", elem_classes="help")
            with gr.Row():
                loss_plot = gr.LinePlot(x="step", y="مقدار", color="نوع", title="Loss")
                wer_plot = gr.LinePlot(x="step", y="مقدار", color="نوع", title="WER / CER روی test (%)")
            curves_b = gr.Button("🔄 به‌روزرسانی نمودار", size="sm")
            timer = gr.Timer(10)
            save_b4.click(save_cfg, ordered, save_msg)
            train_b.click(start_training, ordered, log4)
            stop4.click(stop_job, None, log4)
            curves_b.click(training_curves, None, [loss_plot, wer_plot])
            timer.tick(training_curves, None, [loss_plot, wer_plot])
            demo.load(training_curves, None, [loss_plot, wer_plot])

        # ── تب ۵ ──
        with gr.Tab("🧪 تست و مقایسه"):
            gr.Markdown("### ارزیابی کامل روی بخش test\nمدل پایه و مدل فاین‌تیون‌شده روی همه‌ی کلیپ‌های test اجرا و WER آن‌ها مقایسه می‌شود.", elem_classes="help")
            with gr.Row():
                ev_limit = gr.Number(label="حداکثر تعداد کلیپ (خالی = همه)", precision=0)
                ev_beams = gr.Number(label="beam search", value=1, precision=0, info="۱ سریع‌ترین؛ ۵ کمی دقیق‌تر ولی ۳-۵ برابر کندتر")
                ev_b = gr.Button("▶️ اجرای ارزیابی", variant="primary")
                rep_b = gr.Button("🔄 نمایش آخرین گزارش")
            log5 = gr.Textbox(label="لاگ ارزیابی", lines=6, max_lines=8, elem_classes="log", autoscroll=True)
            rep_md = gr.Markdown()
            rep_df = gr.Dataframe(label="مقایسه‌ی جمله‌به‌جمله (مرتب بر اساس بیشترین بهبود)", wrap=True, interactive=False)
            ev_b.click(job_evaluate, [ev_limit, ev_beams], log5).then(latest_report, None, [rep_md, rep_df])
            rep_b.click(latest_report, None, [rep_md, rep_df])
            demo.load(latest_report, None, [rep_md, rep_df])

            gr.Markdown("---\n### تست زنده\nیک کلیپ test انتخاب کن یا صدای خودت را ضبط کن و خروجی دو مدل را کنار هم ببین.", elem_classes="help")
            choices = model_choices()
            with gr.Row():
                test_pick = gr.Dropdown(test_clip_choices(), label="انتخاب کلیپ از test", allow_custom_value=False)
                refresh5 = gr.Button("🔄", size="sm", scale=0)
            with gr.Row():
                t_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="صدا")
                t_ref = gr.Textbox(label="متن مرجع (اختیاری؛ برای محاسبه‌ی WER)", lines=3, rtl=True)
            with gr.Row():
                m_a = gr.Dropdown(choices, value=tcfg["model"], label="مدل A (معمولاً مدل پایه)", allow_custom_value=True)
                ft_default = next((c for c in choices if c.endswith("/final")), None)
                m_b = gr.Dropdown(choices, value=ft_default, label="مدل B (معمولاً مدل فاین‌تیون‌شده)", allow_custom_value=True)
                t_beams = gr.Number(value=1, label="beam", precision=0)
            t_b = gr.Button("🎯 تبدیل به متن با هر دو مدل", variant="primary")
            t_md = gr.Markdown()
            with gr.Row():
                out_a = gr.Textbox(label="خروجی مدل A", lines=3, rtl=True)
                out_b = gr.Textbox(label="خروجی مدل B", lines=3, rtl=True)
            test_pick.change(pick_test_clip, test_pick, [t_audio, t_ref])
            refresh5.click(lambda: (gr.update(choices=test_clip_choices()), gr.update(choices=model_choices()),
                                    gr.update(choices=model_choices())), None, [test_pick, m_a, m_b])
            t_b.click(compare_models, [t_audio, t_ref, m_a, m_b, t_beams], [t_md, out_a, out_b])

    return demo


def main(host: str = "127.0.0.1", port: int = 7860, share: bool = False):
    demo = build()
    demo.queue().launch(server_name=host, server_port=port, share=share, css=CSS, head=HEAD,
                        allowed_paths=[str(data_dir())], theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
