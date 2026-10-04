"""دانلود صدا از یوتیوب (جستجو یا لینک)، صفحه‌های ایران‌صدا و پوشه‌های محلی.

هر منبع به wav ۱۶ کیلوهرتز مونو تبدیل و در data/raw/ ذخیره می‌شود و
یک ردیف در data/sources.jsonl می‌گیرد. منابعی که قبلاً دانلود شده‌اند دوباره دانلود نمی‌شوند.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin

from . import audio
from .config import data_dir, load_dataset_config, load_sources
from .manifest import Source, load_sources as load_source_records, upsert_source

Log = Callable[[str], None]
AUDIO_EXT = {".mp3", ".m4a", ".aac", ".wav", ".ogg", ".opus", ".flac", ".webm", ".mp4", ".mkv", ".wma"}


def _sid(prefix: str, key: str) -> str:
    return f"{prefix}_{hashlib.sha1(key.encode()).hexdigest()[:10]}"


def _raw_dir() -> Path:
    d = data_dir() / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _rel(p: Path) -> str:
    return str(p.relative_to(data_dir()))


def _known_ids() -> set[str]:
    return {s.source_id for s in load_source_records()}


# ───────────────────────────── یوتیوب و سایت‌های پشتیبانی‌شده‌ی yt-dlp ─────────────────────────────

def search_youtube(query: str, max_results: int, log: Log = print) -> list[dict]:
    """با yt-dlp در یوتیوب جستجو می‌کند (بدون دانلود) و فهرست ویدیوها را برمی‌گرداند."""
    import yt_dlp

    opts = {"quiet": True, "extract_flat": True, "skip_download": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
    entries = [e for e in (info or {}).get("entries", []) if e]
    log(f"🔎 «{query}»: {len(entries)} نتیجه")
    return entries


def download_with_ytdlp(url: str, city: str, query: str = "", kind: str = "youtube",
                        log: Log = print) -> Source | None:
    """یک ویدیو را با زیرنویس فارسی (اگر موجود بود) دانلود و به wav تبدیل می‌کند."""
    import yt_dlp

    cfg = load_dataset_config()
    dl = cfg.get("download", {})
    max_s = float(dl.get("max_minutes", 40)) * 60
    min_s = float(dl.get("min_seconds", 30))
    langs = dl.get("subtitle_langs", ["fa"])

    def length_filter(info, *, incomplete):
        d = info.get("duration")
        if d and (d > max_s or d < min_s):
            return f"طول {d:.0f} ثانیه خارج از محدوده است"
        return None

    tmp = Path(tempfile.mkdtemp(prefix="pasr_"))
    opts = {
        "quiet": True, "no_warnings": True, "noplaylist": False,
        "format": "bestaudio/best",
        "outtmpl": str(tmp / "%(id)s.%(ext)s"),
        "writesubtitles": bool(dl.get("subtitles", True)),
        "writeautomaticsub": bool(dl.get("subtitles", True) and dl.get("allow_auto_subtitles", True)),
        "subtitleslangs": langs,
        "subtitlesformat": "vtt/best",
        "match_filter": length_filter,
        "ignoreerrors": True,
        "retries": 3,
    }
    list_opts = {"quiet": True, "no_warnings": True, "extract_flat": "in_playlist", "ignoreerrors": True}
    results: list[Source] = []
    try:
        with yt_dlp.YoutubeDL(list_opts) as lister:
            info = lister.extract_info(url, download=False)
        with yt_dlp.YoutubeDL(opts) as ydl:
            if not info:
                log(f"⚠️ اطلاعات {url} خوانده نشد")
                return None
            entries = info.get("entries") or [info]  # پلی‌لیست یا ویدیوی تکی
            known = _known_ids()
            for e in entries:
                if not e:
                    continue
                vid = e.get("id") or _sid("v", e.get("webpage_url", url))
                sid = f"{'yt' if kind == 'youtube' else 'web'}_{vid}"
                if sid in known:
                    log(f"⏭️ قبلاً دانلود شده: {e.get('title', vid)}")
                    continue
                reason = length_filter(e, incomplete=False)
                if reason:
                    log(f"⏭️ رد شد ({reason}): {e.get('title', vid)}")
                    continue
                page = e.get("webpage_url") or e.get("url") or url
                log(f"⬇️ دانلود: {e.get('title', vid)}")
                full = ydl.extract_info(page, download=True)
                if not full:
                    log("   ⚠️ دانلود ناموفق")
                    continue
                media = [p for p in tmp.glob(f"{full['id']}.*") if p.suffix.lower() not in {".vtt", ".srt", ".json"}]
                if not media:
                    log("   ⚠️ فایل صوتی پیدا نشد")
                    continue
                wav = audio.to_wav16k(media[0], _raw_dir() / f"{sid}.wav")
                subs_path, subs_kind = "", ""
                manual = full.get("subtitles") or {}
                for lang in langs:
                    vtt = tmp / f"{full['id']}.{lang}.vtt"
                    if vtt.exists():
                        dst = _raw_dir() / f"{sid}.{lang}.vtt"
                        shutil.move(str(vtt), dst)
                        subs_path, subs_kind = _rel(dst), ("manual" if lang in manual else "auto")
                        break
                src = Source(source_id=sid, url=page, title=full.get("title", ""), kind=kind,
                             query=query, city=city, audio_path=_rel(wav), subs_path=subs_path,
                             subs_kind=subs_kind, duration=audio.duration(wav))
                upsert_source(src)
                results.append(src)
                log(f"   ✅ {src.duration/60:.1f} دقیقه" + (f" + زیرنویس ({subs_kind})" if subs_path else " (بدون زیرنویس)"))
                for p in tmp.glob(f"{full['id']}.*"):
                    p.unlink(missing_ok=True)
    except Exception as ex:  # شبکه، محدودیت یوتیوب، ...
        log(f"❌ خطا در {url}: {ex}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return results[0] if results else None


# ───────────────────────────── ایران‌صدا (و هر صفحه‌ای با لینک مستقیم فایل صوتی) ─────────────────────────────

def find_audio_links(page_url: str) -> list[str]:
    """لینک‌های mp3/m4a/... داخل HTML یک صفحه را پیدا می‌کند."""
    import requests

    html = requests.get(page_url, timeout=30, headers={"User-Agent": "Mozilla/5.0"}).text
    links = re.findall(r"""["'(]([^"'()\s]+?\.(?:mp3|m4a|aac|ogg|wav)(?:\?[^"'()\s]*)?)["')]""", html, re.I)
    seen, out = set(), []
    for link in links:
        full = urljoin(page_url, link.replace("\\/", "/"))
        if full not in seen:
            seen.add(full)
            out.append(full)
    return out


def download_direct(url: str, city: str, page: str = "", kind: str = "iranseda",
                    log: Log = print) -> Source | None:
    import requests

    sid = _sid("irs" if kind == "iranseda" else "url", url)
    if sid in _known_ids():
        log(f"⏭️ قبلاً دانلود شده: {url}")
        return None
    log(f"⬇️ دانلود فایل: {url}")
    suffix = Path(url.split("?")[0]).suffix or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        tmp = Path(f.name)
        with requests.get(url, stream=True, timeout=60, headers={"User-Agent": "Mozilla/5.0"}) as r:
            r.raise_for_status()
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
    try:
        wav = audio.to_wav16k(tmp, _raw_dir() / f"{sid}.wav")
    finally:
        tmp.unlink(missing_ok=True)
    src = Source(source_id=sid, url=page or url, title=Path(url.split("?")[0]).name, kind=kind,
                 city=city, audio_path=_rel(wav), duration=audio.duration(wav))
    upsert_source(src)
    log(f"   ✅ {src.duration/60:.1f} دقیقه")
    return src


def download_iranseda(page_url: str, city: str, log: Log = print) -> int:
    try:
        links = find_audio_links(page_url)
    except Exception as ex:
        log(f"❌ صفحه باز نشد {page_url}: {ex}")
        links = []
    if not links:
        log("ℹ️ لینک مستقیم پیدا نشد؛ تلاش با yt-dlp (استخراج‌کننده‌ی عمومی)")
        return 1 if download_with_ytdlp(page_url, city, kind="iranseda", log=log) else 0
    n = 0
    for link in links:
        try:
            if download_direct(link, city, page=page_url, log=log):
                n += 1
        except Exception as ex:
            log(f"❌ {link}: {ex}")
    return n


# ───────────────────────────── فایل‌های محلی ─────────────────────────────

def import_local(folder: str | Path, city: str, log: Log = print) -> int:
    """فایل‌های صوتی یک پوشه را اضافه می‌کند. اگر فایل .txt هم‌نام بود، متن آن ذخیره می‌شود."""
    folder = Path(folder)
    n = 0
    known = _known_ids()
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() not in AUDIO_EXT:
            continue
        sid = _sid("loc", str(p.resolve()))
        if sid in known:
            continue
        wav = audio.to_wav16k(p, _raw_dir() / f"{sid}.wav")
        txt = p.with_suffix(".txt")
        src = Source(source_id=sid, url=str(p), title=p.stem, kind="local", city=city,
                     audio_path=_rel(wav), duration=audio.duration(wav))
        if txt.exists():
            dst = _raw_dir() / f"{sid}.txt"
            shutil.copy(txt, dst)
            src.subs_path, src.subs_kind = _rel(dst), "manual"
        upsert_source(src)
        n += 1
        log(f"✅ {p.name} ({src.duration:.1f} ثانیه)")
    return n


# ───────────────────────────── اجرای همه‌ی منابع config/sources.yaml ─────────────────────────────

def download_all(extra_urls: list[str] | None = None, extra_queries: list[str] | None = None,
                 max_results: int | None = None, log: Log = print) -> None:
    cfg = load_dataset_config()
    city = cfg.get("city", "")
    sources = load_sources()

    searches = list(sources["youtube_searches"])
    searches += [{"query": q} for q in (extra_queries or [])]
    for s in searches:
        q = s["query"] if isinstance(s, dict) else str(s)
        n = max_results or (s.get("max_results", 10) if isinstance(s, dict) else 10)
        try:
            for e in search_youtube(q, n, log=log):
                link = e.get("url") or e.get("webpage_url") or f"https://www.youtube.com/watch?v={e['id']}"
                download_with_ytdlp(link, city, query=q, log=log)
        except Exception as ex:
            log(f"❌ جستجوی «{q}» ناموفق: {ex}")

    for url in list(sources["urls"]) + list(extra_urls or []):
        if Path(url.split("?")[0]).suffix.lower() in AUDIO_EXT - {".webm", ".mp4", ".mkv"}:
            download_direct(url, city, kind="url", log=log)
        else:
            download_with_ytdlp(url, city, kind="youtube" if "youtu" in url else "web", log=log)

    for page in sources["iranseda_pages"]:
        download_iranseda(page, city, log=log)

    for folder in sources["local_dirs"]:
        import_local(folder, city, log=log)

    total = sum(s.duration for s in load_source_records())
    log(f"\n📦 مجموع منابع: {len(load_source_records())} فایل، {total/3600:.2f} ساعت صدای خام")
