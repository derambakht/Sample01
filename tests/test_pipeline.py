"""تست‌های بدون اینترنت و بدون مدل: نرمال‌سازی، زیرنویس، VAD، برش و تقسیم train/test."""

from __future__ import annotations

import numpy as np
import pytest

from persian_asr import audio
from persian_asr.text import clean_transcript, normalize_for_wer, wer_cer

SR = 16000


@pytest.fixture()
def data_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PASR_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


def speechlike(seconds: float, seed: int = 0) -> np.ndarray:
    """نویز مدوله‌شده شبیه هجاهای گفتار."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    env = 0.5 * (1 + np.sin(2 * np.pi * 4 * t)) ** 2
    return (0.3 * env * rng.standard_normal(len(t))).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return (0.001 * np.random.default_rng(1).standard_normal(int(seconds * SR))).astype(np.float32)


def test_normalizer():
    assert clean_transcript("ميخواي كتاب [موسیقی] بخري؟") == "میخوای کتاب بخری؟"
    assert normalize_for_wer("می‌خوام، دِگه!") == normalize_for_wer("می خوام دگه")
    assert normalize_for_wer("۱۲۳") == "123"
    m = wer_cer(["سلام خوبی", ""], ["سلام خوبی", "x"])
    assert m["wer"] == 0 and m["n"] == 1
    m = wer_cer(["سلام خوبی"], [""])
    assert m["wer"] == 1.0


def test_vad_and_grouping():
    wav = np.concatenate([silence(1), speechlike(3), silence(1.5), speechlike(2, 2), silence(1),
                          speechlike(30, 3), silence(1)])
    regions = audio.energy_vad(wav, SR)
    assert len(regions) == 3
    assert abs(regions[0][0] - 1.0) < 0.2
    pieces = []
    for a, b in regions:
        pieces += audio.split_long(wav, SR, a, b, 15)
    assert all(b - a <= 15 for a, b in pieces)
    groups = audio.group_regions(pieces, target_s=8, max_s=15, max_gap_s=2)
    assert all(b - a <= 15 for a, b in groups)


VTT = """WEBVTT
Kind: captions
Language: fa

00:00:01.000 --> 00:00:03.500 align:start position:0%
سلام<00:00:01.500><c> چطوری</c>

00:00:03.500 --> 00:00:03.510 align:start position:0%
سلام چطوری

00:00:03.510 --> 00:00:06.000 align:start position:0%
سلام چطوری
خوبم<00:00:04.000><c> مرسی</c>

00:00:07.000 --> 00:00:09.000
دیگه چه خبر
"""


def test_parse_vtt_rolling(tmp_path):
    from persian_asr.segment import group_cues, parse_vtt

    p = tmp_path / "a.vtt"
    p.write_text(VTT, encoding="utf-8")
    cues = parse_vtt(p)
    assert [c.text for c in cues] == ["سلام چطوری", "خوبم مرسی", "دیگه چه خبر"]
    g = group_cues(cues, target_s=4, max_s=10)
    assert g[0].text == "سلام چطوری خوبم مرسی" and len(g) == 2


def test_segment_and_split(data_env):
    from persian_asr.manifest import Source, load_clips, update_clip, upsert_source
    from persian_asr.segment import segment_all
    from persian_asr.split import assign_splits, export_audiofolder

    raw = data_env / "raw"
    raw.mkdir(parents=True)
    for i in range(5):
        wav = np.concatenate([silence(0.5)] + [np.concatenate([speechlike(4, i * 10 + k), silence(1)]) for k in range(6)])
        audio.save(raw / f"s{i}.wav", wav)
        src = Source(source_id=f"s{i}", title=f"source {i}", kind="local", audio_path=f"raw/s{i}.wav",
                     duration=len(wav) / SR)
        if i == 0:
            (raw / "s0.fa.vtt").write_text(VTT, encoding="utf-8")
            src.subs_path, src.subs_kind = "raw/s0.fa.vtt", "auto"
        upsert_source(src)

    n = segment_all(log=lambda m: None)
    clips = load_clips()
    assert n == len(clips) > 5
    assert any(c.text_source == "subtitle_auto" for c in clips)
    for c in clips:
        assert 2.0 <= c.duration <= 20.0
        update_clip(c.id, status="approved", text=c.text or "متن آزمایشی")

    stats = assign_splits(log=lambda m: None)
    assert stats["test"] > 0 and stats["train"] > 0
    clips = load_clips()
    test_src = {c.source_id for c in clips if c.split == "test"}
    train_src = {c.source_id for c in clips if c.split == "train"}
    assert not test_src & train_src  # هیچ منبعی در هر دو بخش نیست

    out = export_audiofolder(log=lambda m: None)
    meta = (data_env / "export" / "train" / "metadata.csv").read_text(encoding="utf-8")
    assert meta.startswith("file_name,transcription")
    assert out.endswith("export")


def test_training_config_roundtrip(tmp_path, monkeypatch):
    import importlib

    import persian_asr.config as config

    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    config.save_training_config({"learning_rate": "3e-5", "use_lora": "true", "model": "whisper-base"})
    cfg = config.load_training_config()
    assert cfg["learning_rate"] == 3e-5 and cfg["use_lora"] is True and cfg["model"] == "whisper-base"
    text = (tmp_path / "training.yaml").read_text(encoding="utf-8")
    assert "# نرخ یادگیری:" in text
    importlib.reload(config)
