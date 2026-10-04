# Spec: Build the Isfahani Persian dataset and fine-tune Whisper (local run)

> **How to use:** from the repo root run
> `claude "Read docs/SPEC_LOCAL_RUN.md and execute it phase by phase. Stop at every STOP point."`

## 1. Goal

Produce a classroom demo that shows, with numbers, that fine-tuning Whisper on Persian speech with an
**Isfahani accent** improves recognition. The finished demo has:

1. A dataset of **≥ 300 approved clips (≥ 45 min)**, ideally 800+ (1.5 h+), from **≥ 15 different sources/speakers**.
   Every clip in the **test** split has a transcript a human has checked.
2. A fine-tuned `whisper-small` (or a smaller model if the hardware can't handle small) saved in `outputs/<run>/final`.
3. An evaluation report (`reports/latest.json` + `reports/latest.csv`) comparing base and fine-tuned WER/CER.
4. A working panel (`python -m persian_asr.cli panel`) that shows all of the above.
5. A short `docs/RESULTS.md` with the numbers and 5–10 sentence-level examples where the fine-tuned model fixed accent words.

## 2. Context: what already exists

This project was written in a sandbox with **no access to YouTube, iranseda.ir or HuggingFace**. So:

- **Tested in the sandbox:** the unit tests (`pytest -q`), plus segmentation, labeling, split/export, full and LoRA training, and evaluation. All of these ran on synthetic audio with a tiny randomly initialized Whisper. The Gradio panel was also tested: all 5 tabs rendered without JS errors.
- **Never tested against the real world:**
  - real downloads with yt-dlp (search, subtitles, playlists)
  - the iranseda.ir audio-link scraper (`persian_asr/download.py::find_audio_links`)
  - real Whisper checkpoints from HuggingFace
  - GPU training (fp16, gradient checkpointing)
  - real training/eval speed
  - Persian font rendering in the panel (Vazirmatn comes from Google Fonts)

Read `README.md` (Persian) and these files before starting:
`persian_asr/cli.py`, `download.py`, `segment.py`, `label.py`, `split.py`, `train.py`, `evaluate.py`, `config/*.yaml`.

Pipeline:
`download → segment → label → human review (panel) → split → train → evaluate`

Data lives in `data/manifest.jsonl` (one clip per line) and `data/sources.jsonl`.

## 3. Ground rules

- **Never commit** `data/`, `outputs/` or `reports/`. They are gitignored; keep it that way.
- Run `pytest -q` before every commit. Never skip, delete or weaken a test to get green.
- Commit after each phase with a clear message. Push only if the user agrees.
- Keep user-facing strings (CLI logs, panel, config comments) in **Persian**. Code identifiers and comments stay as they are.
- **Never invent or "clean up" transcripts with an LLM and mark them approved.** Approval means a human listened to the clip. You may pre-fill text, flag clips and sort them, but `status=approved` is set only by the human in the panel. The one exception is `add_recording`, where the human typed the text.
- Before any step that runs over ~30 minutes (bulk download, labeling with large-v3, training), say how long it should take and how much disk it needs, then **ask the user to confirm**.
- Never delete `data/manifest.jsonl` or re-run `segment --force` without asking. Both destroy review work.
- If a site blocks you (bot check, 403, geo-block), say so. Then try the documented workaround: `pip install -U yt-dlp`, then `--cookies-from-browser`, then a proxy via `HTTPS_PROXY`. Don't hammer the site.

## 4. Phases

### Phase 0: Set up the environment

1. Report the OS, Python version (needs ≥ 3.10), `ffmpeg -version`, GPU (`nvidia-smi` / Apple MPS / none) and free disk space (needs ≥ 20 GB).
2. Create a `.venv`.
   - With an NVIDIA GPU, install the CUDA build of torch from pytorch.org first.
   - Then run `pip install -r requirements.txt`.
3. Run `pytest -q`. It must pass.
4. Check that the network can reach:
   - YouTube: `yt-dlp --simulate "ytsearch1:لهجه اصفهانی"`
   - HuggingFace: download `openai/whisper-tiny`
   - iranseda.ir: `curl -I https://radio.iranseda.ir`

   Report which ones fail.

**Done when:** tests pass and you've reported which sources are reachable.
**STOP if** there is no GPU. Then propose the hardware plan to the user: `whisper-base` locally on CPU, or Colab with a T4 for `whisper-small`.

### Phase 1: Smoke test with real models

1. Use a throwaway data dir: `PASR_DATA_DIR=/tmp/pasr_smoke`.
2. Download 1–2 short YouTube videos with `download --url ...`.
3. Run `segment`, then `label --model openai/whisper-tiny --limit 20`.
4. Mark ~20 clips approved with a small script. This is the **smoke test only**; the real data dir is never touched.
5. Run `split`, then `train --model whisper-tiny --set max_steps=30 --set eval_steps=15 --set save_steps=15`, then `evaluate`.
6. Start the panel and check every tab against real data.
7. Fix any bug you find. Add a regression test for it if you can do that without network access.

**Done when:** the whole pipeline runs end to end on real audio and real checkpoints.

### Phase 2: Collect sources

1. Check the search queries in `config/sources.yaml`. For each query, list the top results (title, channel, duration) without downloading.
2. Find **stable, high-quality Isfahani sources** and add them to `urls` as channels or playlists. Good candidates:
   - street interviews filmed in Isfahan
   - Isfahan provincial TV (شبکه اصفهان) talk shows
   - Isfahani vloggers and storytellers
   - Isfahani comedians who actually speak with the accent

   Prefer one speaker per video and little background music.
3. **iranseda:**
   - Find program pages for رادیو اصفهان.
   - Test `find_audio_links` on them. If the page structure doesn't work with the regex (for example, links built by JS or behind an API), fix the scraper: inspect the page and use the site's JSON endpoints if they exist. Add an offline unit test with a saved HTML fixture.
   - If iranseda can't be reached, document that and skip it.
4. Show the user the source list: title, URL, why it's Isfahani, estimated minutes.

**STOP:** wait for the user to approve the list or remove items. Then run `download`.

**Done when:** `stats` shows **≥ 6–8 hours** of raw audio from **≥ 20 sources**. About 30–50% of raw audio survives segmentation and review.

### Phase 3: Segment and check

1. Run `segment`. Then report:
   - clip count
   - clip length histogram
   - share of clips cut from subtitles vs. VAD
2. If many clips are under 3 s or hit the 20 s cap, tune `segmentation.vad` in `config/dataset.yaml`. Re-run only on new sources, or with `--force` **after asking**.
3. Optional: add an option to use `silero-vad` when it's installed, with fallback to the current energy VAD. Keep it behind a config flag and add a test.

### Phase 4: Pre-label and auto-flag

1. Run `label` with `openai/whisper-large-v3` if VRAM is ≥ 10 GB, otherwise `openai/whisper-large-v3-turbo`. Use `--replace-auto-subs`, since YouTube's Persian auto-captions are poor.
2. **New feature: an automatic quality flagger.**
   - Add `python -m persian_asr.cli flag`. It writes `extra.flags` (a list of reasons) on suspicious clips and **never changes `status`**. Heuristics:
     - empty text
     - characters per second outside 5–25
     - repeated n-gram hallucinations (the same 3+ words repeated 3+ times)
     - mostly Latin letters
     - text that doesn't look Persian
     - clip RMS too low
     - a very high share of non-speech frames (likely music)
   - In the panel's review tab, add a filter "مشکوک" (flagged) and show the flags in the info line.
   - Add unit tests for every heuristic.
3. Order the review queue so **test-candidate sources come first**. Run `split` once before review, using a provisional split that includes pending clips, so the user knows which clips need careful review.
   - Implement this as a `--provisional` flag on `split`.
   - The final split rule doesn't change: test uses approved clips only.

### Phase 5: Human review

**STOP.** Tell the user exactly what to do:

1. Open the panel and go to the "بررسی نمونه‌ها" tab.
2. Review the test candidates first, then the rest.
3. Fix each transcript so it matches the speech exactly. Keep dialect spellings, using one convention throughout (see the README).
4. Reject music, overlapping speakers, non-Isfahani speech and unclear audio.
5. Target: ≥ 300 approved clips, ideally 800+.

While the user reviews, you can do the optional improvements in Section 6. Resume when the user says review is done. Then run `stats` and confirm the targets are met.

### Phase 6: Train and evaluate

1. Run `split --export`. Check that train and test share no source, and that test is about 20% of approved duration.
2. **Baseline first:** run `evaluate` with only the base model. The base WER on Isfahani test clips is the "before" number.
3. Pick the training config from the hardware:

   | Hardware | Model | Settings |
   |---|---|---|
   | T4 / 16 GB | `whisper-small` | full fine-tune, lr 1e-5, batch 8 × accumulation 2, fp16, gradient checkpointing, 5 epochs, early stopping 3 |
   | ≥ 24 GB | `whisper-small` full, or `whisper-medium` with LoRA | — |
   | CPU only | `whisper-base` | 3 epochs |

   Tell the user the estimated time and **ask for confirmation**.
4. Run `train`, keeping an eye on `train_log.jsonl`. If eval WER rises for 2+ evals while train loss keeps falling (overfitting), say so. Then propose fewer epochs, a lower lr, or `freeze_encoder`.
5. Run `evaluate` with both models, using `--beams 1` and then `--beams 5` for the final numbers.
6. **Sanity check that general Persian didn't regress:**
   - Ask the user for 20–30 clips of standard (Tehrani) Persian. They can record them in the panel under a separate speaker id, or provide a folder.
   - Add an `evaluate --subset <source_prefix>` option, or a separate eval manifest.
   - Report base vs. fine-tuned on that subset as well.

**Done when:** `reports/latest.json` exists, the panel's test tab shows the comparison, and the numbers are believable (fine-tuned WER < base WER on the Isfahani test set).

### Phase 7: Classroom deliverables

1. Write `docs/RESULTS.md` covering:
   - dataset stats (sources, minutes, clips per split)
   - hardware and training time
   - base vs. fine-tuned WER/CER on the Isfahani test set and on the standard-Persian check set
   - relative improvement
   - 5–10 examples from `reports/latest.csv` (reference / base / fine-tuned), picking ones where accent words were fixed
   - one or two failure cases
   - honest limitations: small test set, possible label noise, some speakers on YouTube may not be truly Isfahani
2. Write `docs/CLASS_DEMO.md` (in Persian): a 10-minute live demo script built around the panel tabs. Cover the dashboard, playing a clip, the live test tab comparing two models on a test clip and on a live recording, and the training curves.
3. Optional: `notebooks/colab_train.ipynb` that clones the repo, installs requirements, mounts Drive as `PASR_DATA_DIR`, and runs split, train and evaluate.
4. Run `pytest -q`, commit, and ask the user before pushing.

## 5. Acceptance checklist

- [ ] `pytest -q` passes (including new tests for the flagger and any scraper fix)
- [ ] ≥ 300 approved clips, ≥ 15 sources, and every test clip human-approved
- [ ] No source appears in both train and test
- [ ] `outputs/<run>/final` loads in the panel's live-test tab
- [ ] `reports/latest.json` shows base vs. fine-tuned, and fine-tuned is better on the Isfahani test set
- [ ] A standard-Persian regression check is reported
- [ ] `docs/RESULTS.md` and `docs/CLASS_DEMO.md` are written
- [ ] No data, models or reports are committed

## 6. Optional improvements (only after the core is done, or during Phase 5)

- Keyboard shortcuts in the review tab: Enter = approve and go to next, Esc = reject.
- `push-to-hub` command: upload `data/export` as a private HF dataset with a README card.
- Resume interrupted downloads, and a per-source "max minutes" cap so one long video can't dominate.
- Speaker-level grouping for recordings, so `rec_<speaker>` clips are treated as one source in the split. This already works through `source_id`; add a test for it.
- Show elapsed time and ETA for the running job in the panel.
