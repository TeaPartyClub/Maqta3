# Maqta3 · مقطع

**Turn any video into vertical Arabic reels.** Paste a YouTube link or upload a video, and Maqta3 picks the best 20–60 second highlights, crops them to 9:16, burns in Arabic subtitles in Saudi dialect, and (for English videos) replaces the audio with an Arabic voiceover.

Everything runs locally in a small web app at `http://localhost:8000`.

---

## What it does

For each video, the pipeline:

1. **Downloads** the video with `yt-dlp` (or uses your uploaded file)
2. **Skips sponsor segments** using [SponsorBlock](https://sponsor.ajay.app/) (YouTube only)
3. **Transcribes** the speech — uses YouTube's own subtitles when available, otherwise [Whisper](https://github.com/openai/whisper)
4. **Scores highlight candidates** (20–60 s windows) by length, speech pace, keywords, and position, then picks the top N that don't overlap
5. **Jump-cuts** each clip to remove silence and **crops it to 1080×1920**
6. **Translates** to Arabic with [Helsinki-NLP/opus-mt-en-ar](https://huggingface.co/Helsinki-NLP/opus-mt-en-ar)
7. **Rewrites into Saudi colloquial dialect** with Claude (optional, needs an Anthropic API key)
8. **Burns in** right-to-left Arabic subtitles
9. **Generates an Arabic voiceover** with [Coqui XTTSv2](https://huggingface.co/coqui/XTTS-v2), cloned from a reference voice (English source videos only)

You get `highlight_1.mp4`, `highlight_2.mp4`, … which you can preview and download from the web page.

---

## Requirements

| | |
|---|---|
| **Python** | 3.11 recommended (some dependencies fail to build on 3.13) |
| **FFmpeg** | Must be on your `PATH`. Needs `libass` for subtitle burning. |
| **Disk** | ~3 GB free for the AI models (downloaded automatically on first run) |
| **RAM** | 8 GB minimum; 16 GB recommended |
| **GPU** | Optional. An NVIDIA GPU with CUDA makes voiceover generation much faster. |

### Install FFmpeg

- **Windows:** `winget install Gyan.FFmpeg`
  (or extract a build to `C:\ffmpeg` — `run.bat` adds `C:\ffmpeg\bin` to `PATH` automatically)
- **macOS:** `brew install ffmpeg-full` (includes `libass`; `run.sh` picks it up automatically). Plain `brew install ffmpeg` also works if it was built with `libass`.
- **Ubuntu / Debian:** `sudo apt install ffmpeg fonts-noto-core`

Check it works: `ffmpeg -version`

---

## Setup

### 1. Clone the repo

```bash
git clone git@github.com:TeaPartyClub/Maqta3.git
cd Maqta3
```

### 2. Add your API keys

Copy the example file and fill it in:

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

| Variable | Required? | What it's for | Where to get it |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Optional | Rewrites subtitles into Saudi dialect. Without it you get Modern Standard Arabic instead. | [console.anthropic.com](https://console.anthropic.com/settings/keys) |
| `HF_TOKEN` | Optional | Faster / more reliable model downloads from Hugging Face | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (read access) |

No quotes, no spaces around `=`. **Never commit `.env`** — it's already in `.gitignore`.

### 3. Add a reference voice (for the Arabic voiceover)

The voiceover clones the voice from a sample recording, and no voice ships with the repo — you bring your own. Drop at least one `.wav` file into the empty `backend/dataset/wavs/` folder:

```
backend/dataset/wavs/your_voice_sample.wav
```

A clean 6–15 second clip of a single Arabic speaker (no music or background noise) works best. The first `.wav` in the folder is used, and the voiceover will sound like that speaker. `.wav` files in this folder are gitignored, so your recordings stay local.

> Without this, everything still works — you'll just get subtitled clips with the original audio instead of an Arabic voiceover.

### 4. Run it

**Windows** — double-click `run.bat`, or:

```bat
run.bat
```

**macOS**:

```bash
./run.sh
```

The script installs the Python dependencies, loads your `.env`, starts the server, and opens `http://localhost:8000`.

> **Tip (Windows):** `run.bat` installs into your global Python. To keep things isolated, create a virtual environment first:
> ```bat
> cd backend
> py -3.11 -m venv .venv
> .venv\Scripts\activate
> cd ..
> run.bat
> ```

**Linux / manual start:**

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
set -a; . ../.env; set +a
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open <http://localhost:8000>.

---

## Using it

1. Paste a **YouTube URL** or switch to **Upload Video** and pick a file (MP4 / MOV / AVI)
2. Choose how many highlights you want (1–5)
3. Click **Generate Arabic Highlights**
4. Watch the progress, then preview or download the finished clips

**Expect it to be slow the first time.** The first job downloads Whisper, the translation model, and XTTSv2 (~2–3 GB total). After that, a typical video takes 3–8 minutes depending on length and hardware.

Output files are saved in `backend/results/<job-id>/`.

---

## Project structure

```
Maqta3/
├── Maqta3.html           # Web UI
├── Maqta3Script.js       # Front-end logic (submit job, poll progress, show results)
├── Maqta3Style.css       # Styles
├── run.bat / run.sh      # One-click launchers (Windows / macOS)
├── .env.example          # Template for your API keys
└── backend/
    ├── main.py           # FastAPI server: serves the UI and the /api/jobs endpoints
    ├── pipeline.py       # The video → Arabic reels pipeline
    ├── requirements.txt
    ├── dataset/wavs/     # (you add a .wav) reference voice for TTS
    ├── uploads/          # (generated) uploaded videos
    ├── jobs/             # (generated) job status JSON
    └── results/          # (generated) finished clips
```

### API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/jobs` | Start a job. Form fields: `url` **or** `file`, plus `num_highlights` (default 5). Returns `{ "job_id": ... }` |
| `GET` | `/api/jobs/{job_id}` | Job status, progress %, and results when done |
| `GET` | `/api/results/{job_id}/{filename}` | Download a finished clip |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ffmpeg` not found | Install FFmpeg and make sure `ffmpeg -version` works in a new terminal. |
| Error mentioning `ass` filter / subtitles | Your FFmpeg build lacks `libass`. On macOS use `brew install ffmpeg-full`; on Windows use the Gyan.dev "full" build. |
| Subtitles are in formal Arabic, not Saudi dialect | `ANTHROPIC_API_KEY` is missing or invalid — check your `.env`. The server log will say `Saudi dialect rewrite will be skipped`. |
| No Arabic voiceover | Either the source video isn't English (voiceover only runs for English videos), or there's no `.wav` in `backend/dataset/wavs/`. Check the server log for `TTS failed`. |
| `pip install` fails building a package | You're probably on Python 3.13. Use Python 3.11. |
| YouTube download fails | Update yt-dlp: `pip install -U yt-dlp` |
| "Could not find suitable highlight windows" | The video is too short or mostly silent. Try a longer video with continuous speech. |
