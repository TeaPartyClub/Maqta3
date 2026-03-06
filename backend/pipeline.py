"""
Maqta3 — Video Processing Pipeline
------------------------------------
Handles:
  1. Video download (yt-dlp) or local file passthrough
  2. Audio extraction (FFmpeg)
  3. Speech transcription + language detection (Whisper / youtube-transcript-api)
  4. Highlight extraction using heuristic scoring
  5. Vertical 9:16 center-crop (FFmpeg)
  6. Arabic subtitle generation via translation (Helsinki-NLP/opus-mt-en-ar)
  7. Saudi dialect rewrite via LLM (Anthropic Claude) — converts MSA → Saudi colloquial
  8. Subtitle burning into video (FFmpeg ASS filter)
  9. Arabic TTS audio replacement (Coqui XTTSv2, auto-downloaded; voice-cloned from dataset WAVs)
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import Callable, List, Optional

log = logging.getLogger("maqta3")

# ─── Speaker reference for XTTSv2 voice cloning ───────────────────────────────
# XTTSv2 is downloaded automatically via the TTS package on first use.
# The dataset WAVs are used as the speaker reference (voice cloning).
_BACKEND_DIR     = Path(__file__).parent
_XTTS_WAV_DIR    = _BACKEND_DIR / "dataset" / "wavs"

import numpy as np
import requests
import soundfile as sf

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    YouTubeTranscriptApi = None

# ─── Arabic font selection ────────────────────────────────────────────────────
# The ASS renderer picks the best system fallback if the named font is missing.
ARABIC_FONT = "Noto Naskh Arabic" if platform.system() != "Windows" else "Arabic Typesetting"

# ─── Highlight scoring keywords ───────────────────────────────────────────────
_KW_EN = {
    "important", "amazing", "incredible", "never", "always", "secret",
    "mistake", "best", "worst", "how to", "why", "discovered", "revealed",
    "truth", "actually", "surprising", "shocking", "critical", "essential",
    "key", "major", "powerful", "proven", "study", "research", "wrong",
    "everyone", "nobody", "stop", "warning", "danger", "free", "money",
    "biggest", "reason", "number one", "top", "never do", "must",
}
_KW_AR = {
    "مهم", "رائع", "سر", "خطأ", "أفضل", "أسوأ", "كيف", "لماذا",
    "حقيقة", "مفاجئ", "أساسي", "قوي", "دراسة", "الجميع", "خطورة",
    "لن تصدق", "الحل", "التحذير", "المشكلة", "النتيجة",
}
KEYWORDS = _KW_EN | _KW_AR

# ─── ASS subtitle file template ───────────────────────────────────────────────
_ASS_HEADER = (
    "[Script Info]\n"
    "ScriptType: v4.00+\n"
    "PlayResX: 1080\n"
    "PlayResY: 1920\n"
    "ScaledBorderAndShadow: yes\n\n"
    "[V4+ Styles]\n"
    "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
    "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,"
    "BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
    f"Style: Arabic,{ARABIC_FONT},72,&H00FFFFFF,&H000000FF,&H00000000,"
    "&H80000000,1,0,0,0,100,100,0,0,1,3,1,2,30,30,70,1\n\n"
    "[Events]\n"
    "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
)


def _secs_to_ass(t: float) -> str:
    """Convert seconds → ASS timestamp H:MM:SS.cc"""
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess, raise with stderr on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd[:4])}\n"
            f"stderr: {result.stderr[-800:]}"
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
class VideoProcessor:
    """
    All ML models are loaded lazily on first use so the FastAPI server starts
    fast and models are only pulled into memory if a job actually needs them.
    """

    def __init__(self):
        self._whisper           = None
        self._mt_tokenizer      = None
        self._mt_model          = None
        self._llm_client        = None
        self._xtts_model        = None   # TTS.api.TTS instance

    # ── Lazy loaders ─────────────────────────────────────────────────────────

    def _load_whisper(self):
        if self._whisper is None:
            import whisper
            log.info("Loading Whisper 'base' model…")
            t0 = time.time()
            self._whisper = whisper.load_model("base")
            log.info("Whisper loaded in %.1fs", time.time() - t0)
        return self._whisper

    def _load_mt(self):
        """Helsinki-NLP English → Arabic machine-translation model."""
        if self._mt_model is None:
            from transformers import MarianMTModel, MarianTokenizer
            name = "Helsinki-NLP/opus-mt-en-ar"
            log.info("Loading Helsinki-NLP MT model (%s)…", name)
            t0 = time.time()
            self._mt_tokenizer = MarianTokenizer.from_pretrained(name)
            self._mt_model     = MarianMTModel.from_pretrained(name)
            log.info("MT model loaded in %.1fs", time.time() - t0)
        return self._mt_tokenizer, self._mt_model

    def _load_llm(self):
        """Anthropic Claude client for Saudi dialect rewriting.
        Requires ANTHROPIC_API_KEY environment variable.
        """
        if self._llm_client is not None:
            return None if self._llm_client == "disabled" else self._llm_client

        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            log.warning("ANTHROPIC_API_KEY not set — Saudi dialect rewrite will be skipped")
            self._llm_client = "disabled"
            return None
        self._llm_client = anthropic.Anthropic(api_key=api_key)
        log.info("Anthropic client ready")
        return self._llm_client

    def _load_xtts(self):
        """Download (once) and return the XTTSv2 TTS instance.
        The model is cached by the TTS package in ~/.local/share/tts/ on first run.
        """
        if self._xtts_model is not None:
            return self._xtts_model

        import torch

        # Auto-accept Coqui license so the server doesn't block on an interactive prompt
        os.environ.setdefault("COQUI_TOS_AGREED", "1")

        # Compatibility shim: transformers ≥4.41 removed isin_mps_friendly but
        # coqui-tts 0.28.x still imports it from transformers.pytorch_utils.
        try:
            from transformers.pytorch_utils import isin_mps_friendly  # noqa: F401
        except ImportError:
            import transformers.pytorch_utils as _pt_utils
            _pt_utils.isin_mps_friendly = lambda elements, test_elements: torch.isin(
                elements, test_elements
            )

        from TTS.api import TTS

        gpu = torch.cuda.is_available()
        log.info("Loading XTTSv2 model (gpu=%s) — first run downloads ~1.8 GB…", gpu)
        t0 = time.time()
        self._xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=gpu)
        log.info("XTTSv2 ready in %.1fs", time.time() - t0)
        return self._xtts_model

    # ── Public entry point ────────────────────────────────────────────────────

    def process(
        self,
        job_id:            str,
        source_url:        Optional[str],
        source_path:       Optional[str],
        result_dir:        str,
        num_highlights:    int,
        progress_callback: Callable[[str, int, str], None],
    ) -> List[dict]:

        work = Path(result_dir)
        log.info("=== Job %s started (url=%s, clips=%d) ===",
                 job_id, source_url or "upload", num_highlights)
        job_t0 = time.time()

        # 1 ── Obtain video file
        progress_callback("downloading", 5, "Downloading video…")
        log.info("[1/9] Downloading video…")
        video = self._get_video(source_url, source_path, work)

        video_id = self._extract_video_id(source_url)
        skip_segments = []
        if video_id:
            skip_segments = self._get_sponsorblock_segments(video_id)
            if skip_segments:
                log.info("SponsorBlock: skipping %d segment(s)", len(skip_segments))

        # Get total duration
        total_dur = self._duration(video)
        log.info("Video duration: %.1fs", total_dur)

        transcript = None
        if video_id:
            progress_callback("transcribing", 15, "Fetching YouTube subtitles…")
            log.info("[2/9] Trying YouTube transcript API…")
            transcript = self._get_youtube_transcript(video_id)
            if transcript:
                log.info("YouTube transcript fetched (%d segments, lang=%s)",
                         len(transcript.get("segments", [])),
                         transcript.get("language", "?"))

        if not transcript:
            progress_callback("extracting", 12, "Extracting audio…")
            log.info("[2/9] Extracting audio for Whisper…")
            audio = self._extract_audio(video, work)

            progress_callback("transcribing", 18,
                              "Transcribing speech… (1–3 min depending on length)")
            log.info("[3/9] Transcribing with Whisper (this takes a while)…")
            t0 = time.time()
            transcript = self._transcribe(audio)
            log.info("Whisper transcription done in %.1fs — detected language: %s",
                     time.time() - t0, transcript.get("language", "?"))

        language   = transcript.get("language", "en")
        raw_segments = transcript.get("segments", [])

        segments = []
        for s in raw_segments:
            is_sponsored = any(
                not (s["end"] <= skip[0] or s["start"] >= skip[1])
                for skip in skip_segments
            )
            if not is_sponsored and s.get("text", "").strip():
                segments.append(s)

        skipped = len(raw_segments) - len(segments)
        log.info("Segments after filtering: %d usable, %d skipped (sponsored/empty)",
                 len(segments), skipped)

        if not segments:
            raise ValueError(
                "No valid speech segments found in this video. "
                "Try a video with clearer audio."
            )

        # 5 ── Highlight scoring
        progress_callback("scoring", 55, "Scoring highlight candidates…")
        log.info("[4/9] Scoring highlight candidates…")
        highlights = self._extract_highlights(segments, total_dur, num_highlights)

        if not highlights:
            raise ValueError(
                "Could not find suitable highlight windows (20–60 s). "
                "The video may be too short or consist mostly of silence."
            )

        log.info("Selected %d highlight(s):", len(highlights))
        for idx, hl in enumerate(highlights):
            log.info("  Clip %d: %.1fs – %.1fs (%.1fs, score=%.1f)",
                     idx + 1, hl["start"], hl["end"],
                     hl["end"] - hl["start"], hl["score"])

        results = []
        n = len(highlights)

        for i, hl in enumerate(highlights):
            base = 58 + int(i / n * 38)
            clip_t0 = time.time()
            log.info("── Clip %d/%d ──────────────────────────────", i + 1, n)

            # 6 ── Trim raw clip (Silence Removal Jumpcut)
            progress_callback("clipping", base,
                              f"Cutting and jump-cutting clip {i+1}/{n}…")
            log.info("[5/9] Jump-cutting clip %d (%d segments)…",
                     i + 1, len(hl["segments"]))
            raw_clip = work / f"_raw_{i}.mp4"
            retimed_segs = self._trim_jumpcut(video, hl["segments"], raw_clip)
            hl["segments"] = retimed_segs
            clip_start = 0.0

            # 7 ── Convert to 9:16
            progress_callback("converting", base + 1,
                              f"Converting clip {i+1} to vertical…")
            log.info("[6/9] Converting clip %d to vertical 9:16…", i + 1)
            vert_clip = work / f"_vert_{i}.mp4"
            self._make_vertical(raw_clip, vert_clip)

            # 8 ── Translate subtitle segments to Arabic (Helsinki-NLP MSA)
            progress_callback("subtitling", base + 2,
                              f"Generating Arabic subtitles for clip {i+1}…")
            log.info("[7/9] Translating %d segment(s) to Arabic…",
                     len(hl["segments"]))
            t0 = time.time()
            ar_segs = self._translate_segments(hl["segments"], language)
            log.info("Translation done in %.1fs", time.time() - t0)

            # 8b ── Rewrite Arabic to Saudi colloquial dialect via LLM
            progress_callback("dialect", base + 3,
                              f"Rewriting to Saudi dialect for clip {i+1}…")
            log.info("[8/9] Rewriting %d segment(s) to Saudi dialect via LLM…",
                     len(ar_segs))
            t0 = time.time()
            ar_segs = self._dialectify_segments(ar_segs)
            log.info("Dialect rewrite done in %.1fs", time.time() - t0)

            # 9 ── Write ASS file
            ass_file = work / f"_subs_{i}.ass"
            self._write_ass(ar_segs, clip_start, ass_file)

            # 10 ── Burn subtitles
            sub_clip = work / f"_sub_{i}.mp4"
            self._burn_subtitles(vert_clip, ass_file, sub_clip)

            final = sub_clip

            # 11 ── TTS audio replacement (English → Arabic only)
            if language == "en":
                progress_callback("tts", base + 4,
                                  f"Generating Arabic voiceover for clip {i+1}…")
                log.info("[9/9] Synthesising voiceover for clip %d (%d segment(s))…",
                         i + 1, len(ar_segs))
                t0 = time.time()
                try:
                    tts_clip = work / f"_tts_{i}.mp4"
                    self._replace_audio_tts(sub_clip, ar_segs,
                                            clip_start, tts_clip)
                    final = tts_clip
                    log.info("TTS done in %.1fs", time.time() - t0)
                except Exception as exc:
                    log.error("TTS failed for clip %d, keeping subtitles-only: %s",
                              i + 1, exc)

            # Move to permanent name
            final_name = f"highlight_{i+1}.mp4"
            final_path = work / final_name
            shutil.move(str(final), str(final_path))

            # Cleanup temp files
            for tmp in (raw_clip, vert_clip, sub_clip):
                if tmp.exists() and tmp != final_path:
                    tmp.unlink(missing_ok=True)
            ass_file.unlink(missing_ok=True)

            arabic_preview = " ".join(
                s.get("arabic", "") for s in ar_segs[:3]
            )[:150]

            size_mb = final_path.stat().st_size / 1_048_576
            log.info("Clip %d done in %.1fs → %s (%.1f MB)",
                     i + 1, time.time() - clip_t0, final_name, size_mb)

            results.append({
                "clip_index":    i + 1,
                "filename":      final_name,
                "start":         round(hl["start"], 2),
                "end":           round(hl["end"], 2),
                "duration":      round(hl["end"] - hl["start"], 2),
                "arabic_preview": arabic_preview,
            })

        log.info("=== Job %s complete — %d clip(s) in %.1fs ===",
                 job_id, len(results), time.time() - job_t0)
        return results

    # ── Step implementations ──────────────────────────────────────────────────

    def _get_video(
        self,
        url: Optional[str],
        path: Optional[str],
        work: Path,
    ) -> Path:
        if path and Path(path).exists():
            return Path(path)

        if url:
            out = work / "source.mp4"
            _run([
                "python", "-m", "yt_dlp",
                "--format",
                "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]"
                "/best[ext=mp4]/best",
                "--merge-output-format", "mp4",
                "--output",              str(out),
                "--no-playlist",
                url,
            ])
            return out

        raise ValueError("No video source provided.")

    def _extract_audio(self, video: Path, work: Path) -> Path:
        audio = work / "audio.wav"
        _run([
            "ffmpeg", "-y",
            "-i",     str(video),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar",     "16000",
            "-ac",     "1",
            str(audio),
        ])
        return audio

    def _duration(self, video: Path) -> float:
        r = _run([
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(video),
        ])
        return float(json.loads(r.stdout)["format"]["duration"])

    def _transcribe(self, audio: Path) -> dict:
        model = self._load_whisper()
        return model.transcribe(str(audio), task="transcribe", fp16=False)

    def _extract_video_id(self, url: Optional[str]) -> Optional[str]:
        if not url: return None
        try:
            parsed = urllib.parse.urlparse(url)
            if "youtube.com" in parsed.netloc:
                qs = urllib.parse.parse_qs(parsed.query)
                return qs.get("v", [None])[0]
            elif "youtu.be" in parsed.netloc:
                return parsed.path.lstrip("/")
        except:
            pass
        return None

    def _get_sponsorblock_segments(self, video_id: str) -> list[tuple[float, float]]:
        try:
            url = f"https://sponsor.ajay.app/api/skipSegments?videoID={video_id}&categories=[\"sponsor\",\"intro\",\"outro\",\"interaction\",\"selfpromo\",\"music_offtopic\"]"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return [(seg["segment"][0], seg["segment"][1]) for seg in data]
        except Exception as e:
            log.warning("SponsorBlock lookup failed: %s", e)
        return []

    def _get_youtube_transcript(self, video_id: str) -> Optional[dict]:
        if YouTubeTranscriptApi is None:
            return None
        try:
            # youtube-transcript-api v1.x uses instance-based API
            api = YouTubeTranscriptApi()
            transcript_list = api.list(video_id)
            try:
                transcript = transcript_list.find_transcript(['en', 'ar'])
            except Exception:
                transcript = transcript_list.find_transcript(
                    [t.language_code for t in transcript_list]
                )

            raw_transcript = transcript.fetch()
            language = transcript.language_code

            segments = []
            for item in raw_transcript:
                # v1.x items expose attributes; fall back to dict access for older builds
                start    = getattr(item, "start",    None) or item["start"]
                duration = getattr(item, "duration", None) or item["duration"]
                text     = getattr(item, "text",     None) or item["text"]
                segments.append({
                    "start": start,
                    "end":   start + duration,
                    "text":  text,
                })

            return {
                "language": "en" if "en" in language else ("ar" if "ar" in language else language),
                "segments": segments,
            }
        except Exception as e:
            log.warning("YouTube transcript unavailable (%s) — falling back to Whisper", e)
            return None

    # ── Highlight extraction ──────────────────────────────────────────────────

    def _extract_highlights(
        self,
        segments:      list,
        total_dur:     float,
        n:             int,
        min_dur:       float = 20.0,
        max_dur:       float = 60.0,
    ) -> list:
        candidates = []
        seg_len = len(segments)

        for i in range(seg_len):
            for j in range(i + 1, seg_len + 1):
                window = segments[i:j]
                start  = window[0]["start"]
                end    = window[-1]["end"]
                dur    = end - start

                if dur < min_dur:
                    continue
                if dur > max_dur:
                    break   # windows only grow; skip rest of inner loop

                score = self._score(window, total_dur)
                candidates.append(
                    {"segments": window, "start": start, "end": end, "score": score}
                )

        # Sort by score, greedily pick non-overlapping windows
        candidates.sort(key=lambda x: x["score"], reverse=True)

        selected: list = []
        for c in candidates:
            if not any(
                c["start"] < s["end"] and c["end"] > s["start"]
                for s in selected
            ):
                selected.append(c)
                if len(selected) >= n:
                    break

        return sorted(selected, key=lambda x: x["start"])

    def _score(self, segments: list, total_dur: float) -> float:
        """
        Heuristic highlight score.  Higher = better highlight candidate.

        Factors (all simple, no ML):
          • Duration sweet spot         25 pts
          • Word density (speech pace)  15 pts
          • Sentence completeness        8 pts
          • Keyword presence            ≤24 pts
          • Position in video            8 pts
          • Segment coherence            5 pts
        Max ≈ 85 pts
        """
        start    = segments[0]["start"]
        end      = segments[-1]["end"]
        dur      = end - start
        text     = " ".join(s["text"] for s in segments)
        score    = 0.0

        # Duration sweet spot
        if 25 <= dur <= 45:
            score += 25
        elif min(dur, 60) >= 20:
            score += 10

        # Word density
        wps = len(text.split()) / max(dur, 1)
        if 1.0 <= wps <= 3.5:
            score += 15
        elif wps < 0.5:
            score -= 20   # heavy silence penalty

        # Sentence completeness (last segment ends a sentence)
        if segments[-1]["text"].strip()[-1:] in ".?!؟":
            score += 8

        # Keyword presence
        tl = text.lower()
        hits = sum(1 for kw in KEYWORDS if kw in tl)
        score += min(hits * 4, 24)

        # Positional: avoid first 5% (usually intro) and last 5% (outro)
        if total_dur > 0:
            rel = start / total_dur
            if 0.05 < rel < 0.90:
                score += 8
            elif rel < 0.05:
                score -= 12

        # Coherence: prefer longer runs of connected sentences
        if len(segments) >= 4:
            score += 5

        return score

    # ── Video processing ──────────────────────────────────────────────────────

    def _trim_jumpcut(
        self,
        video:  Path,
        segments: list,
        output: Path,
    ) -> list:
        if not segments:
            raise ValueError("No segments to trim")
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            for seg in segments:
                f.write(f"file '{str(video.absolute().as_posix())}'\n")
                f.write(f"inpoint {seg['start']:.3f}\n")
                f.write(f"outpoint {seg['end']:.3f}\n")
            concat_file = f.name
            
        try:
            _run([
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-c:v", "libx264", "-preset", "fast",
                "-c:a", "aac",
                str(output),
            ])
        finally:
            Path(concat_file).unlink(missing_ok=True)
            
        current_time = 0.0
        retimed_segments = []
        for seg in segments:
            dur = seg["end"] - seg["start"]
            if dur <= 0:
                continue
            retimed_segments.append({
                "start": current_time,
                "end": current_time + dur,
                "text": seg["text"]
            })
            current_time += dur
            
        return retimed_segments

    def _make_vertical(self, src: Path, dst: Path):
        """
        Center-crop any aspect ratio to 9:16 (1080×1920).

        Filter chain:
          scale=iw*max(1080/iw, 1920/ih):ih*max(1080/iw, 1920/ih)
            — scale up so both 1080 wide and 1920 tall thresholds are met
          crop=1080:1920
            — take the center 1080×1920 region
        """
        vf = (
            "scale=iw*max(1080/iw\\,1920/ih):ih*max(1080/iw\\,1920/ih),"
            "crop=1080:1920"
        )
        _run([
            "ffmpeg", "-y",
            "-i",     str(src),
            "-vf",    vf,
            "-c:v",   "libx264", "-preset", "fast",
            "-c:a",   "copy",
            str(dst),
        ])

    # ── Subtitle generation ───────────────────────────────────────────────────

    def _translate_segments(self, segments: list, language: str) -> list:
        """
        Translate each segment's text to Arabic.
        If source language is already Arabic → pass through unchanged.
        """
        if language in ("ar", "arabic"):
            return [
                {"start": s["start"], "end": s["end"],
                 "text": s["text"], "arabic": s["text"]}
                for s in segments
            ]

        tokenizer, model = self._load_mt()
        texts = [s["text"].strip() for s in segments]

        # Batch translate to avoid OOM on long clips
        arabic_texts = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch   = texts[i : i + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            translated = model.generate(**encoded)
            for t in translated:
                arabic_texts.append(tokenizer.decode(t, skip_special_tokens=True))

        return [
            {"start": s["start"], "end": s["end"],
             "text": s["text"], "arabic": ar}
            for s, ar in zip(segments, arabic_texts)
        ]

    def _to_saudi_dialect(self, texts: list) -> list:
        """
        Batch-rewrite Arabic texts into natural Saudi colloquial dialect via LLM.
        Sends all texts in a single API call as a JSON array; falls back to the
        original texts if the API key is missing or the call fails.
        """
        client = self._load_llm()
        if client is None:
            return texts

        serialized = json.dumps(texts, ensure_ascii=False)
        prompt = (
            "أنت متخصص في اللهجة السعودية العامية المحكية.\n"
            "أعد كتابة كل نص في المصفوفة التالية باللهجة السعودية اليومية الطبيعية.\n\n"
            "قواعد صارمة:\n"
            "- استخدم العامية السعودية الحجازية أو النجدية (مثل: وش، الحين، ابغى، زين، وايد، عشان، بس، كذا، يعني، ايش، ما في)\n"
            "- تجنّب الفصحى وأي مصطلحات رسمية تماماً\n"
            "- حافظ على نفس المعنى والمحتوى بالضبط\n"
            "- أعد فقط مصفوفة JSON من النصوص المُعادة الصياغة، بنفس الترتيب والطول\n"
            "- لا تضف أي شرح أو نص خارج المصفوفة\n\n"
            f"المدخل:\n{serialized}\n\n"
            "المخرج (مصفوفة JSON فقط):"
        )

        try:
            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()
            # Strip markdown code fences if the model wraps the output
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            rewritten = json.loads(raw.strip())
            if isinstance(rewritten, list) and len(rewritten) == len(texts):
                return [r if isinstance(r, str) and r.strip() else t
                        for r, t in zip(rewritten, texts)]
            log.warning("Dialect LLM returned unexpected shape (got %s items, expected %s) — using original Arabic",
                        len(rewritten) if isinstance(rewritten, list) else type(rewritten).__name__,
                        len(texts))
            return texts
        except Exception as exc:
            log.error("Dialect LLM call failed: %s — using original Arabic", exc)
            return texts

    def _dialectify_segments(self, ar_segs: list) -> list:
        """
        Apply Saudi dialect rewriting to the 'arabic' field of each segment in-place.
        Segments with empty arabic text are left unchanged.
        """
        texts   = [s.get("arabic", "").strip() for s in ar_segs]
        non_empty_indices = [i for i, t in enumerate(texts) if t]
        non_empty_texts   = [texts[i] for i in non_empty_indices]

        if not non_empty_texts:
            return ar_segs

        rewritten = self._to_saudi_dialect(non_empty_texts)

        result = [dict(s) for s in ar_segs]
        for idx, rewritten_text in zip(non_empty_indices, rewritten):
            result[idx]["arabic"] = rewritten_text
        return result

    def _write_ass(
        self,
        segments:   list,
        clip_start: float,
        path:       Path,
    ):
        """
        Write an ASS subtitle file with RTL Arabic text.
        Timestamps are offset so they start at 0 for the clipped segment.
        """
        lines = [_ASS_HEADER]

        for seg in segments:
            t0 = max(seg["start"] - clip_start, 0.0)
            t1 = seg["end"]  - clip_start

            if t1 <= t0:
                continue

            arabic = seg.get("arabic") or seg.get("text", "")
            arabic = arabic.replace("\n", "\\N")

            # {\rtl} override tag forces right-to-left rendering in libass
            lines.append(
                f"Dialogue: 0,{_secs_to_ass(t0)},{_secs_to_ass(t1)},"
                f"Arabic,,0,0,0,,{{\\rtl}}{arabic}\n"
            )

        path.write_text("".join(lines), encoding="utf-8")

    def _burn_subtitles(self, video: Path, ass: Path, output: Path):
        """
        Burn ASS subtitles into the video frame.
        Uses just the filename + cwd to avoid Windows drive-letter colon issues
        in FFmpeg filter strings.
        """
        _run([
            "ffmpeg", "-y",
            "-i",   str(video),
            "-vf",  f"ass=filename={ass.name}",
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "copy",
            str(output),
        ], cwd=str(ass.parent))

    # ── TTS audio replacement ─────────────────────────────────────────────────

    def _replace_audio_tts(
        self,
        video:      Path,
        ar_segs:    list,
        clip_start: float,
        output:     Path,
    ):
        """
        Synthesise Saudi-dialect Arabic speech for each segment using the local
        fine-tuned XTTSv2 model, build a time-aligned audio track, and replace
        the original video audio.

        Alignment strategy:
          • Each TTS utterance is placed at the segment's start time.
          • If TTS audio is longer than the segment window → truncate.
          • Gaps between segments stay silent.
        """
        tts = self._load_xtts()

        # Use the first dataset WAV as the cloning reference voice
        speaker_ref = next(_XTTS_WAV_DIR.glob("*.wav"), None) if _XTTS_WAV_DIR.exists() else None
        if speaker_ref is None:
            raise FileNotFoundError(
                "No speaker reference WAV found in backend/dataset/dataset/wavs/.\n"
                "Ensure the dataset WAV files are present for voice cloning."
            )

        sample_rate = 24000   # XTTSv2 native output rate
        clip_dur    = self._duration(video)
        n_samples   = int((clip_dur + 0.5) * sample_rate)
        full_audio  = np.zeros(n_samples, dtype=np.float32)

        for seg in ar_segs:
            t0     = seg["start"] - clip_start
            t1     = seg["end"]   - clip_start
            arabic = (seg.get("arabic") or "").strip()

            if not arabic or t1 <= t0 or t0 < 0:
                continue

            seg_samples = int((t1 - t0) * sample_rate)
            start_idx   = int(t0 * sample_rate)

            try:
                wav_list = tts.tts(
                    text=arabic,
                    speaker_wav=str(speaker_ref),
                    language="ar",
                )
                wave = np.array(wav_list, dtype=np.float32)

                if len(wave) > seg_samples:
                    wave = wave[:seg_samples]

                end_idx = min(start_idx + len(wave), n_samples)
                full_audio[start_idx:end_idx] = wave[:end_idx - start_idx]

            except Exception as exc:
                log.warning("TTS segment skipped [%.30s…]: %s", arabic, exc)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name

        sf.write(tmp_wav, full_audio, sample_rate)

        try:
            _run([
                "ffmpeg", "-y",
                "-i",       str(video),
                "-i",       tmp_wav,
                "-map",     "0:v:0",
                "-map",     "1:a:0",
                "-c:v",     "copy",
                "-c:a",     "aac",
                "-shortest",
                str(output),
            ])
        finally:
            Path(tmp_wav).unlink(missing_ok=True)
