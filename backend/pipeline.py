"""
Maqta3 — Video Processing Pipeline
------------------------------------
Handles:
  1. Video download (yt-dlp) or local file passthrough
  2. Audio extraction (FFmpeg)
  3. Speech transcription + language detection (Whisper)
  4. Highlight extraction using heuristic scoring
  5. Vertical 9:16 center-crop (FFmpeg)
  6. Arabic subtitle generation via translation (Helsinki-NLP/opus-mt-en-ar)
  7. Subtitle burning into video (FFmpeg ASS filter)
  8. Arabic TTS audio replacement (AhmedEladl/saudi-tts) — English input only
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import soundfile as sf

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
        self._whisper          = None
        self._mt_tokenizer     = None
        self._mt_model         = None
        self._tts_tokenizer    = None
        self._tts_model        = None

    # ── Lazy loaders ─────────────────────────────────────────────────────────

    def _load_whisper(self):
        if self._whisper is None:
            import whisper
            # "base" balances speed and accuracy well for MVP
            self._whisper = whisper.load_model("base")
        return self._whisper

    def _load_mt(self):
        """Helsinki-NLP English → Arabic machine-translation model."""
        if self._mt_model is None:
            from transformers import MarianMTModel, MarianTokenizer
            name = "Helsinki-NLP/opus-mt-en-ar"
            self._mt_tokenizer = MarianTokenizer.from_pretrained(name)
            self._mt_model     = MarianMTModel.from_pretrained(name)
        return self._mt_tokenizer, self._mt_model

    def _load_tts(self):
        """Saudi-dialect TTS (VITS architecture)."""
        if self._tts_model is None:
            import torch
            from transformers import AutoTokenizer, VitsModel
            name = "AhmedEladl/saudi-tts"
            self._tts_tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False)
            self._tts_model     = VitsModel.from_pretrained(name)
        return self._tts_tokenizer, self._tts_model

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

        # 1 ── Obtain video file
        progress_callback("downloading", 5, "Downloading video…")
        video = self._get_video(source_url, source_path, work)

        # 2 ── Extract mono 16 kHz audio for Whisper
        progress_callback("extracting", 12, "Extracting audio…")
        audio = self._extract_audio(video, work)

        # 3 ── Get total duration
        total_dur = self._duration(video)

        # 4 ── Transcribe + detect language
        progress_callback("transcribing", 18,
                          "Transcribing speech… (1–3 min depending on length)")
        transcript = self._transcribe(audio)
        language   = transcript.get("language", "en")
        segments   = transcript.get("segments", [])

        if not segments:
            raise ValueError(
                "Whisper found no speech in this video. "
                "Try a video with clearer audio."
            )

        # 5 ── Highlight scoring
        progress_callback("scoring", 55, "Scoring highlight candidates…")
        highlights = self._extract_highlights(segments, total_dur, num_highlights)

        if not highlights:
            raise ValueError(
                "Could not find suitable highlight windows (20–60 s). "
                "The video may be too short or consist mostly of silence."
            )

        results = []
        n = len(highlights)

        for i, hl in enumerate(highlights):
            base = 58 + int(i / n * 38)

            # 6 ── Trim raw clip
            progress_callback("clipping", base,
                              f"Cutting clip {i+1}/{n}…")
            raw_clip = work / f"_raw_{i}.mp4"
            self._trim(video, hl["start"], hl["end"], raw_clip)

            # 7 ── Convert to 9:16
            progress_callback("converting", base + 1,
                              f"Converting clip {i+1} to vertical…")
            vert_clip = work / f"_vert_{i}.mp4"
            self._make_vertical(raw_clip, vert_clip)

            # 8 ── Translate subtitle segments to Arabic
            progress_callback("subtitling", base + 2,
                              f"Generating Arabic subtitles for clip {i+1}…")
            ar_segs = self._translate_segments(hl["segments"], language)

            # 9 ── Write ASS file
            ass_file = work / f"_subs_{i}.ass"
            self._write_ass(ar_segs, hl["start"], ass_file)

            # 10 ── Burn subtitles
            sub_clip = work / f"_sub_{i}.mp4"
            self._burn_subtitles(vert_clip, ass_file, sub_clip)

            final = sub_clip

            # 11 ── TTS audio replacement (English → Arabic only)
            if language == "en":
                progress_callback("tts", base + 4,
                                  f"Generating Arabic voiceover for clip {i+1}…")
                try:
                    tts_clip = work / f"_tts_{i}.mp4"
                    self._replace_audio_tts(sub_clip, ar_segs,
                                            hl["start"], tts_clip)
                    final = tts_clip
                except Exception as exc:
                    print(f"[TTS] Clip {i+1} skipped: {exc}")

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

            results.append({
                "clip_index":    i + 1,
                "filename":      final_name,
                "start":         round(hl["start"], 2),
                "end":           round(hl["end"], 2),
                "duration":      round(hl["end"] - hl["start"], 2),
                "arabic_preview": arabic_preview,
            })

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

    def _trim(
        self,
        video:  Path,
        start:  float,
        end:    float,
        output: Path,
    ):
        _run([
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-to", f"{end:.3f}",
            "-i",  str(video),
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "aac",
            str(output),
        ])

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
            "-vf",  f"ass={ass.name}",
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
        Generate Arabic speech for each segment, build a time-aligned
        audio track, and replace the original video audio.

        Alignment strategy:
          • Each TTS utterance is placed at the segment's original start time.
          • If TTS audio is longer than the segment window → truncate.
          • Gaps between segments stay silent (no background music for MVP).
        """
        tokenizer, model = self._load_tts()
        import torch

        clip_dur    = self._duration(video)
        sample_rate = model.config.sampling_rate
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
                inputs = tokenizer(arabic, return_tensors="pt")
                with torch.no_grad():
                    wave = model(**inputs).waveform.squeeze().numpy()

                if len(wave) > seg_samples:
                    wave = wave[:seg_samples]

                end_idx = min(start_idx + len(wave), n_samples)
                full_audio[start_idx:end_idx] = wave[: end_idx - start_idx]

            except Exception as exc:
                print(f"[TTS] Segment skipped ({arabic[:30]}…): {exc}")

        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as tmp:
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
