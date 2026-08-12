#!/usr/bin/env python3
"""Generate ASR transcripts for advertising videos using faster-whisper.

Usage:
    python scripts/generate_asr.py --videos /path/to/videos --output data/asr_raw.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import av
except ImportError:
    sys.exit("PyAV required: pip install av")
try:
    from faster_whisper import WhisperModel
except ImportError:
    sys.exit("faster-whisper required: pip install faster-whisper")


def decode_audio(video_path: Path, sample_rate: int = 16000) -> tuple[bytes, float]:
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sample_rate)
    frames: list[bytes] = []
    with av.open(str(video_path)) as container:
        streams = container.streams.audio
        if not streams:
            return b"", 0.0
        for frame in container.decode(streams[0]):
            converted = resampler.resample(frame)
            for mf in (converted if isinstance(converted, list) else [converted]):
                frames.append(mf.to_ndarray().tobytes())
    raw = b"".join(frames)
    duration = len(raw) / 2 / sample_rate
    return raw, duration


def transcribe(model: WhisperModel, audio_bytes: bytes, beam_size: int = 5):
    import numpy as np
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    segments, info = model.transcribe(
        audio, beam_size=beam_size, language=None, vad_filter=True,
        condition_on_previous_text=False, word_timestamps=True,
    )
    result = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            result.append({"start": round(seg.start, 3), "end": round(seg.end, 3), "text": text})
    return result


def main():
    parser = argparse.ArgumentParser(description="Transcribe video audio with faster-whisper")
    parser.add_argument("--videos", type=Path, required=True, help="Directory of .mp4 files")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--model", type=str, default="large-v3-turbo", help="Whisper model size or CT2 path")
    parser.add_argument("--model-revision", type=str, default="", help="Frozen Hugging Face revision")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compute-type", type=str, default="float16")
    parser.add_argument("--download-root", type=Path, help="Model download/cache directory")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--shard", type=int, default=0, help="0-indexed shard number")
    parser.add_argument("--num-shards", type=int, default=1, help="Total shards for parallel runs")
    parser.add_argument("--resume", action="store_true", help="Skip videos already in output")
    args = parser.parse_args()

    args.videos = args.videos.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.videos.is_dir():
        parser.error(f"video directory not found: {args.videos}")
    if args.num_shards < 1 or not 0 <= args.shard < args.num_shards:
        parser.error("require num-shards >= 1 and 0 <= shard < num-shards")
    videos = sorted(args.videos.glob("*.mp4"))
    videos = [v for i, v in enumerate(videos) if i % args.num_shards == args.shard]
    if not videos:
        parser.error(f"no videos assigned to shard {args.shard} in {args.videos}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict] = {}
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    resume_sources = (args.output, tmp) if args.resume else ()
    for resume_source in resume_sources:
        if not resume_source.exists():
            continue
        for line in resume_source.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    row = json.loads(line)
                    name = row.get("video", "")
                    if name and "error" not in row:
                        existing[name] = row
                except json.JSONDecodeError:
                    pass
    assigned = {video.name for video in videos}
    existing = {name: row for name, row in existing.items() if name in assigned}
    pending = [v for v in videos if v.name not in existing]
    print(f"shard {args.shard}: {len(pending)} pending, {len(existing)} reusable", flush=True)

    model_kwargs = {"device": args.device, "compute_type": args.compute_type}
    if args.download_root:
        model_kwargs["download_root"] = str(args.download_root.expanduser().resolve())
    if args.model_revision:
        model_kwargs["revision"] = args.model_revision
    model = WhisperModel(args.model, **model_kwargs)

    with tmp.open("w", encoding="utf-8") as writer:
        for video in videos:
            if video.name in existing:
                writer.write(json.dumps(existing[video.name], ensure_ascii=False) + "\n")
        failures = 0
        for i, video in enumerate(pending):
            try:
                audio_bytes, duration = decode_audio(video)
                if not audio_bytes:
                    segments = []
                else:
                    segments = transcribe(model, audio_bytes, args.beam_size)
                row = {"video": video.name, "audio_to_text": segments}
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                writer.flush()
                print(f"[{i+1}/{len(pending)}] {video.name}: {len(segments)} segments, {duration:.1f}s", flush=True)
            except Exception as exc:
                failures += 1
                row = {"video": video.name, "audio_to_text": [], "error": str(exc)}
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                writer.flush()
                print(f"[{i+1}/{len(pending)}] {video.name}: FAILED {exc}", flush=True)
    tmp.replace(args.output)
    if failures:
        sys.exit(f"{failures} video(s) failed transcription; rerun with --resume")


if __name__ == "__main__":
    main()
