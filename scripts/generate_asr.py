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


def transcribe(model: WhisperModel, audio_bytes: bytes, sample_rate: int, beam_size: int = 5):
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
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compute-type", type=str, default="float16")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--shard", type=int, default=0, help="0-indexed shard number")
    parser.add_argument("--num-shards", type=int, default=1, help="Total shards for parallel runs")
    parser.add_argument("--resume", action="store_true", help="Skip videos already in output")
    args = parser.parse_args()

    videos = sorted(args.videos.glob("*.mp4"))
    videos = [v for i, v in enumerate(videos) if i % args.num_shards == args.shard]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.resume and args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line).get("video", ""))
                except json.JSONDecodeError:
                    pass
    pending = [v for v in videos if v.name not in done]
    print(f"shard {args.shard}: {len(pending)} pending, {len(done)} done", flush=True)

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    with args.output.open("a", encoding="utf-8") as writer:
        for i, video in enumerate(pending):
            try:
                audio_bytes, duration = decode_audio(video, args.sample_rate)
                if not audio_bytes:
                    segments = []
                else:
                    segments = transcribe(model, audio_bytes, args.sample_rate, args.beam_size)
                row = {"video": video.name, "audio_to_text": segments}
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                writer.flush()
                print(f"[{i+1}/{len(pending)}] {video.name}: {len(segments)} segments, {duration:.1f}s", flush=True)
            except Exception as exc:
                row = {"video": video.name, "audio_to_text": [], "error": str(exc)}
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                writer.flush()
                print(f"[{i+1}/{len(pending)}] {video.name}: FAILED {exc}", flush=True)


if __name__ == "__main__":
    main()
