#!/usr/bin/env python3
"""Post-process raw ASR: merge adjacent segments, fix EOF, truncate, assess reliability.

Usage:
    python scripts/postprocess_asr.py --input data/asr_raw.jsonl --videos /path/to/videos --output data/asr_final.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import av
except ImportError:
    sys.exit("PyAV required: pip install av")


def get_duration(video_path: Path) -> float:
    with av.open(str(video_path)) as container:
        if container.streams.audio:
            return float(container.streams.audio[0].duration * container.streams.audio[0].time_base)
        return float(container.duration / 1_000_000)


def merge_segments(segments: list[dict], gap: float = 0.5) -> list[dict]:
    cleaned: list[dict] = []
    for seg in segments:
        try:
            start, end = float(seg["start"]), float(seg["end"])
        except (KeyError, TypeError, ValueError):
            continue
        text = re.sub(r"\s+", " ", str(seg.get("text", ""))).strip()
        if not text or end <= start:
            continue
        norm = re.sub(r"\W+", " ", text.casefold()).strip()
        if cleaned and norm == cleaned[-1]["norm"]:
            cleaned[-1]["end"] = end
            continue
        if cleaned and start - cleaned[-1]["end"] <= gap:
            cleaned[-1]["end"] = end
            cleaned[-1]["text"] += " " + text
            cleaned[-1]["norm"] = re.sub(r"\W+", " ", cleaned[-1]["text"].casefold()).strip()
        else:
            cleaned.append({"start": start, "end": end, "text": text, "norm": norm})
    for seg in cleaned:
        seg.pop("norm", None)
    return cleaned


def fix_eof(segments: list[dict], duration: float) -> list[dict]:
    result = []
    for seg in segments:
        if seg["start"] >= duration:
            continue
        if seg["end"] > duration:
            seg["end"] = round(duration, 3)
        result.append(seg)
    return result


def assess_reliability(segments: list[dict]) -> str:
    words = sum(len(s["text"].split()) for s in segments)
    norms = [re.sub(r"\W+", " ", s["text"].casefold()).strip() for s in segments if s["text"]]
    repeated = len(norms) - len(set(norms))
    if words <= 2 or repeated > len(norms) / 4:
        return "low"
    return "medium"


def main():
    parser = argparse.ArgumentParser(description="Post-process raw ASR transcripts")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--questions", type=Path, help="Question JSONL used to enforce ID coverage and order")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merge-gap", type=float, default=0.5)
    parser.add_argument("--max-chars", type=int, default=12000)
    args = parser.parse_args()
    if not 1 <= args.max_chars <= 12000:
        parser.error("max-chars must be between 1 and 12000")

    args.input = args.input.expanduser().resolve()
    args.videos = args.videos.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    indexed: dict[str, dict] = {}
    for row in rows:
        video_name = row.get("video", "")
        if not video_name:
            raise ValueError("ASR row is missing video")
        if video_name in indexed:
            raise ValueError(f"duplicate ASR video: {video_name}")
        if row.get("error"):
            raise ValueError(f"ASR generation failed for {video_name}: {row['error']}")
        indexed[video_name] = row
    if args.questions:
        questions = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines() if line.strip()]
        expected = [f"{row['id']}.mp4" for row in questions]
        missing = set(expected) - set(indexed)
        extra = set(indexed) - set(expected)
        if missing or extra or len(indexed) != len(expected):
            raise ValueError(f"ASR coverage mismatch: missing={len(missing)} extra={len(extra)}")
        rows = [indexed[name] for name in expected]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as writer:
        for row in rows:
            video_name = row.get("video", "")
            segments = row.get("audio_to_text", [])
            if not isinstance(segments, list):
                segments = []
            video_path = args.videos / video_name
            if video_path.exists():
                try:
                    duration = get_duration(video_path)
                    segments = fix_eof(segments, duration)
                except Exception:
                    pass
            segments = merge_segments(segments, args.merge_gap)
            reliability = assess_reliability(segments) if segments else "unavailable"
            lines = [f'[{s["start"]:07.2f}-{s["end"]:07.2f}] {s["text"]}' for s in segments]
            transcript = "\n".join(lines)
            if len(transcript) > args.max_chars:
                kept = []
                used = 0
                for segment, line in zip(segments, lines):
                    required = len(line) + (1 if kept else 0)
                    if used + required > args.max_chars:
                        break
                    kept.append(segment)
                    used += required
                segments = kept
                reliability = "low"
            if not segments:
                reliability = "unavailable"
            writer.write(json.dumps({
                "video": video_name,
                "audio_to_text": segments,
                "reliability": reliability,
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
