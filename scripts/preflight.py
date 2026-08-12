#!/usr/bin/env python3
"""Validate reproducibility inputs before expensive ASR or inference work."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path} at line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"expected JSON object in {path} at line {line_number}")
        rows.append(row)
    return rows


def question_ids(path: Path) -> list[str]:
    rows = load_jsonl(path)
    ids = [str(row.get("id", "")).strip() for row in rows]
    if not ids or any(not value for value in ids):
        raise ValueError(f"empty or missing question id in {path}")
    if any(not str(row.get("question", "")).strip() for row in rows):
        raise ValueError(f"empty or missing question text in {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate question IDs in {path}")
    return ids


def validate_videos(ids: list[str], videos: Path) -> None:
    if not videos.is_dir():
        raise FileNotFoundError(f"video directory not found: {videos}")
    missing = [value for value in ids if not (videos / f"{value}.mp4").is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} video(s), first: {missing[0]}.mp4")
    expected = {f"{value}.mp4" for value in ids}
    extra = {path.name for path in videos.glob("*.mp4")} - expected
    if extra:
        raise ValueError(f"unexpected video(s) in input directory, first: {sorted(extra)[0]}")


def validate_asr(ids: list[str], path: Path) -> None:
    rows = load_jsonl(path)
    stems = []
    for row in rows:
        video = str(row.get("video", "")).strip()
        if not video:
            raise ValueError(f"ASR row without video in {path}")
        if row.get("error"):
            raise ValueError(f"ASR error for {video}: {row['error']}")
        if not isinstance(row.get("audio_to_text", []), list):
            raise ValueError(f"audio_to_text must be a list for {video}")
        for index, segment in enumerate(row.get("audio_to_text", [])):
            if not isinstance(segment, dict):
                raise ValueError(f"ASR segment {index} is not an object for {video}")
            try:
                start = float(segment["start"])
                end = float(segment["end"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid ASR times in segment {index} for {video}") from exc
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                raise ValueError(f"invalid ASR range in segment {index} for {video}")
            if not str(segment.get("text", "")).strip():
                raise ValueError(f"empty ASR text in segment {index} for {video}")
        stems.append(Path(video).stem)
    if len(stems) != len(set(stems)):
        raise ValueError(f"duplicate ASR video IDs in {path}")
    missing = set(ids) - set(stems)
    extra = set(stems) - set(ids)
    if missing or extra or len(stems) != len(ids):
        raise ValueError(f"ASR coverage mismatch: missing={len(missing)} extra={len(extra)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    videos_parser = subparsers.add_parser("videos", help="Validate question/video coverage")
    inference_parser = subparsers.add_parser("inference", help="Validate all inference inputs")
    for subparser in (videos_parser, inference_parser):
        subparser.add_argument("--questions", type=Path, required=True)
        subparser.add_argument("--videos", type=Path, required=True)
    inference_parser.add_argument("--asr", type=Path, required=True)
    args = parser.parse_args()

    ids = question_ids(args.questions.expanduser().resolve())
    validate_videos(ids, args.videos.expanduser().resolve())
    if args.command == "inference":
        validate_asr(ids, args.asr.expanduser().resolve())
    print(f"preflight passed: {len(ids)} questions and videos")


if __name__ == "__main__":
    main()
