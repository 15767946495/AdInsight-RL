#!/usr/bin/env python3
"""Merge shard outputs into official submission format."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

def load(path):
    return [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", type=Path, required=True)
    p.add_argument("--shard", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    questions = load(args.questions)
    expected = [q["id"] for q in questions]
    if len(expected) != len(set(expected)):
        raise ValueError("question IDs are not unique")
    results = {}
    for path in args.shard:
        for row in load(path):
            rid = row["id"]
            if rid in results:
                raise ValueError(f"duplicate id: {rid}")
            results[rid] = {"id": rid, "model_prediction": str(row["model_prediction"]).strip()}
    missing = set(expected) - set(results)
    extra = set(results) - set(expected)
    if missing or extra or len(results) != len(expected):
        raise ValueError(
            f"coverage: {len(results)}/{len(expected)} missing={len(missing)} extra={len(extra)}"
        )
    for rid in expected:
        answer = results[rid]["model_prediction"]
        nums = [int(v) for v in re.findall(r"(?m)^\s*(\d+)\.\s+", answer)]
        if (
            not answer
            or "ERROR:" in answer
            or nums != list(range(1, len(nums) + 1))
            or not 2 <= len(nums) <= 4
        ):
            raise ValueError(f"bad format: {rid}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rid in expected:
            f.write(json.dumps(results[rid], ensure_ascii=False) + "\n")
    tmp.replace(args.output)
    print(f"wrote {len(expected)} rows to {args.output}")

if __name__ == "__main__":
    main()
