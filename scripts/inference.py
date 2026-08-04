#!/usr/bin/env python3
"""Run inference: load model endpoint + video + ASR → official submission JSONL.

Usage (single shard):
    python scripts/inference.py --base-url http://127.0.0.1:8240/v1 \
        --model Qwen3.5-9B-GSPO-Step300 --videos /path/to/videos \
        --questions data/MAC_QA.jsonl --asr data/asr_final.jsonl \
        --output outputs/shard_0.jsonl --shard 0 --num-shards 7
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

try:
    import aiohttp
except ImportError:
    sys.exit("aiohttp required: pip install aiohttp")


SYSTEM_PROMPT = Path(__file__).resolve().parent.parent / "prompts" / "final_answer.txt"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def asr_index(path: Path) -> dict[str, list[dict]]:
    result = {}
    for row in load_jsonl(path):
        stem = Path(row.get("video", "")).stem
        if stem:
            result[stem] = row.get("audio_to_text", [])
    return result


def build_transcript(segments: list[dict], merge_gap: float = 0.5, max_chars: int = 12000) -> tuple[str, str]:
    cleaned: list[dict] = []
    for seg in segments:
        try:
            start, end = float(seg["start"]), float(seg["end"])
        except (KeyError, TypeError, ValueError):
            continue
        text = re.sub(r"\s+", " ", str(seg.get("text", ""))).strip()
        if not text or end <= start:
            continue
        if cleaned and start - cleaned[-1]["end"] <= merge_gap:
            cleaned[-1]["end"] = end
            cleaned[-1]["text"] += " " + text
        else:
            cleaned.append({"start": start, "end": end, "text": text})
    if not cleaned:
        return "unavailable", "No usable automatic transcript is available."
    words = sum(len(s["text"].split()) for s in cleaned)
    reliability = "low" if words <= 2 else "medium"
    lines = [f'[{s["start"]:07.2f}-{s["end"]:07.2f}] {s["text"]}' for s in cleaned]
    transcript = "\n".join(lines)
    if len(transcript) > max_chars:
        transcript = transcript[:max_chars].rsplit("\n", 1)[0]
        reliability = "low"
    return reliability, transcript


def validate_answer(text: str) -> list[str]:
    issues = []
    stripped = text.strip()
    if not re.match(r"^1\.\s+\S", stripped):
        issues.append("must start with '1.'")
    if "**" in stripped or "```" in stripped:
        issues.append("no markdown")
    numbers = [int(v) for v in re.findall(r"(?m)^\s*(\d+)\.\s+", stripped)]
    if not numbers or numbers != list(range(1, len(numbers) + 1)):
        issues.append("numbering not continuous")
    if len(numbers) > 4:
        issues.append("more than 4 points")
    return issues


async def call_model(session, base_url, model_name, video_path, system_prompt, user_text, max_tokens, temperature, top_p, seed):
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": f"file://{video_path}"}},
                {"type": "text", "text": user_text},
            ]},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if seed is not None:
        payload["seed"] = seed
    url = base_url.rstrip("/") + "/chat/completions"
    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=600)) as resp:
        body = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {body[:500]}")
        data = json.loads(body)
        content = data["choices"][0]["message"].get("content", "")
        if "know" in content.lower() and content.strip().startswith("<"):
            content = content.rsplit("know", 1)[-1]
        return content.strip()


async def run_shard(args):
    questions = load_jsonl(args.questions)
    asr = asr_index(args.asr)
    system_prompt = SYSTEM_PROMPT.read_text(encoding="utf-8").strip()
    video_dir = args.videos

    selected = [q for i, q in enumerate(questions) if i % args.num_shards == args.shard]
    existing = {}
    if args.resume and args.output.exists():
        existing = {r["id"]: r for r in load_jsonl(args.output) if r.get("model_prediction")}
        selected = [q for q in selected if q["id"] not in existing]

    print(f"shard {args.shard}: {len(selected)} pending, {len(existing)} existing")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    connector = aiohttp.TCPConnector(limit=max(4, args.workers * 3))
    async with aiohttp.ClientSession(connector=connector) as session:
        semaphore = asyncio.Semaphore(args.workers)
        lock = asyncio.Lock()
        results = dict(existing)

        async def worker(item):
            qid = item["id"]
            question = item["question"]
            async with semaphore:
                video_path = video_dir / f"{qid}.mp4"
                segments = asr.get(qid, [])
                reliability, transcript = build_transcript(segments)
                user_text = (
                    f"QUESTION\n{question}\n\n"
                    f"SPEECH TRANSCRIPT (automatic, reliability: {reliability})\n{transcript}\n\n"
                    "Answer the QUESTION using the original video and the supporting transcript above."
                )
                answer = ""
                for attempt in range(3):
                    try:
                        answer = await call_model(
                            session, args.base_url, args.model, video_path,
                            system_prompt, user_text, args.max_tokens,
                            args.temperature, args.top_p, args.seed,
                        )
                        issues = validate_answer(answer)
                        if not issues:
                            break
                        user_text = (
                            f"Rewrite to fix: {'; '.join(issues)}.\n\n"
                            f"QUESTION\n{question}\n\nREJECTED\n{answer}\n\n"
                            "Output 2-4 numbered points starting with '1.', no markdown."
                        )
                    except Exception as exc:
                        if attempt < 2:
                            await asyncio.sleep(2 * (attempt + 1))
                        else:
                            answer = f"ERROR: {exc}"
                            break
                async with lock:
                    results[qid] = {"id": qid, "model_prediction": answer}
                    ordered = [results[q["id"]] for q in questions if q["id"] in results]
                    tmp = args.output.with_suffix(".tmp")
                    with tmp.open("w", encoding="utf-8") as f:
                        for r in ordered:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    tmp.replace(args.output)

        await asyncio.gather(*(worker(q) for q in selected))
    print(f"shard {args.shard}: wrote {len(results)} rows to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="AdInsight-RL inference")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--asr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    asyncio.run(run_shard(parser.parse_args()))


if __name__ == "__main__":
    main()
