#!/usr/bin/env python3
"""Create or verify deterministic input and parameter metadata for a run."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, name: str | None = None) -> dict:
    return {"name": name or path.name, "size": path.stat().st_size, "sha256": sha256(path)}


def parse_pairs(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected NAME=VALUE, got: {value}")
        name, content = value.split("=", 1)
        if not name or name in result:
            raise ValueError(f"invalid or duplicate metadata key: {name}")
        result[name] = content
    return result


def question_ids(path: Path) -> list[str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [str(row["id"]) for row in rows]


def model_record(model: str, revision: str) -> dict:
    path = Path(model).expanduser()
    if not path.is_dir():
        if path.is_absolute() or model.startswith("."):
            raise FileNotFoundError(f"local model directory not found: {path}")
        return {"source": model, "revision": revision or None}
    path = path.resolve()
    files = []
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file_path.relative_to(path).as_posix()
        files.append(file_record(file_path, relative))
    if not files:
        raise ValueError(f"local model directory is empty: {path}")
    return {"source": path.name, "revision": revision or None, "files": files}


def package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("asr", "inference"), required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--file", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--setting", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true", help="Require an identical existing metadata file")
    args = parser.parse_args()

    questions = args.questions.expanduser().resolve()
    videos = args.videos.expanduser().resolve()
    named_files = parse_pairs(args.file)
    settings = parse_pairs(args.setting)
    output = args.output.expanduser().resolve()
    protected_paths = {questions, *(Path(path).expanduser().resolve() for path in named_files.values())}
    protected_paths.update(videos / f"{value}.mp4" for value in question_ids(questions))
    if output in protected_paths:
        parser.error("metadata output must not overwrite an input file")
    model_path = Path(args.model).expanduser()
    if model_path.is_dir() and output.is_relative_to(model_path.resolve()):
        parser.error("metadata output must not be inside the model directory")
    video_records = [file_record(videos / f"{value}.mp4", f"{value}.mp4") for value in question_ids(questions)]
    metadata = {
        "schema_version": 1,
        "kind": args.kind,
        "questions": file_record(questions),
        "videos": video_records,
        "model": model_record(args.model, args.model_revision),
        "files": {
            name: file_record(Path(path).expanduser().resolve())
            for name, path in sorted(named_files.items())
        },
        "settings": dict(sorted(settings.items())),
    }

    if args.verify:
        if not output.is_file():
            raise FileNotFoundError(f"resume metadata not found: {output}")
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != metadata:
            raise ValueError("resume metadata does not match current inputs, model, or settings")
        print(f"resume metadata verified: {output}")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)
    print(f"wrote run metadata to {output}")


if __name__ == "__main__":
    main()
