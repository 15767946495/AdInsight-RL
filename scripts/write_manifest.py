#!/usr/bin/env python3
"""Write a final submission manifest from verified run metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Final submission JSONL")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    run_metadata = args.run_metadata.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    if len({run_metadata, output, manifest_path}) != 3:
        parser.error("run metadata, submission, and manifest paths must be distinct")
    for path in (run_metadata, output):
        if not path.is_file():
            parser.error(f"file not found: {path}")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run": json.loads(run_metadata.read_text(encoding="utf-8")),
        "submission": {
            "name": output.name,
            "size": output.stat().st_size,
            "sha256": sha256(output),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(manifest_path)
    print(f"wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
