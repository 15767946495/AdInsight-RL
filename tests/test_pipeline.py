from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class PipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.work = Path(self.temp_dir.name)
        self.videos = self.work / "videos"
        self.videos.mkdir()
        self.questions = self.work / "questions.jsonl"
        self.asr = self.work / "asr.jsonl"
        self.prompt = self.work / "prompt.txt"
        self.ids = ["sample-a", "sample-b"]
        write_jsonl(self.questions, [{"id": value, "question": "Question?"} for value in self.ids])
        write_jsonl(
            self.asr,
            [
                {"video": f"{value}.mp4", "audio_to_text": [], "reliability": "unavailable"}
                for value in reversed(self.ids)
            ],
        )
        for value in self.ids:
            (self.videos / f"{value}.mp4").touch()
        self.prompt.write_text("System prompt\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def run_script(self, name: str, *args: str | Path, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPTS / name), *map(str, args)],
            check=check,
            capture_output=True,
            text=True,
        )

    def test_preflight_merge_and_manifest(self) -> None:
        self.run_script(
            "preflight.py",
            "inference",
            "--questions", self.questions,
            "--videos", self.videos,
            "--asr", self.asr,
        )

        shard_a = self.work / "shard-a.jsonl"
        shard_b = self.work / "shard-b.jsonl"
        submission = self.work / "submission.jsonl"
        manifest = self.work / "submission.manifest.json"
        write_jsonl(shard_a, [{"id": self.ids[0], "model_prediction": "1. First\n2. Second"}])
        write_jsonl(shard_b, [{"id": self.ids[1], "model_prediction": "1. First\n2. Second"}])
        self.run_script(
            "merge_shards.py",
            "--questions", self.questions,
            "--shard", shard_a,
            "--shard", shard_b,
            "--output", submission,
        )
        self.run_script(
            "run_metadata.py",
            "--kind", "inference",
            "--questions", self.questions,
            "--videos", self.videos,
            "--model", "test-model",
            "--model-revision", "test-revision",
            "--file", f"asr={self.asr}",
            "--file", f"system_prompt={self.prompt}",
            "--setting", "seed=42",
            "--output", self.work / "run_metadata.json",
        )
        self.run_script(
            "write_manifest.py",
            "--run-metadata", self.work / "run_metadata.json",
            "--output", submission,
            "--manifest", manifest,
        )
        manifest_text = manifest.read_text(encoding="utf-8")
        self.assertNotIn(str(self.work), manifest_text)
        self.assertEqual(len(submission.read_text(encoding="utf-8").splitlines()), 2)

        self.prompt.write_text("Changed prompt\n", encoding="utf-8")
        mismatch = self.run_script(
            "run_metadata.py",
            "--kind", "inference",
            "--questions", self.questions,
            "--videos", self.videos,
            "--model", "test-model",
            "--model-revision", "test-revision",
            "--file", f"asr={self.asr}",
            "--file", f"system_prompt={self.prompt}",
            "--setting", "seed=42",
            "--output", self.work / "run_metadata.json",
            "--verify",
            check=False,
        )
        self.assertNotEqual(mismatch.returncode, 0)
        self.assertIn("does not match", mismatch.stderr)

    def test_merge_rejects_single_point_answer(self) -> None:
        shard = self.work / "bad.jsonl"
        write_jsonl(
            shard,
            [
                {"id": self.ids[0], "model_prediction": "1. Only"},
                {"id": self.ids[1], "model_prediction": "1. First\n2. Second"},
            ],
        )
        result = self.run_script(
            "merge_shards.py",
            "--questions", self.questions,
            "--shard", shard,
            "--output", self.work / "bad-submission.jsonl",
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("bad format", result.stderr)

    def test_inference_uses_absolute_video_uri(self) -> None:
        requests: list[dict] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers["Content-Length"])
                requests.append(json.loads(self.rfile.read(length)))
                body = json.dumps({
                    "choices": [{"message": {"content": "1. First\n2. Second"}}]
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            output = self.work / "inference.jsonl"
            self.run_script(
                "inference.py",
                "--base-url", f"http://127.0.0.1:{server.server_port}/v1",
                "--model", "test-model",
                "--videos", self.videos,
                "--questions", self.questions,
                "--asr", self.asr,
                "--system-prompt", self.prompt,
                "--output", output,
                "--workers", "2",
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

        self.assertEqual(len(requests), 2)
        video_url = requests[0]["messages"][1]["content"][0]["video_url"]["url"]
        user_text = requests[0]["messages"][1]["content"][1]["text"]
        self.assertTrue(video_url.startswith("file:///"))
        self.assertIn("reliability: unavailable", user_text)


if __name__ == "__main__":
    unittest.main()
