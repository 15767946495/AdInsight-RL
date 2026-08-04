"""External reward plugin for MAC GRPO training.

Registers a custom AsyncORM `mac_claim_f1` that calls Gemini 3.1 Pro to score
model completions against atomic reference points using claim-level F1.

Reward formula:
    R_total = (0.50*claim_f1 + 0.15*precision + 0.25*recall) - duplicate_penalty - overclaim_penalty

Matching is soft: the judge returns a continuous 0.0-1.0 match_score per pair, so
TP is fractional, F1 is continuous, and far fewer GRPO groups tie on identical
rewards. Recall is an explicit positive term: it directly rewards covering more
reference points, counteracting the under-claim drift that over-weighted
precision alone induces (the model otherwise learns to drop points to maximize
precision). Precision stays as a hallucination disincentive; overclaim_penalty is
soft so a matched 4th point is never deterred.

Overclaim: predicted points beyond max(3, tp) are penalized. The first 3 matched
points are free; a 4th point is free only when it matches a reference (supported
by data); any unmatched extra point loses OVERCLAIM_PENALTY.

Invalid numbered-list formatting receives zero reward.

Usage in swift rlhf:
    --reward_funcs mac_claim_f1 \
    --external_plugins /data/tec-chi/MARS2/MAC/rl/reward_plugin.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any, List

try:
    from swift.rewards import AsyncORM, orms
except ImportError:  # allow standalone testing of pure reward functions
    AsyncORM = object
    orms = {}

if TYPE_CHECKING:
    from swift.rlhf_trainers import GRPOConfig


JUDGE_URL = os.environ.get("MAC_REWARD_JUDGE_URL", "").strip()
JUDGE_MODEL = os.environ.get("MAC_REWARD_JUDGE_MODEL", "gemini-3.1-pro-preview").strip()
JUDGE_API_KEY = os.environ.get("MAC_REWARD_JUDGE_API_KEY", "").strip()
JUDGE_TIMEOUT = 120
JUDGE_MAX_RETRIES = 2
JUDGE_MAX_TOKENS = 4096
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_CONCURRENCY = int(os.environ.get("MAC_REWARD_JUDGE_CONCURRENCY", "28"))
JUDGE_CACHE_SIZE = int(os.environ.get("MAC_REWARD_JUDGE_CACHE_SIZE", "20000"))

DUPLICATE_POINT_PENALTY = 0.05
FREE_POINT_COUNT = 3
OVERCLAIM_PENALTY = 0.05
MATCH_THRESHOLD = 0.5
CLAIM_F1_WEIGHT = 0.50
PRECISION_WEIGHT = 0.15
RECALL_WEIGHT = 0.25

SCORE_SYSTEM_PROMPT = """You score a predicted answer against atomic gold reference claims for advertising video QA.

The references were independently verified against the video and are the source of truth for this scoring pass. Match semantic meaning, not exact wording.

Rules:
- Each reference can match at most one predicted point.
- Each predicted point can match at most one reference. Never award multiple true positives to one numbered point.
- Rate every proposed match on a continuous 0.0-1.0 confidence: 1.0 = exact semantic match, 0.7 = clear paraphrase, 0.5 = partial overlap of the core claim, below 0.5 = not a real match. Only propose a match when genuine overlap exists (score >= 0.5); otherwise leave the prediction and reference unmatched.
- If one predicted point combines multiple independent claims, mark it as bundled even when all claims are plausible.
- If a predicted point adds a factual or marketing claim not supported by its matched reference, mark it as unsupported.
- A duplicate predicted point cannot match a reference that was already matched by an earlier point.
- A broad or generic statement scores low (well below 0.5) against a more specific reference, so do not propose it as a match.
- An unsupported, speculative, irrelevant, duplicate, or generic marketing point remains unmatched and is a false positive.
- A reference not matched by any prediction is a false negative.
- Do not reward formatting or verbosity as content quality. Use the full 0.0-1.0 range; two equally-correct paraphrases may still differ slightly (e.g. 0.92 vs 0.85) on specificity.

The predicted points are already parsed deterministically. Do not split, merge, or rewrite them.

Return only valid JSON:
{
  "matches": [
    {
      "prediction_index": 0,
      "reference_index": 0,
      "match_score": 0.85,
      "reason": "short semantic justification"
    }
  ],
  "unmatched_prediction_indices": [1],
  "unmatched_reference_indices": [2],
  "unsupported_prediction_indices": [1],
  "bundled_prediction_indices": [0],
  "format_valid": true,
  "product_category_error": false,
  "non_english": false,
  "notes": ["short scoring note"]
}"""


class JudgeRoutingMismatch(RuntimeError):
    """The gateway served a different judge model than requested."""


def _parse_prediction_points(prediction: str) -> tuple[list[str], bool]:
    matches = list(re.finditer(r"(?m)^\s*(\d+)\.\s+", prediction))
    if not matches:
        fallback = [line.strip() for line in prediction.splitlines() if line.strip()]
        return fallback, False
    points: list[str] = []
    numbers: list[int] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(prediction)
        text = re.sub(r"\s+", " ", prediction[start:end]).strip()
        if text:
            points.append(text)
            numbers.append(int(match.group(1)))
    return points, numbers == list(range(1, len(numbers) + 1))


def _count_duplicate_points(predicted_points: list[str]) -> int:
    normalized = [re.sub(r"\W+", " ", point.lower()).strip() for point in predicted_points]
    return len(normalized) - len(set(normalized))


def _parse_json_object(text: str) -> dict:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Model output must be a JSON object")
    return value


def _extract_answer(message: dict) -> str:
    content = message.get("content") or ""
    if message.get("reasoning_content") or message.get("reasoning"):
        return content.strip()
    if "</think>" in content:
        answer = content.rsplit("</think>", 1)[-1].strip()
        if answer:
            return answer
    return content.strip()


def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:500]}") from exc


def _build_score_prompt(reference: dict, predicted_points: list[str]) -> str:
    compact_references = [
        {
            "index": index,
            "claim": point.get("claim", ""),
            "evidence": point.get("evidence", ""),
        }
        for index, point in enumerate(reference.get("reference_points") or [])
    ]
    return "\n\n".join(
        [
            f"Question:\n{reference.get('question', '')}",
            f"Product category:\n{reference.get('product_category', 'other')}",
            "Gold reference claims:\n" + json.dumps(compact_references, ensure_ascii=False, indent=2),
            "Deterministically parsed model prediction points:\n"
            + json.dumps(
                [{"index": index, "text": point} for index, point in enumerate(predicted_points)],
                ensure_ascii=False,
                indent=2,
            ),
            "Score the prediction using the required atomic-reference matching JSON.",
        ]
    )


def _call_judge(reference: dict, predicted_points: list[str]) -> dict:
    if not JUDGE_URL or not JUDGE_API_KEY:
        raise RuntimeError("Set MAC_REWARD_JUDGE_URL and MAC_REWARD_JUDGE_API_KEY")
    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": SCORE_SYSTEM_PROMPT},
            {"role": "user", "content": _build_score_prompt(reference, predicted_points)},
        ],
        "max_completion_tokens": JUDGE_MAX_TOKENS,
        "temperature": JUDGE_TEMPERATURE,
    }
    headers = {"Authorization": f"Bearer {JUDGE_API_KEY}"}
    data = _post_json(JUDGE_URL, payload, headers, JUDGE_TIMEOUT)
    if data.get("error"):
        raise RuntimeError(json.dumps(data["error"], ensure_ascii=False)[:500])
    actual_model = str(data.get("model") or "")
    if actual_model and actual_model != JUDGE_MODEL:
        raise JudgeRoutingMismatch(
            f"Judge model routing mismatch: requested={JUDGE_MODEL}, actual={actual_model}"
        )
    return _parse_json_object(_extract_answer(data["choices"][0]["message"]))


def _normalize_score(raw: dict, predicted: list[str], format_valid: bool, reference_count: int) -> dict:
    used_predictions: set[int] = set()
    used_references: set[int] = set()
    # Collect candidate matches with a continuous 0.0-1.0 score, then accept the
    # strongest first (greedy assignment preserving the one-to-one constraint).
    # This makes TP fractional, so F1 is continuous and far fewer groups tie.
    candidate_matches: list[tuple[float, int, int]] = []
    for item in raw.get("matches") or []:
        if not isinstance(item, dict):
            continue
        try:
            prediction_index = int(item.get("prediction_index"))
            reference_index = int(item.get("reference_index"))
        except (TypeError, ValueError):
            continue
        if not 0 <= prediction_index < len(predicted) or not 0 <= reference_index < reference_count:
            continue
        try:
            score = float(item.get("match_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(1.0, score))
        if score < MATCH_THRESHOLD:
            continue
        candidate_matches.append((score, prediction_index, reference_index))
    candidate_matches.sort(reverse=True)
    soft_tp = 0.0
    for score, prediction_index, reference_index in candidate_matches:
        if prediction_index in used_predictions or reference_index in used_references:
            continue
        used_predictions.add(prediction_index)
        used_references.add(reference_index)
        soft_tp += score

    def valid_prediction_indices(key: str) -> set[int]:
        result: set[int] = set()
        for value in raw.get(key) or []:
            try:
                index = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(predicted):
                result.add(index)
        return result

    unsupported_predictions = valid_prediction_indices("unsupported_prediction_indices")
    bundled_predictions = valid_prediction_indices("bundled_prediction_indices")
    tp = soft_tp
    unmatched_prediction_count = len(predicted) - len(used_predictions) if predicted else 0
    flawed_matched_predictions = (unsupported_predictions | bundled_predictions) & used_predictions
    fp = unmatched_prediction_count + len(flawed_matched_predictions)
    fn = reference_count - len(used_references)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "unsupported_count": len(unsupported_predictions),
        "bundled_count": len(bundled_predictions),
        "format_valid": format_valid,
        "product_category_error": bool(raw.get("product_category_error")),
        "non_english": bool(raw.get("non_english")),
    }


def _compute_reward(score: dict, prediction: str, predicted_points: list[str]) -> float:
    if not prediction.strip() or not predicted_points:
        return 0.0
    if not score["format_valid"]:
        return 0.0
    if score["tp"] == 0:
        return 0.0
    duplicate_count = _count_duplicate_points(predicted_points)
    # Overclaim: points beyond max(3, tp). A matched 4th point (tp>=4) is free;
    # only unmatched extra points (the unsupported 4th+) are penalized.
    # tp is fractional (soft match), so cast to int for the free-slot floor.
    free_floor = max(FREE_POINT_COUNT, int(score["tp"]))
    overclaim_count = max(0, len(predicted_points) - free_floor)
    # Multi-component base: f1 (eval-aligned composite) + precision (discourages
    # hallucination) + recall (directly rewards covering reference points, to
    # counter the under-claim drift that over-weighted precision alone induces).
    base = (
        CLAIM_F1_WEIGHT * score["f1"]
        + PRECISION_WEIGHT * score["precision"]
        + RECALL_WEIGHT * score["recall"]
    )
    reward = (
        base
        - DUPLICATE_POINT_PENALTY * duplicate_count
        - OVERCLAIM_PENALTY * overclaim_count
    )
    if score["product_category_error"]:
        reward = min(reward, 0.2)
    if score["non_english"]:
        reward = min(reward, 0.4)
    return round(max(0.0, reward), 6)


class MacClaimF1(AsyncORM):
    """Claim-level F1 reward using Gemini 3.1 Pro as text judge."""

    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        self._semaphore = asyncio.Semaphore(max(1, JUDGE_MAX_CONCURRENCY))
        self._cache: dict[str, dict] = {}
        self._cache_order: list[str] = []
        self._stats = {
            "calls": 0,
            "success": 0,
            "errors": 0,
            "cache_hits": 0,
            "invalid_solution": 0,
            "empty_prediction": 0,
        }

    def _cache_key(self, reference: dict, predicted_points: list[str]) -> str:
        payload = json.dumps(
            {"reference": reference, "predicted_points": predicted_points},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _record(self, key: str) -> None:
        self._stats[key] += 1
        if key == "calls" and self._stats["calls"] % 100 == 0:
            print(f"[mac_claim_f1] {json.dumps(self._stats, sort_keys=True)}", file=sys.stderr, flush=True)

    async def _judge(self, reference: dict, predicted_points: list[str]) -> dict:
        key = self._cache_key(reference, predicted_points)
        cached = self._cache.get(key)
        if cached is not None:
            self._record("cache_hits")
            return cached
        async with self._semaphore:
            raw = await asyncio.to_thread(_call_judge, reference, predicted_points)
        if JUDGE_CACHE_SIZE > 0:
            self._cache[key] = raw
            self._cache_order.append(key)
            if len(self._cache_order) > JUDGE_CACHE_SIZE:
                self._cache.pop(self._cache_order.pop(0), None)
        return raw

    async def __call__(self, completions, **kwargs) -> List[float]:
        solutions = kwargs.get("solution")
        if solutions is None:
            solutions = kwargs.get("solutions")
        if solutions is None:
            return [0.0] * len(completions)
        if isinstance(solutions, str):
            solutions = [solutions]
        if not isinstance(solutions, list):
            solutions = list(solutions)
        if len(solutions) != len(completions):
            if len(solutions) == 1:
                solutions = solutions * len(completions)
            else:
                return [0.0] * len(completions)

        async def score_one(completion: str, solution: Any) -> float:
            self._record("calls")
            prediction = ""
            if isinstance(completion, str):
                prediction = completion
            elif isinstance(completion, list):
                prediction = " ".join(str(p) for p in completion)
            else:
                prediction = str(completion)
            try:
                reference = json.loads(solution) if isinstance(solution, str) else solution
            except Exception:
                self._record("invalid_solution")
                return 0.0
            predicted_points, format_valid = _parse_prediction_points(prediction)
            if not predicted_points:
                self._record("empty_prediction")
                return 0.0
            last_error: Exception | None = None
            for attempt in range(JUDGE_MAX_RETRIES + 1):
                try:
                    raw = await self._judge(reference, predicted_points)
                    score = _normalize_score(
                        raw, predicted_points, format_valid, len(reference.get("reference_points") or [])
                    )
                    if str(reference.get("product_category") or "other").strip().lower() == "other":
                        score["product_category_error"] = False
                    self._record("success")
                    return _compute_reward(score, prediction, predicted_points)
                except JudgeRoutingMismatch:
                    raise
                except Exception as exc:
                    last_error = exc
                    if attempt < JUDGE_MAX_RETRIES:
                        await asyncio.sleep(min(10, 3 * (2**attempt)))
            self._record("errors")
            return 0.0

        tasks = [score_one(c, s) for c, s in zip(completions, solutions)]
        return await asyncio.gather(*tasks)


orms["mac_claim_f1"] = MacClaimF1


if __name__ == "__main__":
    # Self-check: soft matching + multi-component reward + overclaim penalty.
    # Runs standalone (swift import is optional). Fails fast if logic breaks.

    def _mk_score(tp, n_points, refs=None) -> dict:
        refs = refs if refs is not None else tp
        fp = n_points - tp
        fn = max(0, refs - tp)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {
            "tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1,
            "unsupported_count": fp, "bundled_count": 0, "format_valid": True,
            "product_category_error": False, "non_english": False,
        }

    def _rew(tp, n_points, refs=None) -> float:
        pts = [f"point {i}" for i in range(n_points)]  # distinct -> no dup penalty
        return _compute_reward(_mk_score(tp, n_points, refs), "1. a\n2. b", pts)

    def _base(tp, n_points, refs=None) -> float:
        s = _mk_score(tp, n_points, refs)
        return round(
            CLAIM_F1_WEIGHT * s["f1"] + PRECISION_WEIGHT * s["precision"] + RECALL_WEIGHT * s["recall"], 6
        )

    # --- overclaim + multi-component (scale-agnostic invariants) ---
    r3 = _rew(3, 3)
    # 4th point supported (tp=4): free, equals the 3-matched-perfect baseline
    assert _rew(4, 4) == r3, "4th point that matches a reference must be free"
    # 4th point unsupported (tp stays 3): strictly lower
    assert _rew(3, 4) < r3, "unsupported 4th point must be penalized"
    # 5th point with only 4 matchable: penalized
    assert _rew(4, 5) < _rew(4, 4), "unsupported 5th point must be penalized"
    # monotonic: more unsupported extra points -> strictly lower reward
    assert _rew(3, 4) > _rew(3, 5) > _rew(3, 6), "more overclaim must lower reward"
    # no over-claim penalty when points <= 3 even if some miss
    assert _rew(2, 3) == _base(2, 3), "no overclaim penalty for <=3 points"
    # 4 refs but model under-predicts: no penalty, completeness loss is in f1
    assert _rew(2, 2, refs=4) == _base(2, 2, refs=4)

    # --- soft matching: continuous match_score breaks ties that binary matching tied ---
    def _norm(judge_json: dict, predicted: list[str], refs: int) -> dict:
        return _normalize_score(judge_json, predicted, True, refs)

    predicted = ["p0", "p1", "p2"]
    # Two predictions both match ref 0, but one strongly (0.95) and one weakly (0.6).
    # Under binary matching both would be "paraphrase" -> tie; under soft matching
    # the strong match wins the slot and yields higher fractional TP.
    strong = _norm({"matches": [
        {"prediction_index": 0, "reference_index": 0, "match_score": 0.95},
    ], "unmatched_prediction_indices": [1, 2], "unmatched_reference_indices": [1, 2]},
        predicted, 3)
    weak = _norm({"matches": [
        {"prediction_index": 0, "reference_index": 0, "match_score": 0.60},
    ], "unmatched_prediction_indices": [1, 2], "unmatched_reference_indices": [1, 2]},
        predicted, 3)
    assert 0.0 < weak["tp"] < strong["tp"] < 1.0, (weak["tp"], strong["tp"])
    assert strong["f1"] > weak["f1"], "soft score must distinguish strong vs weak match"
    # match below threshold is rejected -> not a match
    rej = _norm({"matches": [
        {"prediction_index": 0, "reference_index": 0, "match_score": 0.30},
    ], "unmatched_prediction_indices": [0, 1, 2], "unmatched_reference_indices": [0, 1, 2]},
        predicted, 3)
    assert rej["tp"] == 0.0 and rej["precision"] == 0.0, "sub-threshold match must be rejected"
    # one-to-one: a reference cannot be claimed by two predictions; strongest wins
    two_v_one = _norm({"matches": [
        {"prediction_index": 0, "reference_index": 0, "match_score": 0.9},
        {"prediction_index": 1, "reference_index": 0, "match_score": 0.7},
    ], "unmatched_prediction_indices": [2], "unmatched_reference_indices": [1, 2]},
        predicted, 3)
    assert abs(two_v_one["tp"] - 0.9) < 1e-9, "strongest match must win the shared reference"
    print("ok: soft-match + multi-component + overclaim self-check passed")

