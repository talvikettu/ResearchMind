"""Capture identical live questions before/after tuning; rubrics are coverage proxies."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import httpx


def score(answer, sample):
    scorable = re.sub(r"\[\d+\]", "", answer)
    checks = {
        label: any(re.search(pattern, scorable, re.I | re.S) for pattern in patterns)
        for label, patterns in sample["rubric"].items()
    }
    tokens = re.findall(r"\w+", answer.lower())
    grams = [tuple(tokens[i : i + 5]) for i in range(max(0, len(tokens) - 4))]
    return dict(
        rubric=checks,
        coverage=sum(checks.values()) / max(1, len(checks)),
        words=len(answer.split()),
        repeated_5grams=1 - len(set(grams)) / len(grams) if grams else 0,
        note="Pattern coverage only: neither correctness nor an independent faithfulness score.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--api", default="http://127.0.0.1:8000")
    p.add_argument("--label", required=True)
    p.add_argument("--dataset", type=Path, default=Path("examples/tuning_questions.json"))
    p.add_argument("--split", choices=["dev", "test", "all"], default="all")
    p.add_argument("--output", type=Path, default=Path("reports/tuning"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.dataset.read_bytes()
    samples = json.loads(raw)
    target = args.output / (args.label + ".jsonl")
    done = (
        {r["id"] for r in map(json.loads, target.read_text(encoding="utf-8").splitlines())}
        if target.exists()
        else set()
    )
    with httpx.Client(base_url=args.api, timeout=600) as client:
        health = client.get("/health")
        health.raise_for_status()
        (args.output / (args.label + "-manifest.json")).write_text(
            json.dumps(dict(dataset_sha256=hashlib.sha256(raw).hexdigest(), health=health.json()), indent=2),
            encoding="utf-8",
        )
        for sample in samples:
            if sample["id"] in done or args.split != "all" and sample["split"] != args.split:
                continue
            response = client.post(
                "/query", json=dict(question=sample["question"], mode="agentic", detail="standard")
            )
            response.raise_for_status()
            result = response.json()
            row = dict(
                id=sample["id"],
                split=sample["split"],
                question=sample["question"],
                result=result,
                scores=score(result["answer"], sample),
            )
            with target.open("a", encoding="utf-8") as file:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                json.dumps(
                    dict(id=sample["id"], **row["scores"], seconds=round(result["latency_ms"] / 1000, 2))
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
