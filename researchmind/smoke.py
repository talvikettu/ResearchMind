"""Live bilingual product smoke checks. Model verification is not semantic accuracy."""

import argparse
import json
from pathlib import Path

import httpx


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--api", default="http://127.0.0.1:8000")
    p.add_argument("--output", type=Path, default=Path("reports/product-smoke"))
    p.add_argument("--limit", type=int, default=8)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    questions = json.loads(Path("examples/product_questions.json").read_text(encoding="utf-8"))[: args.limit]
    rows = []
    with httpx.Client(base_url=args.api, timeout=600) as client:
        health = client.get("/health")
        health.raise_for_status()
        for q in questions:
            response = client.post("/query", json=dict(q, detail="standard"))
            response.raise_for_status()
            result = response.json()
            rows.append(result)
            (args.output / "answers.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                json.dumps(
                    dict(
                        question=q["question"],
                        seconds=round(result["latency_ms"] / 1000, 2),
                        passed=result["verification"]["passed"],
                        words=len(result["answer"].split()),
                        answer_kind=result["answer_kind"],
                        partial=result.get("generation", {}).get("partial"),
                    ),
                    ensure_ascii=True,
                ),
                flush=True,
            )
    summary = dict(
        count=len(rows),
        http_success=len(rows),
        verification_passed=sum(r["verification"]["passed"] for r in rows),
        unique_cards=sum(len(r["articles"]) == len({a["source"] for a in r["articles"]}) for r in rows),
        model_claims=sum(r.get("generation", {}).get("model_claims", 0) for r in rows),
        source_fallback_claims=sum(r.get("generation", {}).get("source_fallback_claims", 0) for r in rows),
        configuration=health.json(),
        limitations="Development smoke only. A refusal can be the right result. Local self-verification is not independent accuracy or faithfulness.",
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = ["# Product smoke", "", "```json", json.dumps(summary, indent=2), "```", ""]
    for r in rows:
        report += [
            "## " + r["question"],
            "",
            r["answer"],
            "",
            "Sources: " + ", ".join(a["title"] for a in r["articles"]),
            "",
        ]
    (args.output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
