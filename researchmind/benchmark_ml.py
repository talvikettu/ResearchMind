"""Evaluate 50 classical ML questions through the live unified API.

Keyword coverage is a diagnostic, never an answer correctness score. Raw drafts,
fallbacks, references and source excerpts are preserved for human review.
"""

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import httpx

from .ingest import tokens


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--api", default="http://127.0.0.1:8000")
    p.add_argument("--output", type=Path, default=Path("reports/classical_ml_50"))
    args = p.parse_args()
    dataset = Path("examples/classical_ml_50.json")
    questions = json.loads(dataset.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    rows, answers = [], []
    with httpx.Client(timeout=300) as client:
        response = client.get(args.api + "/health")
        response.raise_for_status()
        health = response.json()
        for sample in questions:
            response = client.post(
                args.api + "/query", json={"question": sample["question"], "mode": "agentic"}
            )
            response.raise_for_status()
            result = response.json()
            retrieved = [s["source"] for s in result["sources"]]
            answer_words = set(tokens(result["answer"]))
            covered = [key for key in sample["keywords"] if key in answer_words]
            row = dict(
                id=sample["id"],
                topic=sample["topic"],
                expected_document_retrieved=sample["source"] in retrieved,
                expected_document_cited=sample["source"] in [a["source"] for a in result["articles"]],
                citation_ids_valid=result["verification"]["citation_ids_valid"],
                keyword_coverage=len(covered) / len(sample["keywords"]),
                generation_available=result.get("generation", {}).get("available", False),
                fallback=health.get("answer_backend") == "ollama"
                and not result.get("generation", {}).get("available", False),
                latency_ms=result["latency_ms"],
                correctness="not_independently_judged",
            )
            rows.append(row)
            answers.append(dict(sample=sample, metrics=row, result=result))
            (args.output / "answers.json").write_text(
                json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(
                f"{sample['id']:02d}/50 expected_source={row['expected_document_retrieved']} "
                f"fallback={row['fallback']}",
                flush=True,
            )
    summary = {
        "questions": len(rows),
        "language": "English",
        "mode": "agentic",
        "review_protocol": "All answers checked for expected document, citation syntax, keyword coverage "
        "and generation availability. Semantic accuracy is not automatically inferred.",
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "configuration": health,
        "expected_source_retrieved": sum(r["expected_document_retrieved"] for r in rows),
        "expected_source_cited": sum(r["expected_document_cited"] for r in rows),
        "citation_ids_valid": sum(r["citation_ids_valid"] for r in rows),
        "generation_available": sum(r["generation_available"] for r in rows),
        "fallbacks": sum(r["fallback"] for r in rows),
        "mean_keyword_coverage": statistics.mean(r["keyword_coverage"] for r in rows),
        "mean_latency_ms": statistics.mean(r["latency_ms"] for r in rows),
        "faithfulness": None,
        "answer_accuracy": None,
        "limitations": "Author-created development set, English, no independent semantic judge. "
        "Keyword coverage and valid citation IDs are not correctness or faithfulness.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output / "metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = [
        "# Classical ML: 50-question evaluation",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
        "References are author-written; "
        "correctness requires human review. Expected-source retrieval is document-level.",
        "",
    ]
    for item in answers:
        sample, result, metrics = item["sample"], item["result"], item["metrics"]
        report += [
            f"## {sample['id']}. {sample['question']}",
            "",
            "Reference: " + sample["reference"],
            "",
            "Expected source: " + sample["source"],
            "",
            "Returned answer:",
            "",
            result["answer"],
            "",
            "Diagnostics: " + json.dumps(metrics),
            "",
        ]
    (args.output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
