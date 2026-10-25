"""Check retrieval through the running API without locking its local Qdrant."""

import argparse
import json
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    rows = []
    with httpx.Client(timeout=180) as client:
        response = client.get(args.api + "/documents")
        response.raise_for_status()
        documents = response.json()
        print(f"Library: {len(documents)} documents, {sum(d['chunks'] for d in documents)} chunks")
        for line in Path("examples/paper-questions.jsonl").read_text(encoding="utf-8").splitlines():
            sample = json.loads(line)
            response = client.post(
                args.api + "/query", json={"question": sample["question"], "mode": "agentic"}
            )
            response.raise_for_status()
            result = response.json()
            found = {s["source"] for s in result["sources"]}
            expected = set(sample["relevant_sources"])
            row = {
                "id": sample["id"],
                "expected_found": sorted(found & expected),
                "expected_missing": sorted(expected - found),
                "sources": sorted(found),
                "citation_check": result["verification"],
                "latency_ms": result["latency_ms"],
                "demo": result["demo"],
            }
            rows.append(row)
            print(f"{sample['id']}: {len(found & expected)}/{len(expected)} expected papers retrieved")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/corpus-check.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if any(row["expected_missing"] for row in rows):
        raise SystemExit("Some expected sources were not retrieved; inspect reports/corpus-check.json")


if __name__ == "__main__":
    main()
