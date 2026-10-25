"""python -m researchmind.evaluate --dataset examples/questions.jsonl --seed examples/docs"""

import argparse
import asyncio
import csv
import json
import statistics
from pathlib import Path

from .config import Settings
from .engine import Engine
from .store import Store


def retrieval_metrics(sources: list[dict], relevant: list[str]) -> dict:
    # Document-level relevance: count duplicate chunks only once; ranks remain chunk ranks.
    expected = set(relevant)
    found = [s["source"] for s in sources[:5]]
    return {
        "recall_at_5": len(set(found) & expected) / len(expected) if expected else None,
        "mrr_at_5": next((1 / i for i, name in enumerate(found, 1) if name in expected), 0)
        if expected
        else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="examples/questions.jsonl")
    parser.add_argument("--seed", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--ragas", action="store_true")
    args = parser.parse_args()
    settings = Settings()
    store = Store(settings)
    try:
        if args.seed:
            for path in sorted(args.seed.iterdir()):
                if path.suffix.lower() in {".txt", ".md", ".pdf"}:
                    store.ingest(path.name, path.read_bytes())
        engine = Engine(store)
        judge = None
        if args.ragas:
            from langchain_openai import ChatOpenAI
            from ragas.llms import LangchainLLMWrapper
            from ragas.metrics import Faithfulness

            if not settings.llm_url:
                raise ValueError("Ragas requires an LLM endpoint")
            judge = Faithfulness(
                llm=LangchainLLMWrapper(
                    ChatOpenAI(
                        model=settings.llm_model,
                        base_url=settings.llm_url,
                        api_key=settings.llm_key or "local",
                        temperature=0,
                    )
                )
            )
        samples = [
            json.loads(line)
            for line in Path(args.dataset).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows, answers = [], []
        for sample in samples:
            for mode in ("naive", "hybrid", "agentic"):
                result = engine.query(sample["question"], mode)
                faithfulness = None
                if judge and result["sources"]:
                    from ragas.dataset_schema import SingleTurnSample

                    faithfulness = asyncio.run(
                        judge.single_turn_ascore(
                            SingleTurnSample(
                                user_input=sample["question"],
                                response=result["answer"],
                                retrieved_contexts=[s["text"] for s in result["sources"]],
                            )
                        )
                    )
                rows.append(
                    dict(
                        id=sample["id"],
                        mode=mode,
                        **retrieval_metrics(result["sources"], sample["relevant_sources"]),
                        faithfulness=faithfulness,
                        latency_ms=result["latency_ms"],
                        llm_cost_usd=result["llm_cost_usd"],
                        verified=result["verification"]["passed"],
                    )
                )
                answers.append(result)
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / "results.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {}
        for mode in ("naive", "hybrid", "agentic"):
            subset = [r for r in rows if r["mode"] == mode]
            summary[mode] = {}
            for key in ("recall_at_5", "mrr_at_5", "faithfulness", "latency_ms", "llm_cost_usd"):
                values = [r[key] for r in subset if r[key] is not None]
                summary[mode][key] = statistics.mean(values) if values else None
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (args.output / "answers.json").write_text(
            json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
    finally:
        store.client.close()


if __name__ == "__main__":
    main()
