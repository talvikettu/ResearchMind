"""Compare generation profiles on dev, freeze one, then capture held-out answers."""

import argparse
import json
import statistics
from pathlib import Path

from .config import Settings
from .engine import Engine
from .store import Store
from .tune_retrieval import CachedReranker
from .tuning_capture import score

PROFILES = {
    "retrieval_only": dict(
        answer_profile="legacy",
        answer_context=6144,
        answer_max_tokens=2000,
        answer_sources=6,
        answer_temperature=0.0,
    ),
    "structured_6k": dict(
        answer_profile="structured",
        answer_context=6144,
        answer_max_tokens=2000,
        answer_sources=6,
        answer_temperature=0.0,
    ),
    "structured_8k": dict(
        answer_profile="structured",
        answer_context=8192,
        answer_max_tokens=2400,
        answer_sources=8,
        answer_temperature=0.15,
    ),
}
for profile in PROFILES.values():
    profile["technical_expansion"] = False
    profile["answer_source_first"] = False
    profile["strict_entity_evidence"] = False
PROFILES["structured_glossary"] = dict(
    answer_profile="structured",
    answer_context=8192,
    answer_max_tokens=2400,
    answer_sources=8,
    answer_temperature=0.0,
    technical_expansion=True,
)
PROFILES["structured_glossary"]["answer_source_first"] = False
PROFILES["structured_glossary"]["strict_entity_evidence"] = False
PROFILES["source_first"] = dict(
    PROFILES["structured_glossary"], answer_source_first=True, strict_entity_evidence=True
)
for profile in PROFILES.values():
    profile["teaching_precision"] = False
PROFILES["grounded"] = dict(PROFILES["source_first"], answer_profile="grounded", teaching_precision=True)
PROFILES["grounded"].update(per_document=2, evidence_limit=8, retrieval_k=16)
for profile in PROFILES.values():
    profile.update(answer_model="qwen2.5:3b", clean_evidence=False)
PROFILES["mistral_existing"] = dict(
    PROFILES["grounded"],
    answer_profile="structured",
    answer_source_first=False,
    answer_model="mistral:instruct",
    answer_context=6144,
    answer_sources=6,
    answer_max_tokens=2000,
    clean_evidence=True,
)


def summary(rows):
    return dict(
        coverage=statistics.mean(r["scores"]["coverage"] for r in rows),
        words=statistics.mean(r["scores"]["words"] for r in rows),
        repetition=statistics.mean(r["scores"]["repeated_5grams"] for r in rows),
        median_seconds=statistics.median(r["result"]["latency_ms"] / 1000 for r in rows),
        valid_source_cards=sum(
            len(r["result"]["articles"]) == len({a["source"] for a in r["result"]["articles"]}) for r in rows
        ),
        unavailable=sum(not r["result"].get("generation", {}).get("available") for r in rows),
        model_claims=sum(r["result"].get("generation", {}).get("model_claims", 0) for r in rows),
        count=len(rows),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/tuning/answers"))
    parser.add_argument("--retrieval", type=Path, default=Path("reports/tuning/retrieval/selection.json"))
    parser.add_argument("--dataset", type=Path, default=Path("examples/tuning_questions.json"))
    parser.add_argument("--baseline", type=Path, default=Path("reports/tuning/baseline.jsonl"))
    parser.add_argument(
        "--dev-only", action="store_true", help="Stop before selection/test for a review of dev answers"
    )
    parser.add_argument(
        "--preferred-profile", choices=list(PROFILES), help="Explicit choice after dev review"
    )
    parser.add_argument("--selection-note", default="")
    parser.add_argument(
        "--question-ids", nargs="+", help="Optional dev smoke subset; never use this for final selection"
    )
    parser.add_argument(
        "--profiles", nargs="+", choices=list(PROFILES), help="Evaluate only specified dev profiles"
    )
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "selection.json").exists():
        raise FileExistsError("A selection already exists; preserve finished experiments")
    retrieval = json.loads(args.retrieval.read_text())
    samples = json.loads(args.dataset.read_text(encoding="utf-8"))
    settings = Settings(**retrieval["config"], answer_backend="ollama")
    store = Store(settings)
    store.cross_encoder = CachedReranker(store.cross_encoder)
    summaries = {}

    def apply(config):
        for k, v in config.items():
            setattr(settings, k, v)
        return Engine(store)

    def capture(name, engine, split):
        target = out / (name + "-" + split + ".jsonl")
        rows = (
            [json.loads(s) for s in target.read_text(encoding="utf-8").splitlines()]
            if target.exists()
            else []
        )
        done = {r["id"] for r in rows}
        for sample in samples:
            if sample["split"] != split or sample["id"] in done:
                continue
            if split == "dev" and args.question_ids and sample["id"] not in args.question_ids:
                continue
            result = engine.query(sample["question"], mode="agentic", detail="standard")
            row = dict(
                id=sample["id"],
                split=split,
                question=sample["question"],
                result=result,
                scores=score(result["answer"], sample),
            )
            with target.open("a", encoding="utf-8") as file:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            print(
                json.dumps(
                    dict(
                        profile=name,
                        id=sample["id"],
                        coverage=row["scores"]["coverage"],
                        words=row["scores"]["words"],
                        seconds=round(result["latency_ms"] / 1000, 2),
                    )
                ),
                flush=True,
            )
        return rows

    try:
        for name, config in PROFILES.items():
            if args.profiles and name not in args.profiles:
                continue
            summaries[name] = summary(capture(name, apply(config), "dev"))
        (out / "dev-summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        if args.dev_only:
            return

        # Coverage is an explicit diagnostic proxy, not a claim of semantic correctness.
        def objective(m):
            return m["coverage"] - 0.25 * m["repetition"] - m["unavailable"] / max(1, m["count"])

        winner = args.preferred_profile or max(summaries, key=lambda n: objective(summaries[n]))
        selection = dict(
            name=winner,
            config=dict(retrieval["config"], **PROFILES[winner]),
            summaries=summaries,
            selected_on=("Dev review: " + args.selection_note)
            if args.preferred_profile
            else "dev rubric coverage minus repetitions and availability failures",
        )
        (out / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
        final = summary(capture("selected", apply(PROFILES[winner]), "test"))
        baseline = [json.loads(s) for s in args.baseline.read_text(encoding="utf-8").splitlines()]
        by_id = {s["id"]: s for s in samples}
        for row in baseline:
            row["scores"] = score(row["result"]["answer"], by_id[row["id"]])
        comparison = dict(
            selection=selection,
            baseline_test=summary([r for r in baseline if r["split"] == "test"]),
            selected_test=final,
            limitations="Internal, single-seed development study. Rubric pattern coverage is not correctness. Model judge is not independent. Latency includes warm caches and is not an isolated performance benchmark.",
        )
        (out / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        print("COMPARISON " + json.dumps(comparison), flush=True)
    finally:
        store.client.close()


if __name__ == "__main__":
    main()
