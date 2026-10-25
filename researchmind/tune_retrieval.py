"""Search retrieval settings on dev questions; freeze before testing. No re-embedding."""

import argparse
import json
from pathlib import Path

import numpy as np

from .config import Settings
from .engine import Engine
from .store import Store


class CachedReranker:
    """Cache exact query/passage pairs across profiles, never answers or reference labels."""

    def __init__(self, model):
        self.model, self.cache = model, {}

    def predict(self, pairs, **kwargs):
        keys = [tuple(p) for p in pairs]
        missing = list(dict.fromkeys(k for k in keys if k not in self.cache))
        if missing:
            self.cache.update(zip(missing, map(float, self.model.predict(missing, **kwargs))))
        return np.array([self.cache[k] for k in keys])


PROFILES = {
    "baseline": {},
    "wide": dict(
        search_pool=40,
        rerank_candidates=80,
        retrieval_k=12,
        evidence_limit=8,
        per_document=2,
        parent_max_chars=3000,
        collapse_parents=True,
    ),
    "dense": dict(
        search_pool=40,
        rerank_candidates=80,
        dense_weight=0.8,
        retrieval_k=12,
        evidence_limit=8,
        per_document=2,
        parent_max_chars=3000,
        collapse_parents=True,
    ),
    "lexical": dict(
        search_pool=40,
        rerank_candidates=80,
        dense_weight=0.2,
        retrieval_k=12,
        evidence_limit=8,
        per_document=2,
        parent_max_chars=3000,
        collapse_parents=True,
    ),
    "educational": dict(
        search_pool=40,
        rerank_candidates=80,
        handbook_boost=0.15,
        title_weight=0.04,
        retrieval_k=12,
        evidence_limit=8,
        per_document=2,
        parent_max_chars=3000,
        collapse_parents=True,
    ),
    "compact": dict(
        search_pool=40,
        rerank_candidates=80,
        handbook_boost=0.15,
        title_weight=0.04,
        retrieval_k=12,
        evidence_limit=6,
        per_document=1,
        parent_max_chars=3000,
        collapse_parents=True,
    ),
    "parent": dict(
        search_pool=40,
        rerank_candidates=80,
        handbook_boost=0.15,
        title_weight=0.04,
        retrieval_k=12,
        evidence_limit=8,
        per_document=2,
        parent_max_chars=6000,
        collapse_parents=True,
    ),
    "balanced": dict(
        search_pool=32,
        rerank_candidates=64,
        dense_weight=0.6,
        handbook_boost=0.08,
        title_weight=0.08,
        retrieval_k=10,
        evidence_limit=8,
        per_document=2,
        parent_max_chars=4000,
        diversity_penalty=0.3,
        collapse_parents=True,
    ),
}
PROFILES["focused"] = dict(PROFILES["compact"], entity_routing=True)


def retrieve(engine, question):
    state = dict(
        question=question,
        detail="standard",
        contexts=[],
        rounds=0,
        trace=[],
        allow_web=False,
        usage=dict(prompt_tokens=0, completion_tokens=0, llm_calls=0, web_calls=0),
    )
    state.update(engine.plan(state))
    state.update(engine.retrieve(state))
    return state


def evaluate(engine, samples):
    rows = []
    for sample in samples:
        state = retrieve(engine, sample["question"])
        contexts = state["contexts"]
        ranks = [i for i, c in enumerate(contexts[:5], 1) if c["source"] in sample["sources"]]
        row = dict(
            id=sample["id"],
            reference_hit_at_5=bool(ranks),
            mrr_at_5=1 / min(ranks) if ranks else 0,
            contexts=len(contexts),
            sources=[c["source"] for c in contexts],
            trace=state["trace"],
        )
        rows.append(row)
        print(json.dumps(dict(id=sample["id"], hit=bool(ranks), sources=row["sources"][:3])), flush=True)
    return dict(
        reference_hit_at_5=sum(r["reference_hit_at_5"] for r in rows) / len(rows),
        mrr_at_5=sum(r["mrr_at_5"] for r in rows) / len(rows),
        mean_contexts=sum(r["contexts"] for r in rows) / len(rows),
        rows=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--previous", type=Path, help="Preserved first-round summary; evaluate the added focused profile only"
    )
    parser.add_argument("--output", type=Path, default=Path("reports/tuning/retrieval"))
    args = parser.parse_args()
    folder = args.output
    folder.mkdir(parents=True, exist_ok=True)
    selected_path = folder / "selection.json"
    if selected_path.exists():
        raise FileExistsError("A finished search exists; preserve it and use a new report directory")
    samples = json.loads(Path("examples/tuning_questions.json").read_text(encoding="utf-8"))
    dev = [s for s in samples if s["split"] == "dev"]
    settings = Settings(
        answer_backend="extractive",
        strict_entity_evidence=False,
        technical_expansion=False,
        teaching_precision=False,
    )
    original = settings.model_copy(deep=True)
    for key in set().union(*(set(p) for p in PROFILES.values())):
        setattr(original, key, Settings.model_fields[key].get_default())
    store = Store(settings)
    store.cross_encoder = CachedReranker(store.cross_encoder)
    engine = Engine(store)
    summary = json.loads((args.previous / "summary.json").read_text()) if args.previous else {}

    def apply(config):
        for key in set().union(*(set(p) for p in PROFILES.values())):
            setattr(settings, key, config.get(key, getattr(original, key)))

    try:
        for name, config in PROFILES.items():
            if args.previous and name != "focused":
                continue
            apply(config)
            print("PROFILE " + name, flush=True)
            metrics = evaluate(engine, dev)
            (folder / (name + "-dev.json")).write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            summary[name] = {k: v for k, v in metrics.items() if k != "rows"}
        name = max(
            summary,
            key=lambda n: (
                summary[n]["reference_hit_at_5"],
                summary[n]["mrr_at_5"],
                -summary[n]["mean_contexts"],
            ),
        )
        selection = dict(
            name=name,
            config=PROFILES[name],
            dev=summary[name],
            selected_on="dev only; preferred-source labels are not exhaustive relevance judgments",
        )
        selected_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
        (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        test = [s for s in samples if s["split"] == "test"]
        for label, config in (("baseline", PROFILES["baseline"]), ("selected", PROFILES[name])):
            apply(config)
            metrics = evaluate(engine, test)
            (folder / (label + "-test.json")).write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        print("SELECTED " + json.dumps(selection), flush=True)
    finally:
        store.client.close()


if __name__ == "__main__":
    main()
