"""Bounded query decomposition, lexical gap detection and diverse evidence selection."""

import re

from .evidence import lexical_tokens, query_terms
from .ingest import tokens


def facets(question, detail="standard"):
    parts = [p.strip() for p in re.split(r"[;\n]+", question) if len(p.strip()) > 6]
    if len(parts) > 1:
        return [dict(label=p[:100], query=p) for p in parts[:4]]
    methods = re.findall(
        r"\b(?:lasso|ridge|xgboost|catboost|lightgbm|lora|qlora|bagging|boosting|лассо|ридж|бэггинг|бустинг)\b",
        question,
        re.I,
    )
    if len(set(m.lower() for m in methods)) > 1:
        plan = []
        for method in dict.fromkeys(methods):
            query = question
            for other in methods:
                if other.lower() != method.lower():
                    query = re.sub(r"\b" + re.escape(other) + r"\b", "", query, flags=re.I)
            plan.append(dict(label=method, query=" ".join(query.split())))
        return plan[:4]
    if detail == "detailed":
        russian = bool(re.search("[а-яА-Я]", question))
        labels = (
            [
                ("Основная идея", "определение"),
                ("Как это работает", "алгоритм"),
                ("Особенности и ограничения", "ограничения"),
            ]
            if russian
            else [("Main idea", "definition"), ("How it works", "algorithm"), ("Trade-offs", "limitations")]
        )
        return [dict(label=label, query=question + " " + term) for label, term in labels]
    return [dict(label=question[:100], query=question)]


def coverage(plan, contexts):
    words = set().union(*(set(lexical_tokens(c["text"])) for c in contexts)) if contexts else set()
    result = []
    for item in plan:
        required = query_terms(item["query"])
        ratio = len(required & words) / max(len(required), 1)
        result.append(dict(**item, lexical_coverage=ratio, covered=ratio >= 0.2))
    return result


def select_evidence(
    contexts, question, limit=10, per_document=3, prefer_titles=False, diversity_penalty=0.15
):
    from .evidence import excerpt

    candidates = []
    for context in contexts:
        _, relevance = excerpt(context["text"], question)
        if context.get("reranker_score") is not None:
            # Do not discard semantic matches just because Russian/English words differ.
            relevance = 0.85 * context["reranker_score"] + 0.15 * relevance
        if context.get("entity_title_match"):
            relevance += 0.2
        if relevance > 0:
            if prefer_titles:
                title_overlap = query_terms(question) & set(lexical_tokens(context.get("title", "")))
                relevance += 0.04 * min(3, len(title_overlap))
            candidates.append(dict(context, relevance=relevance))
    if prefer_titles and candidates:
        threshold = max(c["relevance"] for c in candidates) * 0.6
        candidates = [c for c in candidates if c["relevance"] >= threshold]
    chosen = []
    counts = {}
    while candidates and len(chosen) < limit:
        best = None
        best_value = -float("inf")
        for row in candidates:
            doc = row.get("document_id", row["source"])
            if counts.get(doc, 0) >= per_document:
                continue
            words = set(tokens(row["text"]))
            redundancy = max(
                (
                    len(words & set(tokens(c["text"]))) / max(len(words | set(tokens(c["text"]))), 1)
                    for c in chosen
                ),
                default=0,
            )
            if redundancy >= 0.85:
                continue
            value = (
                row["relevance"]
                + row.get("fusion_score", 0)
                - diversity_penalty * redundancy
                + 0.04 * (doc not in counts)
            )
            if value > best_value:
                best, best_value = row, value
        if best is None:
            break
        chosen.append(best)
        key = best.get("document_id", best["source"])
        counts[key] = counts.get(key, 0) + 1
        candidates.remove(best)
    return chosen
