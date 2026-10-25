"""Reproducible bilingual smoke evaluation of detailed, grounded answers."""

import json
import statistics
from pathlib import Path

import httpx

QUESTIONS = [
    "Explain how Adam combines momentum and adaptive learning rates.",
    "Compare Lasso and Ridge regularization and their effect on coefficients.",
    "How does CatBoost address categorical features and prediction shift?",
    "Explain DBSCAN density based clustering and its limitations.",
    "How does dropout reduce overfitting in neural networks?",
    "What is calibration of neural network probabilities?",
    "Как работает линейная регрессия и как обучаются её коэффициенты?",
    "Объясни переобучение и способы регуляризации моделей.",
    "Как работает градиентный бустинг над деревьями?",
    "Что такое кросс-валидация и для чего она нужна?",
]


def main():
    root = Path("reports/detailed_rag")
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=600) as client:
        health = client.get("/health").json()
        for question in QUESTIONS:
            response = client.post("/query", json=dict(question=question, detail="detailed"))
            response.raise_for_status()
            result = response.json()
            row = dict(
                question=question,
                words=len(result["answer"].split()),
                sections=len(result.get("generation", {}).get("accepted_claims", [])),
                model_sections=result.get("generation", {}).get("model_claims", 0),
                grounded=result["verification"].get("verbatim_grounded", False),
                unique_cards=len(result["articles"]) == len({a["source"] for a in result["articles"]}),
                rounds=sum(t["step"] == "plan" for t in result["trace"]),
                result=result,
            )
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "result"}, ensure_ascii=True), flush=True)
    summary = dict(
        count=len(rows),
        mean_words=statistics.mean(r["words"] for r in rows),
        grounded=sum(r["grounded"] for r in rows),
        unique_cards=sum(r["unique_cards"] for r in rows),
        model_sections=sum(r["model_sections"] for r in rows),
        max_rounds=max(r["rounds"] for r in rows),
        configuration=health,
        limitations="Development smoke set; source substring support does not establish relevance, completeness or semantic accuracy.",
    )
    (root / "answers.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "# Detailed bilingual QA smoke evaluation",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
    ]
    for row in rows:
        report += [
            "## " + row["question"],
            "",
            row["result"]["answer"],
            "",
            "Sources: " + ", ".join(a["title"] for a in row["result"]["articles"]),
            "",
        ]
    (root / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
