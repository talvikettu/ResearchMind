"""Discover Russian open-access ML articles through publisher metadata (no translations)."""

import json
import re
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup


def main():
    manifest = Path("examples/papers.json")
    records = json.loads(manifest.read_text(encoding="utf-8"))
    for row in records:
        if row["filename"] == "arxiv-1708.07120.pdf":
            row["title"] = (
                "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates"
            )
        if row["filename"] == "arxiv-1905.09713.pdf":
            row.update(
                filename="arxiv-1906.08237.pdf",
                url="https://arxiv.org/pdf/1906.08237",
                page_url="https://arxiv.org/abs/1906.08237",
                external_id="arxiv:1906.08237",
            )
    if not any(r["filename"] == "arxiv-1711.05101.pdf" for r in records):
        records.append(
            dict(
                filename="arxiv-1711.05101.pdf",
                title="Decoupled Weight Decay Regularization",
                url="https://arxiv.org/pdf/1711.05101",
                page_url="https://arxiv.org/abs/1711.05101",
                external_id="arxiv:1711.05101",
                language="en",
                kind="paper",
                topic="Optimization",
            )
        )
    corrections = {
        "ru-deep-learning-survey-2017.pdf": "https://vestnik.susu.ru/cmi/article/view/6152",
        "ru-xgboost-smote-2020.pdf": "https://vestnik.susu.ru/cmi/article/view/9992",
        "ru-parallel-boosting-2013.pdf": "https://num-meth.ru/index.php/journal/article/view/741",
    }
    with httpx.Client(
        timeout=35, follow_redirects=True, headers={"User-Agent": "ResearchMindPersonalLibrary/0.3"}
    ) as client:
        for row in records:
            if row["filename"] in corrections:
                page = corrections[row["filename"]]
                response = client.get(page)
                response.raise_for_status()
                tag = BeautifulSoup(response.text, "html.parser").select_one('meta[name="citation_pdf_url"]')
                if tag:
                    row.update(url=tag["content"], page_url=page)
        pages = set()
        for query in ["машинное обучение", "нейронная сеть", "классификация", "кластеризация", "регрессия"]:
            response = client.get(
                "https://num-meth.ru/index.php/journal/search/search", params={"query": query}
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            pages.update(a["href"] for a in soup.select("h3 a[href]") if "/article/view/" in a["href"])
            time.sleep(0.3)
        identities = {r["page_url"] for r in records}
        for page in sorted(pages):
            if page in identities:
                continue
            response = client.get(page)
            if response.status_code != 200:
                continue
            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.select_one('meta[name="citation_title"]')
            pdf = soup.select_one('meta[name="citation_pdf_url"]')
            if not title or not pdf or not re.search("[а-яА-Я]", title["content"]):
                continue
            identifier = page.rstrip("/").split("/")[-1]
            records.append(
                dict(
                    filename=f"ru-numerical-methods-{identifier}.pdf",
                    title=title["content"],
                    url=pdf["content"],
                    page_url=page,
                    language="ru",
                    kind="paper",
                    external_id="num-meth:" + identifier,
                    topic="Русскоязычная публикация по ML",
                )
            )
            time.sleep(0.3)
    manifest.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    tsv = Path("examples/paper_candidates.tsv")
    lines = (
        tsv.read_text(encoding="utf-8")
        .replace(
            "1708.07120\tDecoupled Weight Decay Regularization",
            "1708.07120\tSuper-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates",
        )
        .replace("1905.09713", "1906.08237")
    )
    if "1711.05101" not in lines:
        lines += "1711.05101\tDecoupled Weight Decay Regularization\n"
    tsv.write_text(lines, encoding="utf-8")
    print("Candidates", len(records), "Russian", sum(r.get("language") == "ru" for r in records))


if __name__ == "__main__":
    main()
