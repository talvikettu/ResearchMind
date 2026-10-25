"""Build a reproducible catalogue; original author terms remain in force."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

import httpx


def main():
    path = Path("examples/papers.json")
    records = json.loads(path.read_text(encoding="utf-8"))
    existing_ids = set()
    for item in records:
        found = re.search(r"arxiv.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", item["url"])
        if found:
            existing_ids.add(found[1])
        item.setdefault("language", "en")
        item.setdefault("kind", "paper")
    for line in Path("examples/paper_candidates.tsv").read_text(encoding="utf-8").splitlines():
        arxiv, title = line.split("\t", 1)
        if arxiv in existing_ids:
            continue
        existing_ids.add(arxiv)
        records.append(
            dict(
                filename=f"arxiv-{arxiv}.pdf",
                title=title,
                language="en",
                kind="paper",
                external_id="arxiv:" + arxiv,
                topic="Machine learning research",
                url=f"https://arxiv.org/pdf/{arxiv}",
                page_url=f"https://arxiv.org/abs/{arxiv}",
            )
        )
    additions = [
        (
            "glove-2014.pdf",
            "GloVe: Global Vectors for Word Representation",
            "https://aclanthology.org/D14-1162.pdf",
            "https://aclanthology.org/D14-1162/",
            "en",
        ),
        (
            "ru-deep-learning-survey-2017.pdf",
            "Обзор методов обучения глубоких нейронных сетей",
            "https://www.mathnet.ru/links/6de0bde8d3b13bfbc255fb9b9042190d/vyurv170.pdf",
            "https://www.mathnet.ru/vyurv170",
            "ru",
        ),
        (
            "ru-xgboost-smote-2020.pdf",
            "Прогнозирование банкротств предприятий с помощью экстремального градиентного бустинга",
            "https://www.mathnet.ru/links/634d7cb93a8eaeb1ca2df8e42fded1f6/vyurv242.pdf",
            "https://www.mathnet.ru/vyurv242",
            "ru",
        ),
        (
            "ru-parallel-boosting-2013.pdf",
            "Параллельная реализация алгоритма градиентного бустинга деревьев решений",
            "https://www.mathnet.ru/php/getFT.phtml?jrnid=vmp&paperid=159&what=fullt",
            "https://www.mathnet.ru/vmp159",
            "ru",
        ),
        (
            "ru-boosting-loss-2022.pdf",
            "Метод повышения эффективности обучения градиентного бустинга, основанный на модифицированных функциях потерь",
            "https://www.mathnet.ru/php/getFT.phtml?jrnid=at&paperid=16098&what=fullt",
            "https://www.mathnet.ru/at16098",
            "ru",
        ),
    ]
    names = {p["filename"] for p in records}
    for name, title, url, page, language in additions:
        if name not in names:
            records.append(
                dict(
                    filename=name,
                    title=title,
                    url=url,
                    page_url=page,
                    language=language,
                    kind="paper",
                    topic="ML research",
                )
            )
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        repo = "esokolov/ml-course-hse"
        checkout = Path("data/sources/hse")
        if not checkout.exists():
            checkout.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--filter=blob:none",
                    "--sparse",
                    f"https://github.com/{repo}",
                    str(checkout),
                ],
                check=True,
            )
        revision = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        listing = subprocess.check_output(["git", "-C", str(checkout), "ls-tree", "-r", "HEAD"], text=True)
        tree = {
            "tree": [dict(path=line.split("\t", 1)[1], sha=line.split()[2]) for line in listing.splitlines()]
        }
        chosen = {}
        for row in tree["tree"]:
            name = row["path"]
            if "/lecture-notes/" not in name or not name.endswith(".pdf"):
                continue
            if not re.match(r"20\d\d-", name):
                continue
            subject = re.sub(r"^(?:lecture|lect|lec)\d*[-_]?", "", Path(name).stem)
            if subject not in chosen or name > chosen[subject]["path"]:
                chosen[subject] = row
        blobs = set()
        for subject, row in sorted(chosen.items()):
            if row["sha"] in blobs:
                continue
            blobs.add(row["sha"])
            source = row["path"]
            title = "Конспект ВШЭ: " + subject
            tex = client.get(f"https://raw.githubusercontent.com/{repo}/{revision}/{source[:-4]}.tex")
            if tex.status_code == 200:
                match = re.search(r"\\title\{(.+?)\}", tex.text, re.S)
                if match:
                    title = re.sub(r"\\[A-Za-z]+|[{}]|Лекция\s*\d+|\\+", " ", match[1])
                    title = " ".join(title.split())
            name = "ru-hse-" + hashlib.sha256(subject.encode()).hexdigest()[:12] + ".pdf"
            if name not in names:
                records.append(
                    dict(
                        filename=name,
                        title=title,
                        language="ru",
                        kind="lecture",
                        topic="Классическое ML / конспект лекции",
                        revision=revision,
                        external_id="hse:" + subject,
                        url=f"https://raw.githubusercontent.com/{repo}/{revision}/{source}",
                        page_url=f"https://github.com/{repo}/blob/{revision}/{source}",
                    )
                )
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            dict(candidates=len(records), russian=sum(p["language"] == "ru" for p in records)), indent=2
        )
    )


if __name__ == "__main__":
    main()
