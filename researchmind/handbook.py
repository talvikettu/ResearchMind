"""Cache public Yandex ML handbook chapters for a personal RAG library."""

import argparse
import hashlib
import json
import time
from collections import deque
from pathlib import Path
from urllib.parse import unquote, urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

from .config import Settings
from .library_sources import read_sources

BASE = "https://contest.yandex.ru/tracks/ml/"


def save_manifest(root, records, errors, pending):
    (root / "manifest.json").write_text(
        json.dumps(
            dict(
                source=BASE,
                chapters=records,
                errors=errors,
                terms="Original Yandex/author terms retained. Personal local retrieval cache; no open license claimed.",
                pending=pending,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def chapter_text(html):
    soup = BeautifulSoup(html, "html.parser")
    main = soup.select_one(".dc-doc-page__content")
    if main is None:
        raise ValueError("No handbook article found; refusing to index navigation or a challenge page")
    for math in main.select(".yfm-latex[data-content]"):
        # Yandex sends empty placeholders: TeX lives in the URL-encoded attribute.
        formula = " $" + unquote(math["data-content"]).strip() + "$ "
        if math.name == "p":
            math.clear()
            math.append(formula)
        else:
            math.replace_with(formula)
    for anchor in main.select(".yfm-anchor, .dc-doc-page__content-mini-toc"):
        anchor.decompose()
    for element in main.select("script, style, nav, footer, .katex-mathml"):
        # Keep TeX before removing the duplicate rendered representation below.
        if element.get("class") and "katex-mathml" in element.get("class", []):
            continue
        element.decompose()
    for math in main.select(".katex"):
        tex = math.find("annotation", attrs={"encoding": "application/x-tex"})
        if tex:
            math.replace_with(" $" + tex.get_text() + "$ ")
    title = main.find("h1")
    title = title.get_text(" ", strip=True) if title else soup.title.get_text()
    blocks = []
    for element in main.find_all(["h1", "h2", "h3", "h4", "p", "pre", "li", "table"]):
        if element.find_parent(["li", "pre", "table"]):
            continue
        text = element.get_text(" ", strip=True)
        if not text or text in {"Предыдущая", "Следующая"}:
            continue
        if element.name.startswith("h"):
            text = "#" * int(element.name[1]) + " " + text
        elif element.name == "pre":
            text = "```\n" + element.get_text() + "\n```"
        elif element.name == "li":
            text = "- " + text
        blocks.append(text)
    content = "\n\n".join(blocks)
    if len(content) < 150:
        raise ValueError("Chapter is empty or not an educational article")
    return title, content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--sources", type=Path, help="Download only handbook URLs from sources.txt")
    args = parser.parse_args()
    root = Settings().data_dir / "handbook"
    root.mkdir(parents=True, exist_ok=True)
    fixed = [r for r in read_sources(args.sources) if r["kind"] == "handbook"] if args.sources else None
    if fixed is not None and any(
        not row["url"].startswith(BASE)
        or row["filename"] != "handbook-" + hashlib.sha256(row["url"].encode()).hexdigest()[:16] + ".md"
        for row in fixed
    ):
        raise ValueError("Handbook sources must use public handbook URLs and matching filenames")
    queue = deque(row["url"] for row in fixed) if fixed is not None else deque([BASE])
    limit = len(fixed) if fixed is not None else args.limit
    visited = set()
    records = []
    errors = []
    with httpx.Client(
        timeout=httpx.Timeout(90, connect=20),
        follow_redirects=True,
        headers={"User-Agent": "ResearchMindPersonalLibrary/0.2"},
    ) as client:
        if not args.download_only:
            ready = client.get(args.api.rstrip("/") + "/health")
            ready.raise_for_status()
        robots = client.get("https://contest.yandex.ru/robots.txt")
        rules = None
        if robots.status_code == 200:
            rules = RobotFileParser()
            rules.parse(robots.text.splitlines())
            (root / "robots.txt").write_text(robots.text, encoding="utf-8")
        elif robots.status_code not in {404, 410}:
            robots.raise_for_status()
        while queue and len(records) < limit:
            url = queue.popleft()
            if url in visited:
                continue
            visited.add(url)
            if rules and not rules.can_fetch("ResearchMindPersonalLibrary", url):
                errors.append(dict(url=url, error="robots.txt disallows automated retrieval"))
                continue
            key = hashlib.sha256(url.encode()).hexdigest()[:16]
            raw = root / (key + ".html")
            try:
                if raw.exists():
                    html = raw.read_text(encoding="utf-8")
                else:
                    response = client.get(url)
                    response.raise_for_status()
                    if not str(response.url).startswith(BASE):
                        raise ValueError("Redirected outside the public handbook")
                    html = response.text
                    raw.write_text(html, encoding="utf-8")
                    time.sleep(0.3)
                soup = BeautifulSoup(html, "html.parser")
                base_element = soup.find("base", href=True)
                document_base = urljoin(url, base_element["href"]) if base_element else BASE
                for anchor in [] if fixed is not None else soup.find_all("a", href=True):
                    link = urldefrag(urljoin(document_base, anchor["href"]))[0]
                    relative = link[len(BASE) :] if link.startswith(BASE) else ""
                    if len(relative.strip("/").split("/")) == 2 and not any(
                        x in relative for x in ("_bundle", ".pdf", ".png")
                    ):
                        if urlparse(link).query == "" and link not in visited:
                            queue.append(link)
                if url == BASE:
                    continue
                title, text = chapter_text(html)
                name = "handbook-" + key + ".md"
                content = (f"# {title}\n\nИсточник: {url}\n\n" + text).encode("utf-8")
                (root / name).write_bytes(content)
                if args.download_only:
                    result_data = dict(
                        document_id=hashlib.sha256(content).hexdigest(), chunks=0, indexed=False
                    )
                else:
                    result = client.post(
                        args.api.rstrip("/") + "/documents",
                        files={"file": (name, content)},
                        data={"title": title, "source_url": url, "kind": "handbook", "language": "ru"},
                    )
                    result.raise_for_status()
                    result_data = result.json()
                records.append(
                    dict(
                        filename=name,
                        title=title,
                        url=url,
                        sha256=hashlib.sha256(content).hexdigest(),
                        **result_data,
                    )
                )
                print(f"{len(records)} chapters; {name}; {result_data['chunks']} chunks", flush=True)
            except (httpx.HTTPError, ValueError) as error:
                errors.append(dict(url=url, error=str(error)))
            save_manifest(root, records, errors, len(set(queue) - visited))
        # Persist errors even when the final URL was skipped by robots.txt.
        save_manifest(root, records, errors, len(set(queue) - visited))
    print(f"Cached and indexed {len(records)} chapters, {len(errors)} errors", flush=True)


if __name__ == "__main__":
    main()
