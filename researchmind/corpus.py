"""Download the curated open paper collection and ingest into the active agent index."""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import httpx

from .config import Settings
from .ingest import parse_document
from .store import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("examples/papers.json"))
    parser.add_argument("--api", help="Running API base URL; otherwise writes the persistent local index")
    parser.add_argument(
        "--download-only", action="store_true", help="Cache and validate without opening Qdrant"
    )
    parser.add_argument("--delay", type=float, default=3.0, help="Minimum seconds between arXiv requests")
    args = parser.parse_args()
    settings = Settings()
    folder = settings.data_dir / "papers"
    folder.mkdir(parents=True, exist_ok=True)
    papers = json.loads(args.manifest.read_text(encoding="utf-8"))
    store = None
    report, errors, duplicates = [], [], []
    hashes, identities = {}, set()
    cache_path = folder / "downloads.json"
    previous = (
        {r["filename"]: r for r in json.loads(cache_path.read_text(encoding="utf-8"))}
        if cache_path.exists()
        else {}
    )
    last_arxiv = 0.0
    try:
        with httpx.Client(
            timeout=httpx.Timeout(120, connect=15),
            follow_redirects=True,
            headers={"User-Agent": "ResearchMind/0.3 (personal ML research library)"},
        ) as client:
            for paper in papers:
                if paper.get("enabled") is False:
                    continue
                name = paper["filename"]
                if Path(name).name != name or not name.endswith(".pdf"):
                    raise ValueError("Manifest filenames must be plain PDF filenames")
                identity = paper.get("external_id") or re.sub(r"[^\w]", "", paper["title"].casefold())
                if identity in identities:
                    duplicates.append(dict(source=name, reason="same canonical identity"))
                    continue
                identities.add(identity)
                path = folder / name
                try:
                    if path.exists():
                        content = path.read_bytes()
                    else:
                        urls = [paper["url"], *paper.get("fallback_urls", [])]
                        for position, url in enumerate(urls):
                            try:
                                if "arxiv.org" in url:
                                    time.sleep(max(0, args.delay - (time.monotonic() - last_arxiv)))
                                    last_arxiv = time.monotonic()
                                response = client.get(url)
                                response.raise_for_status()
                                content = response.content
                                if not content.startswith(b"%PDF-"):
                                    raise ValueError("Response is not a PDF")
                                break
                            except (httpx.HTTPError, ValueError):
                                if position == len(urls) - 1:
                                    raise
                    if not content.startswith(b"%PDF-"):
                        raise ValueError("Response is not a PDF")
                    digest = hashlib.sha256(content).hexdigest()
                    if digest in hashes:
                        duplicates.append(
                            dict(source=name, duplicate_of=hashes[digest], reason="identical PDF hash")
                        )
                        continue
                    cached = previous.get(name, {})
                    if cached.get("sha256") == digest and cached.get("title") == paper["title"]:
                        text_hash, page_count = cached["text_sha256"], cached["pages"]
                    else:
                        pages = parse_document(name, content)
                        if paper.get("external_id", "").startswith("arxiv:"):
                            expected = set(re.findall(r"[a-z]{3,}", paper["title"].lower())) - {
                                "the",
                                "and",
                                "for",
                                "with",
                                "from",
                                "using",
                            }
                            actual = set(re.findall(r"[a-z]{3,}", " ".join(t for _, t in pages[:2]).lower()))
                            if expected and len(expected & actual) / len(expected) < 0.4:
                                raise ValueError(
                                    "PDF text does not match expected title; requires manual review"
                                )
                        page_count = len(pages)
                        text_hash = hashlib.sha256(
                            " ".join(" ".join(t for _, t in pages).split()).encode()
                        ).hexdigest()
                    if text_hash in hashes:
                        duplicates.append(
                            dict(
                                source=name, duplicate_of=hashes[text_hash], reason="identical extracted text"
                            )
                        )
                        continue
                    hashes[digest] = hashes[text_hash] = name
                    if not path.exists():
                        path.write_bytes(content)
                    metadata = {
                        k: paper[k]
                        for k in ("title", "language", "kind", "external_id", "topic")
                        if k in paper
                    }
                    metadata["source_url"] = paper["page_url"]
                    if args.download_only:
                        result = dict(source=name, document_id=digest, indexed=False)
                    elif args.api:
                        response = client.post(
                            args.api.rstrip("/") + "/documents",
                            files={"file": (name, content, "application/pdf")},
                            data=metadata,
                        )
                        response.raise_for_status()
                        result = response.json()
                    else:
                        if store is None:
                            store = Store(settings)
                        result = store.ingest(name, content, metadata)
                    entry = dict(
                        paper,
                        **result,
                        pages=page_count,
                        bytes=len(content),
                        sha256=digest,
                        text_sha256=text_hash,
                    )
                    report.append(entry)
                    print(f"{len(report)}/{len(papers)} {name}: {page_count} pages", flush=True)
                except (httpx.HTTPError, ValueError) as error:
                    errors.append(dict(source=name, url=paper["url"], error=str(error)))
                    print(f"FAILED {name}: {error}", flush=True)
                target = folder / ("downloads.json" if args.download_only else "index.json")
                target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                (folder / "import-status.json").write_text(
                    json.dumps(
                        dict(
                            unique=len(report),
                            candidates=len(papers),
                            excluded=[p["filename"] for p in papers if p.get("enabled") is False],
                            errors=errors,
                            duplicates=duplicates,
                            russian=sum(r.get("language") == "ru" for r in report),
                            terms="Original author/publisher terms retained; personal retrieval cache, not a redistribution license.",
                        ),
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
    finally:
        if store:
            store.client.close()
    print(f"Done: {len(report)} unique, {len(duplicates)} duplicate, {len(errors)} failed", flush=True)


if __name__ == "__main__":
    main()
