"""Rebuild a model-specific collection from cached originals, preserving old collections."""

import argparse
import hashlib
import json
from pathlib import Path

from .config import Settings
from .store import Store


def sources(settings):
    root = settings.data_dir
    result = {}
    catalog = Path("examples/papers.json")
    papers = json.loads(catalog.read_text(encoding="utf-8")) if catalog.exists() else []
    excluded_names = {p["filename"] for p in papers if p.get("enabled") is False}
    for paper in papers:
        if paper.get("enabled") is False:
            continue
        path = root / "papers" / paper["filename"]
        if path.is_file():
            metadata = {
                k: paper[k] for k in ("title", "language", "kind", "external_id", "topic") if k in paper
            }
            metadata["source_url"] = paper["page_url"]
            result[paper["filename"]] = path, metadata
    manifest = root / "handbook" / "manifest.json"
    if manifest.exists():
        for chapter in json.loads(manifest.read_text(encoding="utf-8"))["chapters"]:
            path = root / "handbook" / chapter["filename"]
            if path.is_file():
                result[chapter["filename"]] = (
                    path,
                    dict(title=chapter["title"], source_url=chapter["url"], kind="handbook", language="ru"),
                )
    registry = root / "documents" / "registry.json"
    if registry.exists():
        for name, doc in json.loads(registry.read_text(encoding="utf-8")).items():
            if name in result or name in excluded_names:
                continue
            record = json.loads((root / "documents" / (doc + ".json")).read_text(encoding="utf-8"))
            path = (root / record["raw_path"]).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Cached document path escapes data directory")
            if path.is_file():
                result[name] = path, record["metadata"]
    return result


def run(settings=None, force=False, force_kind=None):
    settings = settings or Settings()
    store = Store(settings)
    results, failures, duplicates = [], [], []
    try:
        existing = {c["document_id"] for c in store.all_chunks() if c.get("schema_version") == 2}
        seen = {}
        for name, (path, metadata) in sources(settings).items():
            raw = path.read_bytes()
            doc = hashlib.sha256(raw).hexdigest()
            if doc in seen:
                duplicates.append(dict(source=name, duplicate_of=seen[doc]))
                continue
            seen[doc] = name
            try:
                if doc in existing and not force and metadata.get("kind") != force_kind:
                    result = dict(document_id=doc, source=name, cached=True)
                else:
                    result = store.ingest(name, raw, metadata)
                results.append(
                    dict(
                        **result,
                        language=metadata.get("language", "unknown"),
                        kind=metadata.get("kind", "document"),
                    )
                )
                print(f"{len(results)} indexed: {name}", flush=True)
            except (ValueError, OSError) as error:
                failures.append(dict(source=name, error=str(error)))
                print(f"FAILED {name}: {error}", flush=True)
        report = dict(
            collection=store.collection,
            embedding_model=settings.embedding_model,
            documents=len(results),
            chunks=len(store.all_chunks()),
            russian=sum(r["language"] == "ru" for r in results),
            results=results,
            duplicates=duplicates,
            failures=failures,
        )
        (settings.data_dir / "index-status.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {k: v for k, v in report.items() if k not in {"results", "failures", "duplicates"}},
                ensure_ascii=True,
            ),
            flush=True,
        )
        if failures:
            raise RuntimeError(f"{len(failures)} documents failed; see data/index-status.json")
        return report
    finally:
        store.client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None, help="Temporary embedding device for bulk indexing")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-kind", choices=["paper", "lecture", "handbook"])
    args = parser.parse_args()
    config = Settings()
    if args.device:
        config.retrieval_device = args.device
    config.reranker = "lexical"  # Reranking is unnecessary during bulk ingestion.
    run(config, args.force, args.force_kind)
