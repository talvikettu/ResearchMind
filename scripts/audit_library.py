"""Audit declared Russian documents, recover reversible font mappings, quarantine unusable text."""

import hashlib
import json
import re
from pathlib import Path

from qdrant_client import QdrantClient, models

from researchmind.config import Settings
from researchmind.ingest import parse_document, repair_pdf_text


def main():
    s = Settings()
    manifest = Path("examples/papers.json")
    catalog = json.loads(manifest.read_text(encoding="utf-8"))
    by_name = {r["filename"]: r for r in catalog}
    path = s.data_dir / "papers" / "downloads.json"
    downloads = json.loads(path.read_text(encoding="utf-8"))
    audit = dict(font_repaired=[], language_corrected=[], excluded=[])
    for item in downloads:
        if item.get("language") != "ru":
            continue
        record_path = s.data_dir / "documents" / (item["document_id"] + ".json")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        text = " ".join(section["text"] for section in record["sections"])
        repaired = repair_pdf_text(text)
        if repaired != text:
            pages = parse_document(item["filename"], (s.data_dir / "papers" / item["filename"]).read_bytes())
            item["text_sha256"] = hashlib.sha256(
                " ".join(" ".join(t for _, t in pages).split()).encode()
            ).hexdigest()
            record.pop("chunks", None)
            record_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            audit["font_repaired"].append(dict(source=item["filename"], document_id=item["document_id"]))
            continue
        ru = len(re.findall("[а-яА-ЯёЁ]", text))
        en = len(re.findall("[a-zA-Z]", text))
        if ru / max(ru + en, 1) >= 0.2:
            continue
        if len(re.findall(r"\b(?:the|and|with|learning|data|using|neural|model)\b", text, re.I)) > 20:
            item["language"] = by_name[item["filename"]]["language"] = "en"
            record["metadata"]["language"] = "en"
            for chunk in record.get("chunks", []):
                chunk["language"] = "en"
            record_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            audit["language_corrected"].append(
                dict(source=item["filename"], document_id=item["document_id"], language="en")
            )
        else:
            by_name[item["filename"]].update(
                enabled=False,
                exclude_reason="Unreliable text/font extraction; needs OCR or a better original",
            )
            audit["excluded"].append(dict(source=item["filename"], document_id=item["document_id"]))
    excluded = {r["source"] for r in audit["excluded"]}
    downloads = [r for r in downloads if r["filename"] not in excluded]
    manifest.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    path.write_text(json.dumps(downloads, ensure_ascii=False, indent=2), encoding="utf-8")
    (s.data_dir / "library-audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    client = QdrantClient(path=str(s.qdrant_path or s.data_dir / "qdrant"))
    signature = s.embedding_model + f"|retrieval-prefix:v1|hierarchy:v2|words:{s.chunk_words}"
    collection = "research_" + hashlib.sha256(signature.encode()).hexdigest()[:12]
    try:
        for row in audit["font_repaired"] + audit["excluded"]:
            client.delete(
                collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id", match=models.MatchValue(value=row["document_id"])
                            )
                        ]
                    )
                ),
                wait=True,
            )
        for row in audit["language_corrected"]:
            client.set_payload(
                collection,
                payload={"language": "en"},
                points=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id", match=models.MatchValue(value=row["document_id"])
                        )
                    ]
                ),
                wait=True,
            )
    finally:
        client.close()
    print(json.dumps(audit, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
