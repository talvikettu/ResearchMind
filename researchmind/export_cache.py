"""Export completed Qdrant embeddings for fast native/Docker re-use, without model inference."""

import hashlib
import json

import numpy as np
from qdrant_client import QdrantClient

from .config import Settings


def main():
    s = Settings()
    signature = s.embedding_model + f"|retrieval-prefix:v1|hierarchy:v2|words:{s.chunk_words}"
    collection = "research_" + hashlib.sha256(signature.encode()).hexdigest()[:12]
    client = (
        QdrantClient(url=s.qdrant_url)
        if s.qdrant_url
        else QdrantClient(path=str(s.qdrant_path or s.data_dir / "qdrant"))
    )
    try:
        grouped = {}
        offset = None
        while True:
            rows, offset = client.scroll(collection, limit=256, offset=offset, with_vectors=True)
            for row in rows:
                grouped.setdefault(row.payload["document_id"], []).append((row.payload, row.vector))
            if offset is None:
                break
        for doc, pairs in grouped.items():
            pairs.sort(key=lambda pair: pair[0].get("ordinal", 0))
            chunks = [p[0] for p in pairs]
            path = s.data_dir / "documents" / (doc + ".json")
            if not path.exists():
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            record.update(chunks=chunks, collection=collection, chunk_words=s.chunk_words)
            path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            inputs = [
                c.get("title", "") + "\n" + " > ".join(c.get("section_path", [])) + "\n" + c["text"]
                for c in chunks
            ]
            cache = s.data_dir / "embeddings" / collection / (doc + ".npz")
            cache.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(json.dumps(inputs, ensure_ascii=False).encode()).hexdigest()
            np.savez_compressed(
                cache,
                input_sha=np.array(digest),
                ids=np.array([c["id"] for c in chunks]),
                vectors=np.array([p[1] for p in pairs], dtype=np.float32),
            )
        print("Exported", len(grouped), "documents", flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
