import hashlib
import json
import threading
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

from .config import Settings
from .evidence import excerpt, lexical_tokens, query_terms, reference_block, search_query
from .ingest import prepare_document, tokens


class Store:
    def __init__(self, settings: Settings):
        self.settings = settings
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.encoder = None
        self.cross_encoder = None
        self._chunks_cache = None
        self._bm25_cache = {}
        self._parents = {}
        self._entity_cache = {}
        manifest = Path("examples/papers.json")
        self.metadata = (
            {p["filename"]: p for p in json.loads(manifest.read_text(encoding="utf-8"))}
            if manifest.exists()
            else {}
        )
        if settings.embeddings == "sentence-transformers":
            import torch

            torch.set_num_threads(4)
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(
                settings.embedding_model,
                device=settings.retrieval_device,
                cache_folder=str(settings.model_cache),
            )
            self.encoder.max_seq_length = 512
            if settings.retrieval_device.startswith("cuda"):
                self.encoder.half()
        elif settings.embeddings != "demo":
            raise ValueError("embeddings must be demo or sentence-transformers")
        if settings.reranker == "cross-encoder":
            from sentence_transformers import CrossEncoder

            self.cross_encoder = CrossEncoder(
                settings.reranker_model,
                device=settings.retrieval_device,
                cache_folder=str(settings.model_cache),
                max_length=512,
            )
        elif settings.reranker != "lexical":
            raise ValueError("reranker must be lexical or cross-encoder")
        self.dimension = int(self.encoder.get_sentence_embedding_dimension()) if self.encoder else 512
        signature = (
            (settings.embedding_model + f"|retrieval-prefix:v1|hierarchy:v2|words:{settings.chunk_words}")
            if self.encoder
            else "demo-hash-v1"
        )
        self.collection = "research_" + hashlib.sha256(signature.encode()).hexdigest()[:12]
        self.client = (
            QdrantClient(url=settings.qdrant_url)
            if settings.qdrant_url
            else QdrantClient(path=str(settings.qdrant_path or settings.data_dir / "qdrant"))
        )
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=self.dimension, distance=models.Distance.COSINE),
            )

    def embed(self, texts: list[str], query=False) -> list[list[float]]:
        if self.encoder:
            if "e5" in self.settings.embedding_model.lower():
                texts = [("query: " if query else "passage: ") + t for t in texts]
            return self.encoder.encode(
                texts,
                normalize_embeddings=True,
                batch_size=self.settings.embedding_batch,
                show_progress_bar=False,
            ).tolist()
        vectors = []
        for text in texts:
            vector = np.zeros(self.dimension)
            for token in tokens(text):
                digest = hashlib.sha256(token.encode()).digest()
                vector[int.from_bytes(digest[:4], "little") % self.dimension] += 1 if digest[4] % 2 else -1
            vector /= max(float(np.linalg.norm(vector)), 1e-10)
            vectors.append(vector.tolist())
        return vectors

    def all_chunks(self) -> list[dict]:
        with self.lock:
            if self._chunks_cache is not None:
                return [dict(c) for c in self._chunks_cache]
            result, offset = [], None
            while True:
                records, offset = self.client.scroll(self.collection, limit=256, offset=offset)
                result.extend(dict(r.payload, id=str(r.id)) for r in records)
                if offset is None:
                    self._chunks_cache = result
                    return [dict(c) for c in result]

    def ingest(self, name: str, content: bytes, metadata: dict | None = None) -> dict:
        doc_id = hashlib.sha256(content).hexdigest()
        cached_record = self.settings.data_dir / "documents" / (doc_id + ".json")
        cached = json.loads(cached_record.read_text(encoding="utf-8")) if cached_record.exists() else {}
        if (
            cached.get("collection") == self.collection
            and cached.get("chunks")
            and not name.lower().endswith(".md")
            and cached.get("chunk_words") == self.settings.chunk_words
        ):
            prepared = dict(document_id=doc_id, chunks=cached["chunks"], sections=cached["sections"])
        else:
            prepared = prepare_document(name, content, size=self.settings.chunk_words)
        chunks = prepared["chunks"]
        if not chunks:
            raise ValueError("No text chunks found")
        with self.lock:
            for ordinal, c in enumerate(chunks):
                c["ordinal"] = ordinal
                c.update(
                    {
                        key: value
                        for key, value in (metadata or {}).items()
                        if key in {"title", "source_url", "kind", "language", "external_id", "topic"}
                        and value
                    }
                )
            inputs = [
                c.get("title", "") + "\n" + " > ".join(c.get("section_path", [])) + "\n" + c["text"]
                for c in chunks
            ]
            input_sha = hashlib.sha256(json.dumps(inputs, ensure_ascii=False).encode()).hexdigest()
            cache_path = self.settings.data_dir / "embeddings" / self.collection / (doc_id + ".npz")
            vectors = None
            if cache_path.exists():
                with np.load(cache_path, allow_pickle=False) as saved:
                    if str(saved["input_sha"]) == input_sha and saved["ids"].tolist() == [
                        c["id"] for c in chunks
                    ]:
                        vectors = saved["vectors"].tolist()
            if vectors is None:
                vectors = self.embed(inputs)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    cache_path,
                    input_sha=np.array(input_sha),
                    ids=np.array([c["id"] for c in chunks]),
                    vectors=np.array(vectors, dtype=np.float32),
                )
            for start in range(0, len(chunks), 128):
                self.client.upsert(
                    self.collection,
                    points=[
                        models.PointStruct(id=c["id"], vector=v, payload=c)
                        for c, v in zip(chunks[start : start + 128], vectors[start : start + 128])
                    ],
                    wait=True,
                )
            # A changed upload replaces the previous version under the same filename.
            # Upsert first, so a failed embedding/upload cannot erase existing evidence.
            self.client.delete(
                self.collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[models.FieldCondition(key="source", match=models.MatchValue(value=name))],
                        must_not=[models.HasIdCondition(has_id=[c["id"] for c in chunks])],
                    )
                ),
                wait=True,
            )
            self._chunks_cache = None
            self._bm25_cache.clear()
            self._entity_cache.clear()
            directory = self.settings.data_dir / "documents"
            directory.mkdir(parents=True, exist_ok=True)
            raw_path = None
            for folder in ("papers", "handbook"):
                candidate = self.settings.data_dir / folder / name
                if (
                    candidate.is_file()
                    and hashlib.sha256(candidate.read_bytes()).hexdigest() == prepared["document_id"]
                ):
                    raw_path = candidate
                    break
            if raw_path is None:
                raw_path = directory / (prepared["document_id"] + Path(name).suffix.lower())
                raw_path.write_bytes(content)
            record = dict(
                document_id=prepared["document_id"],
                source=name,
                metadata=metadata or {},
                raw_path=str(raw_path.relative_to(self.settings.data_dir)),
                schema_version=2,
                sections=prepared["sections"],
                chunks=chunks,
                collection=self.collection,
                chunk_words=self.settings.chunk_words,
            )
            (directory / (prepared["document_id"] + ".json")).write_text(
                json.dumps(record, ensure_ascii=False), encoding="utf-8"
            )
            registry_path = directory / "registry.json"
            registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
            registry[name] = prepared["document_id"]
            registry_path.write_text(json.dumps(registry, ensure_ascii=False), encoding="utf-8")
            self._parents[prepared["document_id"]] = {s["id"]: s for s in prepared["sections"]}
        return {"document_id": chunks[0]["document_id"], "chunks": len(chunks), "source": name}

    def neighbors(self, chunk: dict, radius: int = 1) -> list[dict]:
        if "ordinal" not in chunk:
            return []
        return [
            c
            for c in self.all_chunks()
            if c["document_id"] == chunk["document_id"]
            and c.get("parent_id") == chunk.get("parent_id")
            and 0 < abs(c.get("ordinal", -100) - chunk["ordinal"]) <= radius
        ]

    def parent(self, chunk):
        doc = chunk["document_id"]
        if doc not in self._parents:
            path = self.settings.data_dir / "documents" / (doc + ".json")
            if not path.exists():
                return None
            self._parents[doc] = {
                s["id"]: s for s in json.loads(path.read_text(encoding="utf-8"))["sections"]
            }
        return self._parents[doc].get(chunk.get("parent_id"))

    def search(self, query: str, mode: str = "hybrid", k: int = 5) -> list[dict]:
        original_query = query
        pool = max(k, self.settings.search_pool)
        entities = {"turboquant", "qlora", "lora", "xgboost", "ddpm"}
        synonyms = {
            "pca": "главных компонент principal components covariance eigenvectors",
            "svm": "опорных векторов support vector machines",
            "knn": "ближайших соседей nearest neighbors",
            "lasso": "лассо регуляризация",
            "ridge": "ридж регуляризация",
            "adamw": "decoupled weight decay",
        }
        original_terms = set(tokens(query))
        if self.settings.entity_routing:
            entities |= set(synonyms)
        if self.settings.strict_entity_evidence:
            import re

            if re.search(r"random\s+forests?|случайн\w*\s+лес", original_query, re.I):
                entities.add("randomforest")
                original_terms.add("randomforest")
            if re.search(r"k[-\s]?means", original_query, re.I):
                entities.add("kmeans")
                original_terms.add("kmeans")
            if not re.search(r"ранж|ranking|top.?k|@", original_query, re.I):
                entities |= {"precision", "recall"}
        named = original_terms & entities
        query = search_query(query)
        if self.settings.entity_routing:
            query += " " + " ".join(v for key, v in synonyms.items() if key in original_terms)
        technical = {
            "lasso": "L_1 L1 penalty sparse регуляризация разреженность",
            "ridge": "L_2 L2 penalty regularization регуляризация",
            "логистической": "logistic sigmoid log loss вероятность",
            "логистическую": "logistic sigmoid log loss вероятность",
            "logistic": "logistic sigmoid log loss вероятность",
            "sgd": "stochastic gradient descent mini batch стохастический градиентный спуск",
            "precision": "precision TP FP true positive false positive",
            "recall": "recall sensitivity TP FN false negative",
            "утечки": "data leakage preprocessing train validation",
            "утечку": "data leakage preprocessing train validation",
        }
        extra = " ".join(v for key, v in technical.items() if key in original_terms)
        rerank_query = original_query
        if self.settings.technical_expansion and extra:
            query += " " + extra
            rerank_query += "\nSearch terminology: " + extra
        with self.lock:
            chunks = [
                c for c in self.all_chunks() if not c.get("is_reference") and not reference_block(c["text"])
            ]
            if not chunks:
                return []
            matching = [c for c in chunks if named & set(tokens(c["source"]))]
            if self.settings.strict_entity_evidence and named:
                patterns = {
                    "pca": r"\bpca\b|главн\w*\s+компонент",
                    "lasso": r"\blasso\b|лассо|\bL_?1\b",
                    "ridge": r"\bridge\b|ридж|\bL_?2\b",
                    "svm": r"\bsvm\b|опорн\w*\s+вектор",
                    "knn": r"\bk.?nn\b|ближайш\w*\s+сосед",
                    "adamw": r"\badamw\b|decoupled",
                    "randomforest": r"random\s+forests?|случайн\w*\s+лес",
                    "kmeans": r"k[-\s]?means|k-средн",
                    "precision": r"\bTP\b|\bFP\b|ложнополож",
                    "recall": r"\bTP\b|\bFN\b|ложноотриц",
                }
                import re

                for entity in named & set(patterns):
                    if entity not in self._entity_cache:
                        pattern = re.compile(patterns[entity], re.I)
                        self._entity_cache[entity] = {c["id"] for c in chunks if pattern.search(c["text"])}
                eligible = {c["id"] for c in matching}
                eligible.update(set().union(*(self._entity_cache.get(name, set()) for name in named)))
                if eligible:
                    matching = [c for c in chunks if c["id"] in eligible]
            query_filter = None
            if matching:
                chunks = matching
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source", match=models.MatchAny(any=sorted({c["source"] for c in chunks}))
                        )
                    ]
                )
                if self.settings.strict_entity_evidence:
                    query_filter = models.Filter(
                        must=[models.HasIdCondition(has_id=[c["id"] for c in chunks])]
                    )
            dense = self.client.query_points(
                self.collection,
                query=self.embed([original_query], query=True)[0],
                limit=pool * 3,
                query_filter=query_filter,
            ).points
            eligible_ids = {c["id"] for c in chunks}
            dense_rows = [dict(p.payload, score=float(p.score)) for p in dense if str(p.id) in eligible_ids][
                :pool
            ]
            if mode == "naive":
                return dense_rows[:k]
            key = tuple(sorted(named)) if matching else ()
            if key not in self._bm25_cache:
                corpus = [
                    lexical_tokens(
                        c["text"]
                        + " "
                        + c.get("section_title", "")
                        + " "
                        + c.get("title", self.metadata.get(c["source"], {}).get("title", ""))
                    )
                    or ("_",)
                    for c in chunks
                ]
                self._bm25_cache[key] = BM25Okapi(corpus)
            scores = self._bm25_cache[key].get_scores(lexical_tokens(query))
            bm25_scores = {c["id"]: max(float(score), 0) for c, score in zip(chunks, scores)}
            lexical = [chunks[i] for i in np.argsort(-scores)[:pool]]
            merged, ranks = {}, {}
            for rows, weight in (
                (dense_rows, 2 * self.settings.dense_weight),
                (lexical, 2 * (1 - self.settings.dense_weight)),
            ):
                for rank, row in enumerate(rows, 1):
                    merged[row["id"]] = row
                    ranks[row["id"]] = ranks.get(row["id"], 0) + weight / (self.settings.rrf_k + rank)
            candidates = sorted(merged.values(), key=lambda c: ranks[c["id"]], reverse=True)[
                : self.settings.rerank_candidates
            ]
            if self.cross_encoder:
                rerank = self.cross_encoder.predict(
                    [(rerank_query, c.get("section_title", "") + "\n" + c["text"]) for c in candidates],
                    batch_size=self.settings.reranker_batch,
                    show_progress_bar=False,
                )
            else:
                maximum = max(bm25_scores.values(), default=1) or 1
                rerank = [
                    excerpt(c["text"], query)[1] + bm25_scores.get(c["id"], 0) / maximum * 0.5
                    for c in candidates
                ]
            for c, score in zip(candidates, rerank):
                if self.settings.strict_entity_evidence and named:
                    c["entity_title_match"] = any(
                        re.search(patterns[name], c.get("title", "") + " " + c.get("section_title", ""), re.I)
                        for name in named & set(patterns)
                    )
                c["score"] = float(score) + ranks[c["id"]]
                if self.cross_encoder:
                    title_overlap = query_terms(original_query) & set(
                        lexical_tokens(c.get("section_title", "") + " " + c.get("title", ""))
                    )
                    prior = self.settings.handbook_boost * (
                        c.get("kind") == "handbook"
                    ) + self.settings.title_weight * min(3, len(title_overlap))
                    c["score"] += prior * 5
                    c["reranker_score"] = float(min(1, 1 / (1 + np.exp(-np.clip(score, -30, 30))) + prior))
            return sorted(candidates, key=lambda c: c["score"], reverse=True)[:k]
