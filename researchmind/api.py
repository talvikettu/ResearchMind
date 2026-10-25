from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import Settings
from .engine import Engine
from .store import Store


def create_app(settings: Settings | None = None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.store = Store(settings or Settings())
        app.state.engine = Engine(app.state.store)
        yield
        app.state.store.client.close()

    app = FastAPI(title="ResearchMind", lifespan=lifespan)

    @app.get("/health")
    def health():
        s = app.state.store.settings
        return {
            "status": "ok",
            "embeddings": s.embeddings,
            "llm": bool(s.llm_url),
            "answer_backend": s.answer_backend,
            "answer_model": s.answer_model if s.answer_backend == "ollama" else None,
            "embedding_model": s.embedding_model,
            "reranker": s.reranker,
            "reranker_model": s.reranker_model,
            "collection": app.state.store.collection,
            "tuning": {
                key: getattr(s, key)
                for key in (
                    "search_pool",
                    "rerank_candidates",
                    "dense_weight",
                    "retrieval_k",
                    "evidence_limit",
                    "per_document",
                    "parent_max_chars",
                    "answer_profile",
                    "answer_context",
                    "answer_temperature",
                    "answer_sources",
                    "answer_max_tokens",
                    "entity_routing",
                    "technical_expansion",
                    "handbook_boost",
                    "title_weight",
                    "collapse_parents",
                    "answer_source_first",
                    "strict_entity_evidence",
                    "teaching_precision",
                )
            },
        }

    @app.get("/documents")
    def documents():
        docs = {}
        for c in app.state.store.all_chunks():
            entry = docs.setdefault(
                c["document_id"],
                dict(
                    document_id=c["document_id"],
                    source=c["source"],
                    chunks=0,
                    title=c.get("title", c["source"]),
                    language=c.get("language", "unknown"),
                    kind=c.get("kind", "document"),
                    source_url=c.get("source_url", ""),
                ),
            )
            entry["chunks"] += 1
        return list(docs.values())

    @app.get("/papers/{filename}")
    def paper(filename: str):
        root = (app.state.store.settings.data_dir / "papers").resolve()
        path = (root / filename).resolve()
        if path.parent != root or path.suffix.lower() != ".pdf" or not path.is_file():
            raise HTTPException(404, "Paper not found")
        return FileResponse(
            path, media_type="application/pdf", filename=path.name, content_disposition_type="inline"
        )

    @app.post("/documents")
    async def upload(
        file: UploadFile = File(...),
        title: str = Form(""),
        source_url: str = Form(""),
        kind: str = Form("document"),
        language: str = Form(""),
        external_id: str = Form(""),
        topic: str = Form(""),
    ):
        content = await file.read(25 * 1024 * 1024 + 1)
        if len(content) > 25 * 1024 * 1024:
            raise HTTPException(413, "Maximum file size is 25 MB")
        try:
            if source_url and not source_url.startswith(("http://", "https://")):
                raise ValueError("Source URL must use HTTP(S)")
            return await run_in_threadpool(
                app.state.store.ingest,
                Path(file.filename or "document.pdf").name,
                content,
                {
                    "title": title[:500],
                    "source_url": source_url[:2000],
                    "kind": kind[:40],
                    "language": language[:20],
                    "external_id": external_id[:200],
                    "topic": topic[:300],
                },
            )
        except (ValueError, UnicodeError) as error:
            raise HTTPException(400, str(error)) from error

    @app.post("/query")
    def query(request: Query):
        try:
            return app.state.engine.query(
                request.question, request.mode, request.allow_web, request.detail, request.answer_mode
            )
        except Exception as error:
            import logging

            logging.exception("Query failed")
            raise HTTPException(502, "Query failed; check backend logs and provider configuration") from error

    return app


class Query(BaseModel):
    question: str = Field(min_length=3, max_length=4000, pattern=r"\S")
    mode: Literal["naive", "hybrid", "agentic"] = "agentic"
    allow_web: bool = False
    detail: Literal["short", "standard", "detailed"] = "standard"
    answer_mode: Literal["auto", "explain", "formula", "compare", "research"] = "auto"


app = create_app()
