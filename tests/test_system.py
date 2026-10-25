from fastapi.testclient import TestClient

from researchmind.api import create_app
from researchmind.config import Settings
from researchmind.engine import Engine
from researchmind.evaluate import retrieval_metrics
from researchmind.ingest import chunk_document
from researchmind.store import Store


def test_chunk_bounds_and_stable_ids():
    content = ("A sentence about adapters. " * 500).encode()
    chunks = chunk_document("paper.txt", content, size=60, overlap=10)
    assert len(chunks) > 1
    assert all(len(c["text"].split()) <= 60 for c in chunks)
    assert chunks == chunk_document("paper.txt", content, size=60, overlap=10)
    assert all(c["page"] == 1 for c in chunks)


def test_ingestion_search_persistence_and_modes(tmp_path):
    settings = Settings(data_dir=tmp_path, _env_file=None)
    store = Store(settings)
    store.ingest("lora.txt", b"LoRA trains low rank adapters and freezes base weights.")
    store.ingest("qlora.txt", b"QLoRA uses NormalFloat four bit quantization to reduce memory.")
    store.ingest("lora.txt", b"LoRA trains low rank adapters and freezes base weights.")
    assert len(store.all_chunks()) == 2
    for mode in ("naive", "hybrid", "agentic"):
        result = Engine(store).query("NormalFloat quantization", mode)
        assert result["sources"][0]["source"] == "qlora.txt"
        assert result["verification"]["citation_ids_valid"]
        assert result["llm_cost_usd"] == 0
    store.client.close()
    reopened = Store(settings)
    assert len(reopened.all_chunks()) == 2
    reopened.client.close()


def test_api_empty_upload_query_and_validation(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path, _env_file=None))) as client:
        assert client.get("/health").status_code == 200
        empty = client.post("/query", json={"question": "unknown subject"}).json()
        assert not empty["verification"]["passed"]
        assert len([x for x in empty["trace"] if x["step"] == "plan"]) == 1
        assert any(x.get("stop_reason") == "no_new_evidence" for x in empty["trace"])
        assert client.post("/documents", files={"file": ("empty.txt", b"")}).status_code == 400
        assert (
            client.post("/documents", files={"file": ("paper.txt", b"LoRA trains adapters.")}).status_code
            == 200
        )
        assert len(client.get("/documents").json()) == 1
        assert client.post("/query", json={"question": "LoRA adapters", "mode": "hybrid"}).status_code == 200
        assert client.post("/query", json={"question": "foo", "mode": "invalid"}).status_code == 422


def test_metrics_do_not_overcount_chunks():
    result = retrieval_metrics([{"source": "a"}, {"source": "a"}, {"source": "b"}], ["a", "b"])
    assert result == {"recall_at_5": 1.0, "mrr_at_5": 1.0}


def test_changed_upload_replaces_version_and_invalidates_cache(tmp_path):
    store = Store(Settings(data_dir=tmp_path, _env_file=None))
    store.ingest("chapter.md", b"Old machine learning chapter.")
    cached = store.all_chunks()
    cached[0]["text"] = "Corrupted caller copy"
    assert store.all_chunks()[0]["text"] != "Corrupted caller copy"
    store.ingest(
        "chapter.md",
        b"Updated regression chapter with new evidence.",
        {"title": "Regression", "source_url": "https://example.org/chapter", "kind": "handbook"},
    )
    chunks = store.all_chunks()
    assert len(chunks) == 1
    assert chunks[0]["text"].startswith("Updated")
    assert chunks[0]["title"] == "Regression"
    store.client.close()


def test_verifier_rejects_fabricated_reference(tmp_path):
    store = Store(Settings(data_dir=tmp_path, _env_file=None))
    result = Engine(store).verify(dict(answer="Claim [99]", contexts=[{"text": "evidence"}], trace=[]))
    assert not result["verification"]["passed"]
    store.client.close()


def test_llm_tool_plan_and_retry_on_unsupported_claim(tmp_path, monkeypatch):
    import json

    store = Store(Settings(data_dir=tmp_path, llm_url="http://test/v1", _env_file=None))
    store.ingest("paper.txt", b"LoRA trains low rank adapters.")
    engine = Engine(store)
    checks = []

    def fake_chat(messages, state, tools=None):
        if tools:
            return {
                "tool_calls": [
                    {
                        "function": {
                            "name": "search",
                            "arguments": json.dumps({"query": "LoRA adapters", "source": "local"}),
                        }
                    }
                ]
            }
        if "Check each" in messages[0]["content"]:
            checks.append(1)
            return {"content": json.dumps({"supported": len(checks) > 1, "reason": "test judgement"})}
        return {"content": "LoRA trains adapters [1]"}

    monkeypatch.setattr(engine, "chat", fake_chat)
    result = engine.query("Explain LoRA adapters")
    assert result["verification"]["passed"]
    assert len(checks) == 2
    assert len([x for x in result["trace"] if x["step"] == "plan"]) == 2
    assert result["llm_cost_usd"] is None
    store.client.close()


def test_russian_named_query_targets_paper_before_retrieval(tmp_path):
    store = Store(Settings(data_dir=tmp_path, _env_file=None))
    store.ingest(
        "turboquant-2025.pdf.txt", b"Random rotation improves vector quantization and memory storage."
    )
    store.ingest("other.txt", b"Memory storage training quality quantization TurboQuant related work.")
    result = Engine(store).query("Как TurboQuant уменьшает память?")
    assert result["articles"]
    assert {a["source"] for a in result["articles"]} == {"turboquant-2025.pdf.txt"}
    store.client.close()
