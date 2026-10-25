"""Prepare models and a persistent index for native or Docker startup."""

import argparse
import json
import subprocess
import sys

import httpx

from .config import Settings


def models(settings):
    from huggingface_hub import snapshot_download

    repositories = []
    if settings.embeddings == "sentence-transformers":
        repositories.append(settings.embedding_model)
    if settings.reranker == "cross-encoder":
        repositories.append(settings.reranker_model)
    revisions = {}
    for repository in repositories:
        folder = snapshot_download(
            repository,
            cache_dir=str(settings.model_cache),
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.model",
                "*.safetensors",
                "pytorch_model.bin",
                "1_Pooling/*",
                "README.md",
            ],
        )
        revisions[repository] = folder.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if settings.answer_backend == "ollama":
        with httpx.Client(timeout=httpx.Timeout(600, connect=10)) as client:
            response = client.get(settings.ollama_url.rstrip("/") + "/api/tags")
            response.raise_for_status()
            present = any(m["name"] == settings.answer_model for m in response.json().get("models", []))
            if not present:
                with client.stream(
                    "POST",
                    settings.ollama_url.rstrip("/") + "/api/pull",
                    json=dict(model=settings.answer_model, stream=True),
                ) as response:
                    response.raise_for_status()
                    previous = -1
                    for line in response.iter_lines():
                        if not line:
                            continue
                        row = json.loads(line)
                        if row.get("error"):
                            raise RuntimeError(row["error"])
                        percent = int(row.get("completed", 0) / max(row.get("total", 1), 1) * 100)
                        if percent >= previous + 10 or row.get("status") == "success":
                            print(row.get("status"), percent, flush=True)
                            previous = percent
            response = client.get(settings.ollama_url.rstrip("/") + "/api/tags")
            response.raise_for_status()
            revisions[settings.answer_model] = next(
                m["digest"] for m in response.json()["models"] if m["name"] == settings.answer_model
            )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "model-revisions.json").write_text(json.dumps(revisions, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-only", action="store_true")
    parser.add_argument("--download", action="store_true", help="Refresh library downloads before indexing")
    args = parser.parse_args()
    settings = Settings()
    models(settings)
    if args.models_only:
        return
    if args.download or not (settings.data_dir / "papers" / "downloads.json").exists():
        subprocess.run([sys.executable, "-m", "researchmind.corpus", "--download-only"], check=True)
    if args.download or not (settings.data_dir / "handbook" / "manifest.json").exists():
        subprocess.run([sys.executable, "-m", "researchmind.handbook", "--download-only"], check=True)
    from .reindex import run

    settings.reranker = "lexical"
    run(settings)


if __name__ == "__main__":
    main()
