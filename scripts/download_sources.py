"""Download the exact sources.txt collection; no LLM or running API required."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from researchmind.library_sources import read_sources  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=ROOT / "sources.txt")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--kind", choices=("all", "pdf", "handbook"), default="all")
    parser.add_argument("--dry-run", action="store_true", help="Validate/list counts without downloads")
    args = parser.parse_args()
    source_path = args.sources.resolve()
    rows = read_sources(source_path)
    pdfs = [row for row in rows if row["kind"] != "handbook"]
    chapters = [row for row in rows if row["kind"] == "handbook"]
    print(f"Catalog: {len(pdfs)} PDFs + {len(chapters)} handbook chapters", flush=True)
    if args.dry_run:
        return
    data = args.data_dir.resolve()
    data.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, RESEARCH_DATA_DIR=str(data), PYTHONIOENCODING="utf-8")
    if args.kind in {"all", "pdf"} and pdfs:
        catalog = json.loads((ROOT / "examples/papers.json").read_text(encoding="utf-8"))
        metadata = {row["filename"]: row for row in catalog}
        manifest = data / "library-papers.json"
        manifest.write_text(
            json.dumps([dict(metadata.get(row["filename"], {}), **row) for row in pdfs], ensure_ascii=False),
            encoding="utf-8",
        )
        subprocess.run(
            [sys.executable, "-m", "researchmind.corpus", "--manifest", str(manifest), "--download-only"],
            cwd=ROOT,
            env=env,
            check=True,
        )
        status = json.loads((data / "papers/import-status.json").read_text(encoding="utf-8"))
        if status["errors"]:
            raise SystemExit("Some PDFs failed; see papers/import-status.json. Rerun to retry.")
    if args.kind in {"all", "handbook"} and chapters:
        subprocess.run(
            [sys.executable, "-m", "researchmind.handbook", "--sources", str(source_path), "--download-only"],
            cwd=ROOT,
            env=env,
            check=True,
        )
        status = json.loads((data / "handbook/manifest.json").read_text(encoding="utf-8"))
        if status["errors"] or status["pending"]:
            raise SystemExit("Some chapters failed; see handbook/manifest.json. Rerun to retry.")
    print(f"Saved to {data}. Start the app to build/use the search index.")


if __name__ == "__main__":
    main()
