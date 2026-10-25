import csv
import hashlib
import json

import httpx
import pytest

from researchmind.library_sources import FIELDS, read_sources


def test_public_catalog_has_unique_documents_and_download_links():
    rows = read_sources("sources.txt")
    assert len(rows) == len({row["filename"] for row in rows})
    assert len(rows) >= 298
    assert sum(row["kind"] == "handbook" for row in rows) >= 73
    assert all(row["title"] and row["url"] and row["page_url"] for row in rows)


@pytest.mark.parametrize("name", ["../bad.pdf", "..\\bad.pdf", "bad.md"])
def test_catalog_rejects_unsafe_or_wrong_pdf_names(tmp_path, name):
    path = tmp_path / "sources.txt"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            dict(
                title="Paper",
                page_url="https://example.org",
                url="https://example.org/a.pdf",
                filename=name,
                kind="paper",
                language="en",
            )
        )
    with pytest.raises(ValueError):
        read_sources(path)


def test_handbook_reports_a_final_url_disallowed_by_robots(tmp_path, monkeypatch):
    from researchmind import handbook

    url = handbook.BASE + "introduction/about-this-book"
    filename = "handbook-" + hashlib.sha256(url.encode()).hexdigest()[:16] + ".md"
    path = tmp_path / "sources.txt"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            dict(title="Chapter", page_url=url, url=url, filename=filename, kind="handbook", language="ru")
        )
    client = httpx.Client
    monkeypatch.setattr(
        handbook.httpx,
        "Client",
        lambda **kwargs: client(
            **kwargs,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, text="User-agent: *\nDisallow: /")
            ),
        ),
    )
    monkeypatch.setenv("RESEARCH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr("sys.argv", ["handbook", "--sources", str(path), "--download-only"])
    handbook.main()
    manifest = json.loads((tmp_path / "data/handbook/manifest.json").read_text(encoding="utf-8"))
    assert manifest["chapters"] == []
    assert len(manifest["errors"]) == 1
    assert "robots.txt" in manifest["errors"][0]["error"]
