"""Read the portable UTF-8, one-document-per-line source catalog."""

import csv
from pathlib import Path
from urllib.parse import urlparse

FIELDS = ("title", "page_url", "url", "filename", "kind", "language")


def read_sources(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("Expected tab-separated columns: " + ", ".join(FIELDS))
        rows, names = [], set()
        for row in reader:
            name = row["filename"]
            if not name or Path(name).name != name or name in names or "/" in name or "\\" in name:
                raise ValueError(f"Invalid or duplicate filename: {name}")
            if row["kind"] not in {"paper", "lecture", "handbook"}:
                raise ValueError(f"Unknown document kind: {row['kind']}")
            expected = ".md" if row["kind"] == "handbook" else ".pdf"
            if not name.endswith(expected):
                raise ValueError(f"Expected {expected}: {name}")
            for key in ("url", "page_url"):
                url = urlparse(row[key])
                if url.scheme not in {"https", "http"} or not url.netloc:
                    raise ValueError(f"Invalid {key}: {row[key]}")
            names.add(name)
            rows.append(row)
    return rows
