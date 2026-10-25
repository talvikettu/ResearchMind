import hashlib
import io
import re
from uuid import NAMESPACE_URL, uuid5

from pypdf import PdfReader
from pypdf.errors import LimitReachedError, PdfReadError


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def repair_pdf_text(text):
    """Reverse CP1251-as-Latin1 PDF fonts only when Russian markers confirm it."""
    if len(re.findall(r"[À-ÿ]", text)) < 40 or len(re.findall(r"[а-яА-Я]", text)) > 30:
        return text
    if not re.search(r"Ëåêöè|Ñîêîë|Ââåäåí|êëàñ|íåéð|ìåòîä", text):
        return text
    result = []
    for part in re.split(r"([^\x00-\xff]+)", text):
        try:
            result.append(part.encode("latin1").decode("cp1251"))
        except (UnicodeEncodeError, UnicodeDecodeError):
            result.append(part)
    decoded = "".join(result)
    if (
        sum(
            word in decoded.lower()
            for word in ["лекция", "соколов", "введение", "данных", "метод", "для", "нейр"]
        )
        >= 2
    ):
        return decoded
    return text


def parse_document(name: str, content: bytes) -> list[tuple[int, str]]:
    if name.lower().endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(content))
        except PdfReadError as error:
            raise ValueError("Invalid or damaged PDF") from error
        if reader.is_encrypted:
            raise ValueError("Encrypted PDF is not supported")
        try:
            pages = [
                (i + 1, repair_pdf_text(page.extract_text() or "")) for i, page in enumerate(reader.pages)
            ]
        except (PdfReadError, LimitReachedError, RecursionError) as error:
            raise ValueError("PDF text extraction failed safely; review or OCR this document") from error
    elif name.lower().endswith((".txt", ".md")):
        pages = [(1, content.decode("utf-8-sig"))]
    else:
        raise ValueError("Supported formats: PDF, TXT, MD")
    if not any(text.strip() for _, text in pages):
        raise ValueError("No text found. Scanned PDFs require OCR before upload.")
    return pages


def chunk_document(name: str, content: bytes, size: int = 220, overlap: int = 40) -> list[dict]:
    return prepare_document(name, content, size, overlap)["chunks"]


def heading(line, markdown=False):
    line = line.strip()
    if markdown:
        match = re.match(r"^(#{1,6})\s+(.+)$", line)
        return (len(match[1]), match[2].strip()) if match else None
    if len(line) > 110 or len(line.split()) > 14:
        return None
    numbered = re.match(r"^(\d+(?:\.\d+)*\.?)[ \t]+([A-Za-zА-Яа-яЁё].+)$", line)
    if numbered and not re.search(r"[=<>\\]", line):
        return (min(6, numbered[1].rstrip(".").count(".") + 1), line)
    if re.fullmatch(
        r"abstract|introduction|conclusions?|references|discussion|введение|заключение|литература|список литературы",
        line,
        re.I,
    ):
        return 1, line
    return None


def prepare_document(name: str, content: bytes, size: int = 220, overlap: int = 40) -> dict:
    """Preserve section ancestry and atomic Markdown equations/code/table blocks.

    PDF hierarchy is inferred from numbered text-layer headings; no OCR or equation
    reconstruction is claimed for PDF glyphs that lack usable Unicode mappings.
    """
    if size < 20 or not 0 <= overlap < size:
        raise ValueError("Invalid chunk size/overlap")
    doc_id = hashlib.sha256(content).hexdigest()
    chunks, sections, stack = [], [], []
    markdown = name.lower().endswith(".md")
    current = None
    pending = []
    fence = False
    display_math = False

    def start_section(title, level, page):
        nonlocal current
        while stack and stack[-1][0] >= level:
            stack.pop()
        section_id = str(uuid5(NAMESPACE_URL, f"{doc_id}:section:{len(sections)}"))
        current = dict(
            id=section_id,
            title=title,
            path=[s[1] for s in stack] + [title],
            parent_id=stack[-1][2] if stack else None,
            page_start=page,
            page_end=page,
            blocks=[],
        )
        stack.append((level, title, section_id))
        sections.append(current)

    def flush(page):
        nonlocal pending
        text = "\n".join(pending).strip()
        pending = []
        if not text:
            return
        if current is None:
            start_section(name, 0, page)
        kind = (
            "code"
            if text.startswith("```")
            else "equation"
            if text.startswith("$")
            else "table"
            if text.startswith("|")
            else "paragraph"
        )
        current["blocks"].append(
            dict(id=f"{current['id']}:{len(current['blocks'])}", text=text, kind=kind, page=page)
        )
        current["page_end"] = page

    for page, text in parse_document(name, content):
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("```"):
                if not fence:
                    flush(page)
                fence = not fence
                pending.append(line)
                if not fence:
                    flush(page)
                continue
            if stripped.startswith("$$"):
                if not display_math:
                    flush(page)
                pending.append(line)
                if stripped.count("$$") >= 2:
                    flush(page)
                else:
                    display_math = not display_math
                    if not display_math:
                        flush(page)
                continue
            found = heading(line, markdown) if not fence and not display_math else None
            if found:
                flush(page)
                start_section(found[1], found[0], page)
            elif not stripped and not fence and not display_math:
                flush(page)
            else:
                pending.append(line)
        flush(page)
    for section in sections:
        section["text"] = "\n\n".join(b["text"] for b in section["blocks"])
        buffer = []

        def emit(items):
            if not items:
                return
            chunks.append(
                dict(
                    id=str(uuid5(NAMESPACE_URL, f"{doc_id}:v2:{section['id']}:{len(chunks)}")),
                    document_id=doc_id,
                    source=name,
                    page=items[0]["page"],
                    page_end=items[-1]["page"],
                    text="\n\n".join(b["text"] for b in items),
                    section_id=section["id"],
                    section_title=section["title"],
                    section_path=section["path"],
                    parent_id=section["id"],
                    paragraph_ids=[b["id"] for b in items],
                    block_types=sorted({b["kind"] for b in items}),
                    is_reference=bool(
                        re.search(
                            r"^(?:references|bibliography|список литературы|литература)$",
                            section["title"],
                            re.I,
                        )
                    ),
                    schema_version=2,
                )
            )

        for block in section["blocks"]:
            pieces = [block]
            words = block["text"].split()
            if len(words) > size and block["kind"] == "paragraph":
                if "$" in block["text"]:
                    atomic = re.findall(r"\$\$[\s\S]*?\$\$|\$[^$]+\$|\S+", block["text"])
                    groups, group, count = [], [], 0
                    for unit in atomic:
                        weight = len(unit.split())
                        if group and count + weight > size:
                            groups.append(" ".join(group))
                            group, count = [], 0
                        group.append(unit)
                        count += weight
                    if group:
                        groups.append(" ".join(group))
                    pieces = [dict(block, text=text) for text in groups]
                else:
                    pieces = [
                        dict(block, text=" ".join(words[i : i + size]))
                        for i in range(0, len(words), size - overlap)
                    ]
            for piece in pieces:
                if buffer and sum(len(b["text"].split()) for b in buffer) + len(piece["text"].split()) > size:
                    emit(buffer)
                    # Overlap only whole small paragraphs: never bisect an equation.
                    buffer = [buffer[-1]] if len(buffer[-1]["text"].split()) <= overlap else []
                buffer.append(piece)
        emit(buffer)
    return dict(document_id=doc_id, chunks=chunks, sections=sections, schema_version=2)
