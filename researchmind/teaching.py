"""Short explanations assembled from indexed handbook sections, with exact provenance.

This is a general section/paragraph extractor, not a bank of topic-specific answers.
Explanations are assembled from source paragraphs and verified against them.
"""

import hashlib
import re
from functools import lru_cache
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from .evidence import STOP, lexical_tokens

GENERIC = set(
    lexical_tokens(
        "опиши описать объясни объяснение расскажи рассказать такое мне пожалуйста алгоритм "
        "метод модель небольшой небольшое кратко подробно простыми словами выглядит работает "
        "формула формулу формулы обучение обучается обучаются коэффициенты её его это ещё например"
    )
) | set(lexical_tokens(" ".join(STOP)))


def explanation_topic(question, precise=False):
    if precise:
        question = re.split(r"[:;]", question, maxsplit=1)[0]
        question = re.sub(r"\bSVM\b", "метод опорных векторов", question, flags=re.I)
    if not re.search(r"\b(?:опиши|объясни|расскажи|что такое|как работает|формул\w*)\b", question, re.I):
        return None
    if re.search(r"\b(?:сравни\w*|отлич\w*|versus)\b", question, re.I):
        return None
    subject = re.split(r",|;|\bи как\b|\bи для\b|\bеё\b|\bего\b", question, maxsplit=1, flags=re.I)[0]
    terms = set(lexical_tokens(subject)) - GENERIC
    return terms if 1 <= len(terms) <= 5 else None


@lru_cache(maxsize=96)
def sections(content):
    headings = list(re.finditer(r"^(#{1,6})\s+(.+)$", content, re.M))
    result = []
    for index, heading in enumerate(headings):
        level = len(heading[1])
        end = next((h.start() for h in headings[index + 1 :] if len(h[1]) <= level), len(content))
        direct_end = headings[index + 1].start() if index + 1 < len(headings) else len(content)
        introduction = content[heading.end() : direct_end].strip()
        if index + 1 < len(headings) and headings[index + 1].start() < end:
            child = headings[index + 1]
            if re.fullmatch(r"Введение|Определение|Основные определения", child[2], re.I):
                child_end = headings[index + 2].start() if index + 2 < len(headings) else end
                introduction = content[child.end() : min(child_end, end)].strip()
        result.append(
            dict(
                title=heading[2],
                level=level,
                start=heading.end(),
                text=content[heading.end() : end].strip(),
                introduction=introduction,
            )
        )
    return result


def lookup(store, question, precise=False):
    topic = explanation_topic(question, precise)
    if not topic:
        return None
    documents = {c["source"]: c for c in store.all_chunks() if c.get("kind") == "handbook"}
    candidates = []
    for name, metadata in documents.items():
        if name != Path(name).name:
            continue
        path = store.settings.data_dir / "handbook" / name
        if not path.is_file():
            continue
        raw = path.read_bytes()
        # A cache file modified without re-indexing is not trusted as indexed evidence.
        if hashlib.sha256(raw).hexdigest() != metadata["document_id"]:
            continue
        for section in sections(raw.decode("utf-8")):
            if precise and (
                re.match(r"Модель\s*\d|Пример\s*\d", section["title"], re.I)
                or "sklearn" in section["introduction"]
            ):
                continue
            heading_terms = set(lexical_tokens(section["title"]))
            if not topic <= heading_terms or len(section["introduction"].split()) < 10:
                continue
            expanded_heading = set(lexical_tokens(re.sub(r"\([^)]*\)", "", section["title"])))
            extras = {t for t in expanded_heading - topic - GENERIC if not t.isdigit()}
            score = 1 - 0.025 * len(extras)
            if precise:
                if len(heading_terms) > len(topic) + 4:
                    score -= 0.5
                score += 0.15 * bool(re.search(r"\$[^$]*=[^$]*\$", section["text"]))
                score -= 0.8 * bool(
                    re.search(
                        r"sklearn|Обучим на нём|Вам скажут|Но почему",
                        section["title"] + " " + section["introduction"],
                        re.I,
                    )
                )
            if re.search(r"пример|обобщ[её]н|вероятност|байес", section["title"], re.I):
                score -= 0.5
            if len(section["text"].split()) < 40:
                continue
            candidates.append((score, section, metadata))
    if not candidates:
        return None
    _, section, metadata = max(candidates, key=lambda c: c[0])
    return dict(
        metadata,
        id=str(uuid5(NAMESPACE_URL, f"{metadata['document_id']}:section:{section['start']}")),
        page=None,
        text=section["text"],
        section_title=section["title"],
        introduction=section["introduction"],
        teaching_section=True,
        teaching_precise=precise,
        score=1.0,
    )


def paragraphs(text):
    result = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        if not paragraph or paragraph.startswith(("#", "Источник:", "Авторы:", "Автор:", "```", "|")):
            continue
        result.append(paragraph)
    return result


def render(claims):
    blocks = []
    for claim in claims:
        if claim["kind"] == "formula":
            blocks.append(f"**Формула**\n\n$$\n{claim['text']}\n$$\n\n[{claim['citation']}]")
        else:
            blocks.append(f"**{claim['label']}**\n\n{claim['text']} [{claim['citation']}]")
    return "\n\n".join(blocks)


def first_sentences(text, count=2):
    math = False
    parentheses = 0
    for index, char in enumerate(text):
        if char == "$" and (index == 0 or text[index - 1] != "\\"):
            math = not math
        if math:
            continue
        parentheses += int(char == "(") - int(char == ")")
        if char in ".!?" and parentheses <= 0 and (index + 1 == len(text) or text[index + 1].isspace()):
            count -= 1
            if not count:
                return text[: index + 1]
    return text


def explain(context, detail="standard"):
    intro = paragraphs(context["introduction"])
    body = paragraphs(context["text"])
    prose = [p for p in intro if 8 <= len(p.split()) <= 80 and not p.startswith(("$", "-"))]
    if context.get("teaching_precise"):
        prose = [p for p in prose if not re.match(r"В\s+предыдущ|Мы уже|Ранее|Отдельно", p, re.I)]
    if not prose:
        return None

    def definition_score(p):
        return (
            2 * bool(re.search(r"моделир\w* зависим|предсказ|представляет собой|— это|— процедура", p, re.I))
            + bool(re.search(r"называется|позволяет|используется", p, re.I))
            + 2 / (prose.index(p) + 1)
            + (
                3 * len(set(lexical_tokens(context["section_title"])) & set(lexical_tokens(p)))
                if context.get("teaching_precise")
                else 0
            )
            - 3 * bool(re.search(r"^Отдельно заметим|^Для контраста|^достаточно сделать", p, re.I))
        )

    definition = first_sentences(max(prose, key=definition_score))
    claims = [dict(kind="prose", label="Суть", text=definition, citation=1)]
    # Standalone formulas are already delimited in the handbook. Preserve their TeX.
    formulas = []
    for index, p in enumerate(intro):
        if not (
            p.startswith("$")
            and p.endswith("$")
            and p.count("$") == 2
            and 8 <= len(p) <= 240
            and "=" in p
            and "\\begin" not in p
        ):
            continue
        if index + 1 < len(intro) and intro[index + 1].startswith("$") and intro[index + 1].endswith("$"):
            continue  # In a derivation chain, prefer its last equation.
        before = index - 1
        while before >= 0 and intro[before].startswith("$"):
            before -= 1
        anchor = intro[before] if before >= 0 else ""
        score = (
            3
            * bool(
                re.search(r"общий вид|выглядит следующим|каждую следующую|искомую|имеет вид", anchor, re.I)
            )
            + bool(re.search(r"формул|получаем|следующим образом", anchor, re.I))
            - 2 * bool(re.search(r"Тейлор|сходимост|неравенств", anchor, re.I))
        )
        formulas.append((score, p[1:-1].strip().rstrip(".,")))
    if formulas:
        claims.append(
            dict(kind="formula", label="Формула", text=max(formulas, key=lambda x: x[0])[1], citation=1)
        )
    learning = [
        p
        for p in body
        if 10 <= len(p.split()) <= 90
        and p != definition
        and re.search(r"минимиз|обучаем\w* параметр|функци\w* потерь|градиентн\w* спуск", p, re.I)
        and not p.startswith(("#", "-", "$"))
    ]
    if learning:

        def learning_score(p):
            return (
                3 * bool(re.search(r"минимиз", p, re.I))
                + 2 * bool(re.search(r"обучаем|веса|параметр", p, re.I))
                + bool(re.search(r"среднеквадратич|mean squared|MSE", p, re.I))
            )

        selected = first_sentences(max(learning, key=learning_score))
        label = "Как обучается" if re.search(r"обучаем|веса|параметр", selected, re.I) else "Как работает"
        claims.append(dict(kind="prose", label=label, text=selected, citation=1))
    if detail != "short":
        loss = next((p for p in body if "MSE" in p and "называется" in p and len(p.split()) <= 90), None)
        if loss and loss not in [c["text"] for c in claims]:
            claims.append(
                dict(kind="prose", label="Ошибка модели", text=first_sentences(loss, 1), citation=1)
            )
    return dict(
        answer=render(claims),
        claims=claims,
        section=context["section_title"],
        method="handbook_section_extraction",
    )


def verify(answer, explanation, contexts):
    claims = explanation.get("claims", [])
    if not claims or answer != render(claims):
        return False

    def normalize(s):
        return " ".join(s.split())

    return all(
        1 <= c["citation"] <= len(contexts)
        and normalize(c["text"]) in normalize(contexts[c["citation"] - 1]["text"])
        for c in claims
    )
