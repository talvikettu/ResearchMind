"""Generic evidence units and output aspects. Contains no topic-specific answers."""

import re

from .evidence import layout_noise, lexical_tokens, query_terms, reference_block


def output_plan(question, category):
    tail = re.split(r"[:;]|\bпо\b|\bin terms of\b", question, maxsplit=1, flags=re.I)
    requested = []
    if len(tail) == 2:
        requested = [x.strip(" .?!") for x in re.split(r",|;|\bи\b|\band\b", tail[1], flags=re.I)]
        requested = [x for x in requested if 3 <= len(x) <= 80]
    if len(requested) >= 2:
        return requested[:5]
    russian = bool(re.search("[а-яА-Я]", question))
    defaults = {
        "explain": (
            [
                "Определение и идея",
                "Формула и обозначения",
                "Как работает или обучается",
                "Условия и ограничения",
            ],
            ["Definition", "Formula and notation", "How it works", "Assumptions and limitations"],
        ),
        "formula": (
            ["Формула", "Обозначения", "Условия применимости"],
            ["Formula", "Notation", "Assumptions"],
        ),
        "compare": (
            ["Основное различие", "Сопоставление механизмов", "Практические последствия"],
            ["Main difference", "Mechanisms compared", "Practical consequences"],
        ),
        "research": (
            ["Прямой ответ", "Обоснование и механизм", "Условия и ограничения"],
            ["Direct answer", "Mechanism and evidence", "Conditions and limitations"],
        ),
    }
    return defaults[category][0 if russian else 1]


def evidence_units(context, question, max_chars=1800, exclude_exercises=False):
    """Pick whole paragraphs around the retrieved child, preserving exact provenance."""
    text = context["text"]

    def exercise(block):
        if layout_noise(block):
            return True
        if len(block.split()) < 9 and re.match(
            r"Легко видеть|Как видно|Далее получаем|По формуле", block, re.I
        ):
            return True
        return bool(
            re.search(
                r"(?m)^\s*\(?[а-гa-d][).]\s|Вопрос\s+на\s+подумать|Выберите\s+правильн|Контрольн\w*\s+вопрос",
                block,
                re.I,
            )
        ) or bool(re.search(r"sklearn|^Давайте построим|^В\s+предыдущих разделах|^```", block, re.I))

    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) < 2:
        passage = context.get("retrieved_text", text)
        return [] if exclude_exercises and exercise(passage) else [("Source passage", passage)]
    required = query_terms(question)
    anchor = set(lexical_tokens(context.get("retrieved_text", "")))
    ranked = []
    for index, block in enumerate(blocks):
        if block.startswith("#") or reference_block(block) or (len(block.split()) < 4 and "$" not in block):
            continue
        if exclude_exercises and exercise(block):
            continue
        terms = set(lexical_tokens(block))
        value = len(terms & required) / max(1, len(required))
        value += 0.4 * len(terms & anchor) / max(1, len(terms))
        value += 0.03 / (index + 1)
        ranked.append((value, index, block))
    chosen, used = [], 0
    for _, index, block in sorted(ranked, reverse=True):
        if len(block) + used > max_chars and chosen:
            continue
        if len(block) > max_chars:
            continue
        chosen.append((index, block))
        used += len(block)
        if len(chosen) >= 4:
            break
    if not chosen:
        passage = context.get("retrieved_text", text)
        return [] if exclude_exercises and exercise(passage) else [("Retrieved passage", passage)]
    return [("Source paragraph", b) for _, b in sorted(chosen)]
